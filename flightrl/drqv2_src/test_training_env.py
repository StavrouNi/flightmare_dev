import os
import numpy as np
import cv2
from ruamel.yaml import YAML
from wrapper import FlightmareDrQWrapper

def main():
    # 1. Setup
    fm_path = os.getenv("FLIGHTMARE_PATH")
    cfg_path = os.path.join(fm_path, "flightlib/configs/vec_env.yaml")
    cfg = YAML().load(open(cfg_path, "r"))
    
    # Force 1 drone for clarity during testing
    cfg["env"]["num_envs"] = 1
    
    env = FlightmareDrQWrapper(cfg, stack_frames=1) # Stack 1 for easier visual debug
    env.connect_and_warmup()
    obs = env.reset()
    
    print("\n=== TEST 1: VISUAL CHECK ===")
    # Save the first frame to check if camera works
    # Obs shape is (3, 84, 84) -> Transpose to (84, 84, 3) for saving
    img = np.transpose(obs[:3], (1, 2, 0)) 
    # RGB to BGR for OpenCV saving
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("debug_test_view.png", img_bgr)
    print("📸 Saved 'debug_test_view.png'. Check it manually! Is it black?")

    print("\n=== TEST 2: PHYSICS (THRUST) CHECK ===")
    env.reset()
    # Action: [Thrust=1.0, 0, 0, 0]
    # action = np.zeros((1, 4), dtype=np.float32)
    
    zero_action = np.zeros((4,), dtype=np.float32)
    _, _, _, info = env.step(zero_action)
    initial_z = info['pos'][2] # Get Z directly from C++ wrapper if possible, or info

    action = np.zeros((4,), dtype=np.float32)
    action[0] = 1.0 
    for i in range(10):
        _, _, _, info = env.step(action)
    
    final_z = info['pos'][2]
    print(f"Initial Z: {initial_z:.2f} | Final Z (after 10 steps max thrust): {final_z:.2f}")
    if final_z > initial_z + 0.05: # Expect some ascent
        print("✅ PASS: Drone ascended.")
    else:
        print("❌ FAIL: Drone did not ascend significantly. Check physics config.")

    print("\n=== TEST 3: REWARD LOGIC CHECK ===")
    env.reset()
    
    # A. Stationary (Should be near 0 or slightly negative due to existence penalty/noise)
    action_hover = np.zeros((4,), dtype=np.float32)
    _, r_hover, _, _ = env.step(action_hover)
    print(f"Reward (Hover): {r_hover:.4f}")
    
    # B. High Body Rates (Should be negative penalty)
    action_spin = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32) # Max rotation
    _, r_spin, _, _ = env.step(action_spin)
    print(f"Reward (Max Spin): {r_spin:.4f}")
    
    if r_spin < r_hover:
        print("✅ PASS: Spinning is penalized more than hovering.")
    else:
        print("❌ FAIL: Penalty logic might be inverted or weak.")

    print("\n=== TEST 4: CRASH & RESET CHECK ===")
    env.reset()
    print("Flying down to crash...")
    
    # Fly down implies low thrust (or negative if allowed, but usually just 0 thrust + gravity)
    # Actually, we can just teleport it in logic or wait. 
    # Let's wait: Action = -1.0 thrust (min thrust)
    action_drop = np.ones((4,), dtype=np.float32) * -1.0
    
    for i in range(100):
        obs, reward, done, info = env.step(action_drop)
        z_pos = info['pos'][2]
        
        if done:
            print(f"💥 CRASH DETECTED at Step {i}!")
            print(f"   - Height at crash frame: {z_pos:.3f}")
            print(f"   - Reward received: {reward} (Should be -4.0)")
            print(f"   - C++ Done Flag: {done}")
            
            # Check NEXT step to see if it reset
            # We send a dummy action to see the new state
            _, _, _, info_next = env.step(action_drop)
            z_next = info_next['pos'][2]
            print(f"   - Position AFTER crash: {z_next:.3f} (Should be ~2.5)")
            
            if abs(reward + 4.0) < 0.001:
                print("✅ PASS: Reward is -4.0")
            else:
                print(f"❌ FAIL: Reward was {reward}")
                
            if z_next > 1.0:
                print("✅ PASS: Drone auto-reset to start height.")
            else:
                print(f"❌ FAIL: Drone is still on the ground at {z_next:.3f}")
            break
            
    env.close()

if __name__ == "__main__":
    main()