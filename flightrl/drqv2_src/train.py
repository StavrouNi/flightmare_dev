import os
import time
import ctypes
try:
    ctypes.cdll.LoadLibrary('/usr/lib/x86_64-linux-gnu/libcuda.so.1')
except OSError:
    print("WARNING: Could not manually load libcuda.so.1")
import torch
from ruamel.yaml import YAML

# Import your wrapper
from wrapper import FlightmareDrQWrapper

# We will create these two files next!
from drqv2 import DrQV2Agent
from replay_buffer import ReplayBuffer

def main():
    # 1. Setup Paths
    fm_path = os.getenv("FLIGHTMARE_PATH")
    if fm_path is None:
        raise RuntimeError("FLIGHTMARE_PATH is not set")
    
    cfg_path = os.path.join(fm_path, "flightlib/configs/vec_env.yaml")
    cfg = YAML().load(open(cfg_path, "r"))

    # 2. Initialize Environment (The "Sandwich")
    # This loads Unity, Physics, and the Butterfly Track logic
    env = FlightmareDrQWrapper(cfg, stack_frames=3)
    
    # 3. Setup Hyperparameters
    # DrQ-v2 standard settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    action_shape = env.action_space.shape
    obs_shape = env.observation_space.shape # (9, 84, 84)
    
    # 4. Initialize Agent (DrQ-v2)
    agent = DrQV2Agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        lr=1e-4,
        feature_dim=50,
        hidden_dim=1024,
        batch_size=256
    )

    # 5. Initialize Replay Buffer (Stores 100k frames)
    # We store frames in RAM. 100k * 64*64*3 bytes is ~1.2GB. Safe.
    replay_buffer = ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=action_shape,
        capacity=100000,
        batch_size=256,
        device=device
    )

    # 6. NOW Connect to Unity (Safe!) 🤝
    print("4. Connecting to Unity...")
    env.connect_and_warmup()

    # 7. Training Loop
    print("5. Starting Training Loop!")
    step = 0
    episode = 0
    max_steps = 1000000 # 1 Million steps for initial trial
    
    # Start the first episode
    obs = env.reset()
    
    while step < max_steps:
        # time.sleep(0.005)
        
        # A. Select Action
        # Random exploration for the first 2000 steps to fill buffer
        if step < 2000:
            action = env.action_space.sample()
        else:
            # Add noise for exploration during training
            with torch.no_grad():
                action = agent.act(obs, step, eval_mode=False)

        # --- LOG 1: Before Step ---

        # B. Step Environment
        print(f"[Train] Step {step}: Taking action {action}")
        t0 = time.time()
        next_obs, reward, done, info = env.step(action)
        dt_step = time.time() - t0
        
        # --- LOG 2: Check Step Duration ---
        if dt_step > 0.001: # Only print if it's suspiciously slow (>100ms)
            print(f"[Train] Step {step} took {dt_step:.4f}s (Slow!)")
        # C. Add to Buffer
        # Note: We don't verify 'done' mask here because racing is continuous
        # until crash.
        replay_buffer.add(obs, action, reward, next_obs, done)
        
        obs = next_obs
        step += 1

        # D. Update Agent
        if step >= 200:
            agent.update(replay_buffer, step)

        # E. Logging / Reset
        if done:
            print(f"[Train] CRASH/DONE detected at Step {step}. Reward: {reward:.2f}")

            obs = env.reset()
            
            episode += 1
            
        # Optional: Save weights every 50k steps
        if step % 50000 == 0:
            torch.save(agent.state_dict(), f"agent_step_{step}.pt")
            print(f"Saved weights at step {step}")

    # Done
    env.close()
    print("Training Finished!")

if __name__ == "__main__":
    main()