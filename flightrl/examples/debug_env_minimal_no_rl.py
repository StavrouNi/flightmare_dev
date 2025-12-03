import os
import numpy as np
import imageio
from ruamel.yaml import YAML, dump, RoundTripDumper
from flightgym import QuadrotorEnv_v1
from rpg_baselines.envs import vec_env_wrapper as wrapper


def main():
    fm_path = os.getenv("FLIGHTMARE_PATH")
    if fm_path is None:
        raise RuntimeError("FLIGHTMARE_PATH is not set")

    # Load vec_env config
    cfg = YAML().load(open(
        os.path.join(fm_path, "flightlib/configs/vec_env.yaml"), "r"
    ))
    cfg_str = dump(cfg, Dumper=RoundTripDumper)

    # 1 env
    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(cfg_str, False))

    env.connectUnity()
    obs = env.reset()

    print("Connecting to Unity...")
    os.makedirs("frames", exist_ok=True)

    n_steps = 250
    save_every = 10
    ramp_up_steps = 30  # gradual thrust increase

    # Actions MUST be in [-1, 1] range (normalized)
    # act_mean = mass*g/4 ≈ 1.79N, act_std = mass*2g/4 ≈ 3.58N
    # normalized_action = (actual_thrust - act_mean) / act_std
    # 
    # Examples:
    #   0.0 = hover (1.79N per rotor)
    #   1.0 = max thrust (1.79 + 3.58 = 5.37N per rotor)
    #  -1.0 = min thrust (1.79 - 3.58 = -1.79N, motors off)
    #
    # For roll left: increase motors 0,1, decrease motors 2,3
    hover = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    roll_left = np.array([0.25, 0.25, -0.25, -0.25], dtype=np.float32)
    
    base_action = hover + roll_left  # combined command

    last_pos = None
    print("Starting flight with gradual motor ramp-up...")

    for t in range(n_steps):
        # Gradual ramp-up for first 30 steps to avoid sudden jump
        if t < ramp_up_steps:
            ramp_factor = t / ramp_up_steps
            current_action = base_action * ramp_factor
        else:
            current_action = base_action
        
        # (num_envs=1, act_dim=4)
        action = np.zeros((1, 4), dtype=np.float32)
        action[0] = current_action

        obs, rew, done, info = env.step(action)

        # Debug: print position from obs to see if physics changes
        # Assuming obs[0, 0:3] is position (POSX, POSY, POSZ)
        pos = obs[0, 0:3]
        if t == 0:
            print(f"t={t}, pos={pos}")
        elif t % 10 == 0:
            print(f"t={t}, pos={pos} (Δ={pos - last_pos})")

        last_pos = pos.copy()

        if t % save_every == 0:
            img = env.get_rgb_image(0)
            if img is not None and img.size > 0:
                if img.dtype != np.uint8:
                    img8 = np.clip(img, 0, 255).astype(np.uint8)
                else:
                    img8 = img
                filename = f"frames/frame_{t:06d}.png"
                imageio.imwrite(filename, img8)
                print(f"[{t}] saved {filename}")
            else:
                print(f"[{t}] WARNING: empty image")

        if np.any(done):
            print(f"Episode finished at step {t}")
            obs = env.reset()
            last_pos = None  # reset reference

    env.close()


if __name__ == "__main__":
    main()
