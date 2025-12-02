#!/usr/bin/env python3
import os
import time
import numpy as np
from ruamel.yaml import YAML, dump, RoundTripDumper
import imageio

from flightgym import QuadrotorEnv_v1

yaml = YAML()


def make_env():
    cfg_path = os.path.join(
        os.environ["FLIGHTMARE_PATH"],
        "flightlib",
        "configs",
        "vec_env.yaml",
    )
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f)

    # single env, single thread, with rendering
    cfg["env"]["num_envs"] = 1
    cfg["env"]["num_threads"] = 1
    cfg["env"]["render"] = "yes"

    scene_id = cfg["env"].get("scene_id", None)
    print(f"[CONFIG] scene_id from vec_env.yaml: {scene_id}")

    cfg_str = dump(cfg, Dumper=RoundTripDumper)
    env = QuadrotorEnv_v1(cfg_str, False)
    return env


def main():
    env = make_env()
    print("Connecting to Unity...")
    env.connectUnity()
    time.sleep(2.0)

    n_envs = env.getNumOfEnvs()
    obs_dim = env.getObsDim()
    act_dim = env.getActDim()
    extra_info_names = env.getExtraInfoNames()
    extra_info_size = len(extra_info_names)

    print(f"n_envs={n_envs}, obs_dim={obs_dim}, act_dim={act_dim}")
    print("extra_info_names:", extra_info_names)

    # VecEnv expects 2D arrays
    obs = np.zeros((n_envs, obs_dim), dtype=np.float32)
    reward = np.zeros((n_envs, 1), dtype=np.float32)
    done = np.zeros((n_envs, 1), dtype=bool)
    extra_info = np.zeros((n_envs, extra_info_size), dtype=np.float32)

    print("Resetting env...")
    env.reset(obs)
    print("[DEBUG] obs[0][:8] after reset:", obs[0, :8])

    # Save initial obs as reference to see if anything moves
    obs0 = obs.copy()

    # Folder for saved frames
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)

    N_STEPS = 300
    print("Running simulation with random actions...")

    rng = np.random.default_rng(seed=0)

    for i in range(N_STEPS):
        # --- GENERIC RANDOM ACTIONS ---
        # Most Flightmare envs expect actions in [-1, 1] per dimension.
        # This will cause movement regardless of specific control_mode.
        action = rng.uniform(low=-1.0, high=1.0, size=(n_envs, act_dim)).astype(
            np.float32
        )
        env.step(action, obs, reward, done, extra_info)

        if i % 10 == 0:
            # Approximate "position change" by looking at first 3 obs entries.
            # In most configs these are position or some function of it.
            delta = np.linalg.norm(obs[0, :3] - obs0[0, :3])
            print(
                f"[STEP {i}] reward={reward[0,0]:.4f}, done={done[0,0]}, "
                f"||obs[:3] - obs0[:3]|| = {delta:.4f}"
            )

        # Save RGB image every 5 steps
        if i % 5 == 0:
            print(f"\nStep {i}: requesting RGB image...")
            img = env.get_rgb_image(0)

            print("  shape:", img.shape)
            print("  dtype:", img.dtype)
            print("  first pixel:", img[0, 0, :])
            print("  min/max:", img.min(), img.max())

            fname = os.path.join(frames_dir, f"frame_{i:03d}.png")
            imageio.imwrite(fname, img)
            print(f"  saved {fname}")

    print("RGB camera exploration finished. Disconnecting Unity...")
    env.disconnectUnity()


if __name__ == "__main__":
    main()
