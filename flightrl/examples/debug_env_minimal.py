#!/usr/bin/env python3
import os
import time
import numpy as np
from ruamel.yaml import YAML, dump, RoundTripDumper
from rpg_baselines.envs import vec_env_wrapper as wrapper
import tensorflow as tf
from rpg_baselines.ppo.ppo2 import PPO2
from rpg_baselines.ppo.ppo2_test import test_model
from flightgym import QuadrotorEnv_v1

yaml = YAML()

def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def make_env():
    # Load vec_env.yaml
    # cfg_path = os.path.join(
    #     os.environ["FLIGHTMARE_PATH"],
    #     "flightlib",
    #     "configs",
    #     "vec_env.yaml",
    # )
    # with open(cfg_path, "r") as f:
    #     cfg = yaml.load(f)
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightlib/configs/vec_env.yaml", 'r'))
    # Force single env, single thread, with rendering
    # cfg["env"]["num_envs"] = 1
    # cfg["env"]["num_threads"] = 1
    # cfg["env"]["render"] = "yes"

    scene_id = cfg["env"].get("scene_id", None)
    print(f"[CONFIG] scene_id = {scene_id}")

    cfg_str = dump(cfg, Dumper=RoundTripDumper)
    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(cfg_str, False))
    configure_random_seed(0, env=env)

    return env


def main():
    env = make_env()
    print("Environment created")
    print("max episode steps:", env.max_episode_steps)

    print("[INFO] Connecting to Unity...")
    # env.connectUnity()

    # time.sleep(2.0)

    # n_envs = env.getNumOfEnvs()
    # obs_dim = env.getObsDim()
    # act_dim = env.getActDim()

    # extra_info_names = env.getExtraInfoNames()
    # extra_info_size = len(extra_info_names)

    # print(f"[INFO] n_envs={n_envs}, obs_dim={obs_dim}, act_dim={act_dim}")
    # print(f"[INFO] extra_info_names={extra_info_names}")

    model = PPO2.load("./saved/quadrotor_env.zip")

    test_model(env, model, render= True)
    
    # Allocate VecEnv buffers
    # obs = np.zeros((n_envs, obs_dim), dtype=np.float32)
    # reward = np.zeros((n_envs, 1), dtype=np.float32)
    # done = np.zeros((n_envs, 1), dtype=bool)
    # extra_info = np.zeros((n_envs, extra_info_size), dtype=np.float32)

    # print("[INFO] Resetting env...")
    # env.reset(obs)
    # print("[DEBUG] obs[0][:8] after reset:", obs[0, :8])

    # # Store initial obs to see if state changes
    # obs0 = obs.copy()

    # # Take ONE image just to inspect its stats
    # print("\n[DEBUG] Requesting one RGB image right after reset...")
    # # img = env.get_rgb_image(0)
    # # print("  img.shape:", img.shape)
    # # print("  img.dtype:", img.dtype)
    # # print("  img[0,0]:", img[0, 0, :])
    # # print("  img.min/max:", img.min(), img.max())

    # # Now step the env with random actions, no saving
    # N_STEPS = 100
    # rng = np.random.default_rng(seed=0)

    # print("\n[INFO] Stepping with random actions...")
    # for i in range(N_STEPS):
    #     # Generic random actions in [-1, 1]
    #     action = rng.uniform(low=-1.0, high=1.0, size=(n_envs, act_dim)).astype(
    #         np.float32
    #     )
    #     env.step(action, obs, reward, done, extra_info)

    #     if i % 10 == 0:
    #         # Check how much obs[:3] moved from initial obs0[:3]
    #         delta = np.linalg.norm(obs[0, :3] - obs0[0, :3])
    #         print(
    #             f"[STEP {i}] reward={reward[0,0]:.4f}, done={done[0,0]}, "
    #             f"||obs[:3] - obs0[:3]|| = {delta:.4f}"
    #         )

    # print("\n[INFO] Done stepping. Disconnecting Unity...")
    # env.disconnectUnity()


if __name__ == "__main__":
    main()
