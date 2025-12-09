import os
import numpy as np
import gym
from collections import deque
from ruamel.yaml import YAML, dump, RoundTripDumper
import cv2

# Import Flightmare
from flightgym import QuadrotorEnv_v1
from rpg_baselines.envs import vec_env_wrapper as wrapper

CRASH_PENALTY = -4.0

class FlightmareDrQWrapper(gym.Env):
    def __init__(self, cfg, stack_frames=3):
        """
        Wrapper for DrQ-v2.
        :param cfg: Dictionary loaded from vec_env.yaml (passed from train.py)
        """
        self.cfg = cfg
        
        # 1. Load Gates directly from 'quadrotor_env.yaml' file
        # "flightlib/configs/quadrotor_env.yaml" Is the environment file we want here, not env.yaml
        self.gate_positions = self._load_gates_from_file()
        print(f"[Wrapper] Loaded {len(self.gate_positions)} gates from quadrotor_env.yaml")

        # 2. Initialize C++ Environment
        # This uses the vec_env.yaml config passed from main()
        cfg_str = dump(cfg, Dumper=RoundTripDumper)
        self.env = wrapper.FlightEnvVec(QuadrotorEnv_v1(cfg_str, False))

        # spaces, buffer, logic vars ...
        self.action_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.stack_frames = stack_frames
        self.channels = 3 * stack_frames 
        self.observation_space = gym.spaces.Box(0, 255, shape=(self.channels, 84, 84), dtype=np.uint8)
        self.frame_buffer = deque(maxlen=stack_frames)
        self.current_gate_idx = 0
        self.gate_radius = 0.75  # Meters ??
        self.prev_dist = 0.0

        self.current_step = 0  # For debugging
    def connect_and_warmup(self):
            """Connects to Unity only when we are actually ready."""
            print("[Wrapper] Connecting to Unity now...")
            self.env.connectUnity()
            
            print("[Wrapper] Warming up Unity Bridge...")
            warmup_action = np.zeros((1, 4), dtype=np.float32)
            for _ in range(10):
                self.env.step(warmup_action)

    def _load_gates_from_file(self):
        """Parses quadrotor_env.yaml to find gate positions."""
        fm_path = os.getenv("FLIGHTMARE_PATH")
        if fm_path is None:
            # Fallback if env var is missing, though train.py usually checks this
            raise RuntimeError("FLIGHTMARE_PATH is not set")
            
        # Construct path to the PHYSICS config (where gates live)
        yaml_path = os.path.join(fm_path, "flightlib/configs/quadrotor_env.yaml")
        
        try:
            quad_cfg = YAML().load(open(yaml_path, 'r'))
            # Access the nested list: quadrotor_env -> gates
            gates_list = quad_cfg.get("quadrotor_env", {}).get("gates", [])
        except Exception as e:
            print(f"[Wrapper] Error loading gates from {yaml_path}: {e}")
            return np.zeros((1, 3)) # Safe fallback

        positions = []
        for g in gates_list:
            # g['pos'] is a list [x, y, z] in the yaml
            positions.append(np.array(g['pos'], dtype=np.float32))
        
        if not positions:
            print("[Wrapper] WARNING: No gates found in quadrotor_env.yaml!")
            return np.zeros((1, 3))
            
        return np.array(positions)

    def reset(self):
            # print("  [Wrapper] Resetting Physics...")
            obs_state = self.env.reset()
            
            # Reset Logic variables
            self.current_gate_idx = 0
            drone_pos = obs_state[0, 0:3]
            target_pos = self.gate_positions[self.current_gate_idx]
            self.prev_dist = np.linalg.norm(drone_pos - target_pos)
            
            self.frame_buffer.clear()
            
            # Image Acquisition with retry logic because unity might initialize after 
            # we do a zero RL step to wait rendering
            img = None
            max_retries = 5
            
            for attempt in range(max_retries):
                try:
                    # Attempt to get the image
                    img = self._get_image_obs()
                    if img is not None:
                        print(f"  [Wrapper] Reset Image acquired on attempt {attempt+1}")
                        break
                except Exception as e:
                    # If it failed (empty image), step the sim to force a render
                    zero_action = np.zeros((1, 4), dtype=np.float32)
                    self.env.step(zero_action)
                    
            if img is None:
                raise RuntimeError("CRITICAL: Unity failed to render after reset!")

            # Fill buffer with the valid image
            for _ in range(self.stack_frames):
                self.frame_buffer.append(img)
                
            return self._get_stacked_obs()

    def step(self, action):
        # Step Physics
        action_vec = np.expand_dims(action, axis=0) 
        obs_state, raw_rewards, cpp_done, info = self.env.step(action_vec)
        cpp_crash = cpp_done[0]
        # Log the critical output before Python processes it
        # Assuming full_state = obs_state[0] is already extracted
        full_state = obs_state[0] 
        V_linear = full_state[6:9]
        linear_speed = np.linalg.norm(V_linear)
        roll_angle = full_state[5]  # Index 5 (Rad)
        pitch_angle = full_state[4] # Index 4 (Rad)

        # --- Add to the existing print block ---
        print(f"--- DEBUG STEP {self.current_step} ---")
        x_pos = full_state[0]
        y_pos = full_state[1]
        z_pos = full_state[2]

        print(f"drone full STATE: X={x_pos:.4f}, Y={y_pos:.4f}, Z={z_pos:.4f} ...")
        print(f"SPEED: {linear_speed:.2f} m/s")
        print(f"ROLL/PITCH: {roll_angle:.2f} / {pitch_angle:.2f} rad")
        print(f"RAW C++ DONE: {cpp_done[0]}")
        print(f"RAW C++ REWARD: {raw_rewards[0]:.4f}")
        # ---------------------------------------
        self.current_step += 1 # DEBUGGING
        # Reward
        full_state = obs_state[0]
        reward, done = self._compute_racing_reward(full_state, action, cpp_crash)
        # Update Visuals
        img = self._get_image_obs()
        self.frame_buffer.append(img)

        drone_pos = obs_state[0, 0:3]
    
        info = {
            "pos": drone_pos, 
            "gate_idx": self.current_gate_idx,
            "gate_passed": False # You can update this in _compute_racing_reward if a gate is passed
        }

        return self._get_stacked_obs(), reward, done, info

    def _get_image_obs(self):
        img = self.env.get_rgb_image(0)
        # 2. Check and Fix Colors
        # Flightmare/OpenCV returns BGR. DrQ-v2. We will change to RGB for sim to real
        img_rgb = img[..., ::-1]
        # Flip to (3, H, W) for PyTorch
        return np.transpose(img_rgb, (2, 0, 1)) 

    def _get_stacked_obs(self):
        return np.concatenate(list(self.frame_buffer), axis=0)

    def _compute_racing_reward(self, full_state, action, cpp_done):

        drone_pos = full_state[0:3]
        body_rates = full_state[9:12]
        if cpp_done:
            target_pos = self.gate_positions[self.current_gate_idx]
            self.prev_dist = np.linalg.norm(drone_pos - target_pos)
            return CRASH_PENALTY, True
        
        target_pos = self.gate_positions[self.current_gate_idx]
        curr_dist = np.linalg.norm(drone_pos - target_pos)
        print(f"Distance to Gate {self.current_gate_idx}: {curr_dist:.4f} m")
        # Progress Reward
        progress = (self.prev_dist - curr_dist)

        # Large Body rates action penalty
        high_body_rates_penalty = 0.01 * np.linalg.norm(body_rates)

        # Reward
        reward = 1.0 * progress - high_body_rates_penalty
        
        self.prev_dist = curr_dist
        
        # Gate Completion
        if curr_dist < self.gate_radius:
            reward += 10.0
            print(f"Gate {self.current_gate_idx + 1} PASSED!")
            
            self.current_gate_idx += 1
            if self.current_gate_idx >= len(self.gate_positions):
                self.current_gate_idx = 0 # Loop
                
            # Update distance metric for new target
            new_target = self.gate_positions[self.current_gate_idx]
            self.prev_dist = np.linalg.norm(drone_pos - new_target)
            
            
        # 4. Python-Side Bounds Check (Ceiling/Walls)
        # We can still check ceiling here since C++ might only check floor
        if (drone_pos[2] > 8.0 or 
            np.abs(drone_pos[0]) > 20.0 or np.abs(drone_pos[1]) > 20.0):
            return CRASH_PENALTY, True

        return reward, False
            