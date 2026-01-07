import os
import numpy as np
import gym
from collections import deque
from ruamel.yaml import YAML, dump, RoundTripDumper

# Import Flightmare
from flightgym import QuadrotorEnv_v1
from rpg_baselines.envs import vec_env_wrapper as wrapper

CRASH_PENALTY = -4.0
PROGRESS_REWARD_SCALE = 0.5
BODY_RATE_PENALTY_SCALE = 0.02
PERCEPTION_REWARD_SCALE = 0.05
GATE_HALF_WIDTH = 1.0 
PASS_GATE_REWARD = 10.0

class FlightmareDrQWrapper(gym.Env):
    def __init__(self, cfg, stack_frames=3):
        self.cfg = cfg
        
        # 1. Load Gates
        self.gate_data = self._load_gates_from_file()
        self.gate_positions = self.gate_data[:, 0:3] 
        print(f"[Wrapper] Loaded {len(self.gate_positions)} gates")

        # 2. Initialize C++ Environment
        cfg_str = dump(cfg, Dumper=RoundTripDumper)
        self.env = wrapper.FlightEnvVec(QuadrotorEnv_v1(cfg_str, False))

        # 3. Spaces
        # Actions: [collective_thrust, roll_rate, pitch_rate, yaw_rate]
        # DrQv2 outputs actions in [-1, 1] for all dimensions (due to tanh)
        # We remap in step(): thrust [-1,1]→[0,1], body_rates stay [-1,1]
        # Final mapping: collective_thrust: [0, 1] → scaled to [0, max_thrust] in C++
        #                body_rates: [-1, 1] → scaled to [-omega_max, omega_max] in C++
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        print(f"[Wrapper] Action space set to {self.action_space}")
        self.stack_frames = stack_frames
        self.channels = 3 * stack_frames 
        self.observation_space = gym.spaces.Box(0, 255, shape=(self.channels, 84, 84), dtype=np.uint8)
        self.frame_buffer = deque(maxlen=stack_frames)
        
        # Logic Vars
        self.current_gate_idx = 0
        self.prev_dist = 0.0
        self.prev_pos = np.zeros(3)

    def connect_and_warmup(self):
        print("[Wrapper] Connecting to Unity now...")
        self.env.connectUnity()
        print("[Wrapper] Warming up Unity Bridge...")
        # Warmup action: thrust=0 in [-1,1] maps to 0.5 in [0,1] after remapping
        # This gives ~half max thrust for stable warmup
        warmup_action = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        print(f"[Wrapper] Warmup action (pre-remap): {warmup_action}")
        for _ in range(10):
            # Remap: thrust from [-1,1] to [0,1]
            remapped = warmup_action.copy()
            remapped[0, 0] = (warmup_action[0, 0] + 1.0) / 2.0
            print(f"[Wrapper] Warmup action (post-remap): {remapped}")
            self.env.step(remapped)

    def _load_gates_from_file(self):
        fm_path = os.getenv("FLIGHTMARE_PATH")
        if fm_path is None:
            raise RuntimeError("FLIGHTMARE_PATH is not set")
        yaml_path = os.path.join(fm_path, "flightlib/configs/quadrotor_env.yaml")
        try:
            quad_cfg = YAML().load(open(yaml_path, 'r'))
            gates_list = quad_cfg.get("quadrotor_env", {}).get("gates", [])
        except Exception as e:
            print(f"[Wrapper] Error loading gates: {e}")
            return np.zeros((1, 3))

        positions = []
        orientations = []
        for g in gates_list:
            positions.append(np.array(g['pos'], dtype=np.float32))
            orientations.append(g['yaw'])
        if not positions:
            return np.zeros((1, 3))
        return np.hstack((np.array(positions), np.array(orientations).reshape(-1, 1)))

    def reset(self):
        obs_state = self.env.reset()
        quat = obs_state[0, 3:7] # [x, y, z, w]
        # Reset Logic variables
        self.prev_pos = obs_state[0, 0:3]
        self.current_gate_idx = 0
        drone_pos = obs_state[0, 0:3]
        target_pos = self.gate_positions[self.current_gate_idx]
        self.prev_dist = np.linalg.norm(drone_pos - target_pos)
        
        self.frame_buffer.clear()
        
        # Image Acquisition
        img = None
        max_retries = 5
        for attempt in range(max_retries):
            try:
                img = self._get_image_obs()
                if img is not None:
                    break
            except Exception as e:
                self.env.step(np.zeros((1, 4), dtype=np.float32))
                
        if img is None:
            raise RuntimeError("CRITICAL: Unity failed to render after reset!")

        for _ in range(self.stack_frames):
            self.frame_buffer.append(img)
            
        return self._get_stacked_obs()

    def step(self, action):
        # 1. Remap action from [-1, 1] (DrQv2 output) to [0, 1] for thrust, [-1, 1] for body rates
        # DrQv2 actor outputs actions in [-1, 1] due to tanh squashing
        # We need: thrust ∈ [0, 1], body_rates ∈ [-1, 1]
        remapped_action = action.copy()
        remapped_action[0] = (action[0] + 1.0) / 2.0  # [-1, 1] → [0, 1] for thrust
        # body rates [1:4] already in [-1, 1], no change needed
        
        action_vec = np.expand_dims(remapped_action, axis=0) 
        
        # DEBUG: Log first few steps
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        if self._step_count < 5:
            print(f"[Step {self._step_count}] Raw action: {action}, Remapped: {remapped_action}")
        self._step_count += 1
        
        obs_state, raw_rewards, cpp_done, info = self.env.step(action_vec)
        cpp_crash = cpp_done[0]
        
        full_state = obs_state[0] # Shape: (13,) changed euler to quat

        drone_pos = full_state[0:3]   # [x, y, z]
        # drone_quat = full_state[3:7]  # [x, y, z, w]
        drone_vel = full_state[7:10]   # [vx, vy, vz]
        body_rates = full_state[10:13] # [wx, wy, wz]

        # DEBUG: Log state in first few steps
        if self._step_count <= 5:
            print(f"[Step {self._step_count}] Pos: {drone_pos}, Vel: {drone_vel}")

        qx, qy, qz, qw = full_state[3:7] # [x, y, z, w]



        # Reconstruct quaternion vector for logic [x, y, z, w]
        # (This matches Flightmare/Eigen convention used in your helpers)
        drone_quat = np.array([qx, qy, qz, qw])


        # 3. Calculate Reward Components
        target_pos = self.gate_positions[self.current_gate_idx]
        target_yaw = self.gate_data[self.current_gate_idx, 3]
        curr_dist = np.linalg.norm(drone_pos - target_pos)

        # A. Progress
        progress = np.clip(self.prev_dist - curr_dist, -1.0, 1.0)
        
        # B. Perception 
        r_perc, raw_dot_prod = self._compute_perception_components(drone_pos, drone_quat, target_pos)
        
        # C. Penalty
        high_body_rates_penalty = np.linalg.norm(body_rates)

        # D. Total Reward
        reward = (PROGRESS_REWARD_SCALE * progress) + \
                 (PERCEPTION_REWARD_SCALE * r_perc) - \
                 (BODY_RATE_PENALTY_SCALE * high_body_rates_penalty)
        
        # Gate Logic
        gate_passed = False
        if self._check_gate_pass(self.prev_pos, drone_pos, target_pos, target_yaw):
            reward += PASS_GATE_REWARD
            gate_passed = True
            print(f"Gate {self.current_gate_idx + 1} PASSED! (Robust Check)")
            
            self.current_gate_idx += 1
            if self.current_gate_idx >= len(self.gate_positions):
                self.current_gate_idx = 0 
            
            new_target = self.gate_positions[self.current_gate_idx]
            self.prev_dist = np.linalg.norm(drone_pos - new_target)
        else:
            self.prev_dist = curr_dist
        
        #  Check Crash
        if cpp_crash:
            total_reward = reward + CRASH_PENALTY
            
            info = {
                "action": remapped_action,  # Save the REMAPPED action
                "gate_passed": gate_passed,         # True
                "gate_idx": self.current_gate_idx,  # The gate we just passed
                "collision": True
            }
            
            return self._get_stacked_obs(), total_reward, True, info
        
        
        self.prev_pos = drone_pos.copy()

        # Visuals & Info
        img = self._get_image_obs()
        self.frame_buffer.append(img)
        
        info = {
            "pos": drone_pos, 
            "vel": drone_vel,
            "quat": drone_quat,
            "action": remapped_action,  # Save the REMAPPED action that was actually executed
            "gate_idx": self.current_gate_idx,
            "gate_passed": gate_passed,
            "dist_to_gate": curr_dist,
            "look_dot_prod": raw_dot_prod, 
            "rew_total": reward,
            "rew_progress": progress,
            "rew_perception": r_perc * PERCEPTION_REWARD_SCALE,
            "rew_penalty": high_body_rates_penalty * BODY_RATE_PENALTY_SCALE
        }

        return self._get_stacked_obs(), reward, False, info


    ## --- Helper Methods --- ##

    def _get_image_obs(self):
        img = self.env.get_rgb_image(0)
        img_rgb = img[..., ::-1]
        return np.transpose(img_rgb, (2, 0, 1)) 

    def _get_stacked_obs(self):
        return np.concatenate(list(self.frame_buffer), axis=0)

    # ---  With this setup, we initialize the drone at yaw=0 but it is looking on the +y. 
    # So when we initialize the camera to look to the same direction with the drones nose it is returning 
    # images from +y but thinks it is on +X, yaw = 0. ---
    def _compute_perception_components(self, drone_pos, drone_quat, target_pos):
        target_vector = target_pos - drone_pos
        dist_to_gate = np.linalg.norm(target_vector)
        
        if dist_to_gate < 0.2: return 1.0, 1.0
        
        target_vector_norm = target_vector / dist_to_gate
        
        # 30 deg Camera Tilt
        angle = 0.5236 
        camera_ray_body = np.array([0.0, np.cos(angle), np.sin(angle)]) 
        camera_vector_world = self._rotate_vector_by_quaternion(camera_ray_body, drone_quat)

        dot_prod = np.dot(target_vector_norm, camera_vector_world)
        r_perc = (dot_prod + 1.0) / 2.0 
        # --- LOGGING TO CONFIRM ALIGNMENT ---
        # Add this momentarily to verify "Target" and "Camera" are pointing the same way
        # print(f"Target: {target_vector_norm} | Camera: {camera_vector_world} | Dot: {dot_prod:.3f}")
        return r_perc, dot_prod

    def _check_gate_pass(self, prev_pos, curr_pos, gate_pos, gate_yaw):
        gate_normal = np.array([np.cos(gate_yaw), np.sin(gate_yaw), 0.0])
        vec_prev = prev_pos - gate_pos
        vec_curr = curr_pos - gate_pos
        dist_prev = np.dot(vec_prev, gate_normal)
        dist_curr = np.dot(vec_curr, gate_normal)

        if not (np.sign(dist_prev) != np.sign(dist_curr)):
            return False

        total_dist_change = dist_curr - dist_prev 
        t = (0 - dist_prev) / total_dist_change
        intersection_point = prev_pos + t * (curr_pos - prev_pos)
        
        vec_intersect = intersection_point - gate_pos
        vec_in_gate_plane = vec_intersect - np.dot(vec_intersect, gate_normal) * gate_normal
        lateral_dist = np.linalg.norm(vec_in_gate_plane[0:2])
        vertical_dist = np.abs(vec_in_gate_plane[2])

        if lateral_dist <= GATE_HALF_WIDTH and vertical_dist <= GATE_HALF_WIDTH:
            return True
        return False

    def _rotate_vector_by_quaternion(self, v, q):
        vx, vy, vz = v
        qx, qy, qz, qw = q
        x_new = (1 - 2*qy*qy - 2*qz*qz)*vx + (2*qx*qy - 2*qz*qw)*vy + (2*qx*qz + 2*qy*qw)*vz
        y_new = (2*qx*qy + 2*qz*qw)*vx + (1 - 2*qx*qx - 2*qz*qz)*vy + (2*qy*qz - 2*qx*qw)*vz
        z_new = (2*qx*qz - 2*qy*qw)*vx + (2*qy*qz + 2*qx*qw)*vy + (1 - 2*qx*qx - 2*qy*qy)*vz
        return np.array([x_new, y_new, z_new])
    
    def _convert_euler_to_quaternion(self, yaw, pitch, roll):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return np.array([qx, qy, qz, qw])  # [x, y, z, w] format
    
    