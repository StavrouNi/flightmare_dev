import os
import argparse
import re
import time
import torch
import numpy as np
import cv2
from ruamel.yaml import YAML
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Custom modules
from logger_utils import DebugLogger
from wrapper import FlightmareDrQWrapper
from drqv2 import DrQV2Agent
from replay_buffer import ReplayBuffer

# --- Constants & Hyperparameters ---
WARMUP_STEPS = 2000         
TOTAL_STEPS = 10_000_000     
SAVE_INTERVAL = 20_000      
LOG_INTERVAL = 200         
BUFFER_SIZE = 50_000
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
VIDEO_INTERVAL = 500  # Save video every 50 episodes

def parse_args():
    parser = argparse.ArgumentParser(description='Train DrQ-v2 on Flightmare')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint file')
    parser.add_argument('--seed', type=int, default=3, help='Random seed')
    return parser.parse_args()

def setup_environment(seed):
    fm_path = os.getenv("FLIGHTMARE_PATH")
    if fm_path is None:
        raise RuntimeError("FLIGHTMARE_PATH is not set.")
    
    cfg_path = os.path.join(fm_path, "flightlib/configs/vec_env.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found at {cfg_path}")

    print(f"[Setup] Loading config from {cfg_path}")
    cfg = YAML().load(open(cfg_path, "r"))
    
    # Initialize Wrapper (Manual video saving, so no RecordVideo wrapper)
    env = FlightmareDrQWrapper(cfg, stack_frames=3)
    env.seed(seed)
    
    return env

def setup_agent_and_buffer(env, device):
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    agent = DrQV2Agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        lr=LEARNING_RATE,
        feature_dim=50,
        hidden_dim=1024,
        batch_size=BATCH_SIZE,
        num_expl_steps=2000,
        update_every_steps=2,
        stddev_schedule='linear(1.0,0.1,100000)',
        stddev_clip=0.3
    )
    replay_buffer = ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=action_shape,
        capacity=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        device=device,
        n_step=3,
        discount=0.99
    )

    return agent, replay_buffer

def load_checkpoint_if_available(agent, resume_path):
    start_step = 0
    if resume_path and os.path.exists(resume_path):
        print(f"[Train] 🔄 Resuming from: {resume_path}")
        agent.load(resume_path)
        match = re.search(r"step_(\d+)", resume_path)
        if match:
            start_step = int(match.group(1))
            print(f"[Train] ⏭️  Fast-forwarding to step {start_step}")
        else:
            print("[Train] ⚠️ Checkpoint loaded, starting at 0.")
    elif resume_path:
        print(f"[Train] ❌ Checkpoint not found. Starting fresh.")
    else:
        print("[Train] ✨ Starting fresh training.")
    return start_step

# --- HELPER FUNCTIONS ---

def setup_logging_dirs(run_dir):
    """Creates separate directories for .npy trajectory files and plots."""
    npy_dir = os.path.join(run_dir, "npy_trajectories")
    os.makedirs(npy_dir, exist_ok=True)
    
    plot_dir = os.path.join(run_dir, "trajectories") 
    os.makedirs(plot_dir, exist_ok=True)
    
    return npy_dir, plot_dir

def select_action(agent, env, obs, global_step):
    if global_step < WARMUP_STEPS:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return agent.act(obs, global_step, eval_mode=False)

def log_tensorboard_metrics(writer, reward, length, episode, global_step):
    writer.add_scalar("Train_Steps/Episode_Reward", reward, global_step)
    writer.add_scalar("Train_Steps/Episode_Length", length, global_step)
    writer.add_scalar("Train/Episode_Reward", reward, episode)
    writer.add_scalar("Train/Episode_Length", length, episode)

def save_checkpoint(agent, global_step, run_dir):
    filename = f"agent_step_{global_step}.pt"
    save_path = os.path.join(run_dir, "checkpoints", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True) 
    agent.save(save_path)
    return save_path

def save_full_state_trajectory(full_state_buffer, episode_count, npy_dir):
    """Saves the complete episode state (drone and gates) to a .npy file."""
    if not full_state_buffer:
        return
    
    filename = f"full_state_ep_{episode_count:05d}.npy"
    save_path = os.path.join(npy_dir, filename)
    
    np.save(save_path, np.array(full_state_buffer)) 
    print(f"[Train] 💾 Saved full state trajectory to: {save_path}")

def save_episode_video(frames, episode_count, run_dir):
    """Saves frames as an MP4 video using OpenCV."""
    if not frames:
        return

    video_dir = os.path.join(run_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    save_path = os.path.join(video_dir, f"episode_{episode_count}.mp4")

    # Frame is (3, H, W). Need (H, W, 3).
    _, height, width = frames[0].shape
    
    # 'mp4v' is generally safe. If it fails, try 'avc1'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))

    for frame in frames:
        # Transpose (C, H, W) -> (H, W, C)
        frame_hwc = np.transpose(frame, (1, 2, 0))
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_hwc, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"[Train] 🎥 Video saved: {save_path}")

# --- MAIN LOOP ---

def run_training_loop(env, agent, replay_buffer, writer, debugger, start_step, run_dir, npy_dir, gate_positions_for_save):
    
    global_step = start_step
    episode_reward = 0
    episode_step = 0
    episode_count = 0

    # Buffers
    full_state_buffer = []  
    current_episode_frames = []
    
    obs = env.reset()
    pbar = tqdm(total=TOTAL_STEPS, initial=start_step, desc="Training", unit="step")

    while global_step < TOTAL_STEPS:
        
        # --- 1. Frame Capture (Optimization: Only if this ep is a video ep) ---
        if episode_count % VIDEO_INTERVAL == 0:
            # obs is (9, H, W), take last 3 channels for current frame
            frame = obs[-3:, :, :].copy()
            current_episode_frames.append(frame)

        action = select_action(agent, env, obs, global_step)

        try:
            next_obs, reward, done, info = env.step(action)
            
            # --- Collect Data ---
            if "pos" in info and "quat" in info:
                if not done: 
                    debugger.log_step(info["pos"])
                    
                    # Construct state vector
                    drone_state_7d = np.concatenate([info["pos"], info["quat"]])
                    step_reward_1d = np.array([reward])
                    flat_gate_pos = gate_positions_for_save.flatten()

                    r_prog = info.get("rew_progress", 0.0)
                    r_perc = info.get("rew_perception", 0.0)
                    r_pen  = info.get("rew_penalty", 0.0)
                    r_gate = 10.0 if info.get("gate_passed", False) else 0.0
                    
                    reward_components = np.array([r_prog, r_perc, r_pen, r_gate])
                    
                    full_state_row = np.concatenate([
                        drone_state_7d, 
                        step_reward_1d, 
                        flat_gate_pos,
                        reward_components 
                    ])
                    
                    full_state_buffer.append(full_state_row) 

        except RuntimeError as e:
            print(f"[Python] ⚠️ Environment Error: {e}")
            next_obs = env.reset()
            reward = 0.0
            done = True
            
        replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        global_step += 1
        pbar.update(1)

        if global_step >= WARMUP_STEPS and len(replay_buffer) > BATCH_SIZE:
            agent.update(replay_buffer, global_step)

        episode_reward += reward
        episode_step += 1

        if done:
            log_tensorboard_metrics(writer, episode_reward, episode_step, episode_count, global_step)

            # --- Video Save ---
            if episode_count % VIDEO_INTERVAL == 0:
                save_episode_video(current_episode_frames, episode_count, run_dir)
            
            # --- Trajectory Save ---
            if episode_count % LOG_INTERVAL == 0:
                debugger.save_trajectory(episode_count, episode_reward)
                pbar.write(f"📸 Episode {episode_count} | Reward: {episode_reward:.2f}")
            
            if full_state_buffer: 
                save_full_state_trajectory(full_state_buffer, episode_count, npy_dir)
            
            # Reset all buffers
            debugger.reset()
            full_state_buffer = []
            current_episode_frames = []
            
            obs = env.reset()
            episode_count += 1
            episode_reward = 0
            episode_step = 0

        # --- Checkpoint ---
        if global_step > 0 and global_step % SAVE_INTERVAL == 0:
            path = save_checkpoint(agent, global_step, run_dir)
            pbar.write(f"✅ Checkpoint saved: {path}")

    pbar.close()

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Main] Training on device: {device}")
    
    # Setup Env (No more RecordVideo wrapper)
    env = setup_environment(args.seed)
    
    # Run Dir
    run_name = f"flight_racing_{int(time.time())}"
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    run_dir = writer.log_dir
    print(f"[Main] 📂 Run directory: {run_dir}")

    # Directories
    npy_dir, plot_dir = setup_logging_dirs(run_dir)

    # Debugger
    debugger = DebugLogger(log_dir=plot_dir)
    # env is just the wrapper now, so we don't need .unwrapped unless you wrap it again later
    # But to be safe if you add wrappers later, we can check.
    # Since we removed RecordVideo, 'env' IS the FlightmareDrQWrapper.
    debugger.set_gates(env.gate_positions)
    gate_positions_for_saving = env.gate_positions

    # Agent
    agent, replay_buffer = setup_agent_and_buffer(env, device)
    start_step = load_checkpoint_if_available(agent, args.resume)
    
    print("[Main] Connecting to Unity...")
    env.connect_and_warmup()
    
    print(f"[Main] Starting training loop from step {start_step}...")
    try:
        run_training_loop(env, agent, replay_buffer, writer, debugger, start_step, run_dir, npy_dir, gate_positions_for_saving)
    except KeyboardInterrupt:
        print("\n[Main] Training interrupted. Saving emergency checkpoint...")
        save_checkpoint(agent, "interrupted", run_dir)
    finally:
        env.close()
        writer.close()
        print("[Main] Training Finished.")

if __name__ == "__main__":
    main()