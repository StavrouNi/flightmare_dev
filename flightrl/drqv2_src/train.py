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
WARMUP_STEPS = 2000         # Random steps before training starts
TOTAL_STEPS =2_000_000     # Total training duration
SAVE_INTERVAL = 20_000      # Save weights every N steps
LOG_INTERVAL = 200         # Save trajectory plots every N episodes maybe go to 1000
BUFFER_SIZE = 50_000
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPISODE_BEFORE_VIDEO_SAVE = 50 # Save video every 50 episodes

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Train DrQ-v2 on Flightmare')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint file (e.g., agent_step_50000.pt)')
    parser.add_argument('--seed', type=int, default=3, help='Random seed')
    return parser.parse_args()

def setup_environment(seed):
    """Initializes the Flightmare environment and configs."""
    fm_path = os.getenv("FLIGHTMARE_PATH")
    if fm_path is None:
        raise RuntimeError("FLIGHTMARE_PATH is not set. Please export it.")
    
    cfg_path = os.path.join(fm_path, "flightlib/configs/vec_env.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found at {cfg_path}")

    print(f"[Setup] Loading config from {cfg_path}")
    cfg = YAML().load(open(cfg_path, "r"))
    
    # Initialize Wrapper
    env = FlightmareDrQWrapper(cfg, stack_frames=3)
    env.seed(seed)
    
    return env

def setup_agent_and_buffer(env, device):
    """Initializes the Agent and Replay Buffer."""
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
    """Loads weights and parses the starting step number."""
    start_step = 0
    if resume_path and os.path.exists(resume_path):
        print(f"[Train] 🔄 Resuming training from: {resume_path}")
        agent.load(resume_path)
        match = re.search(r"step_(\d+)", resume_path)
        if match:
            start_step = int(match.group(1))
            print(f"[Train] ⏭️  Fast-forwarding global step to {start_step}")
        else:
            print("[Train] ⚠️ Checkpoint loaded, but step count not found. Starting at 0.")
    elif resume_path:
        print(f"[Train] ❌ Checkpoint not found at {resume_path}. Starting fresh.")
    else:
        print("[Train] ✨ Starting fresh training.")
    return start_step

# --- HELPER FUNCTIONS ---

def setup_logging_dirs(run_dir):
    """Creates a single directory for all trajectory data inside the run folder."""
    traj_dir = os.path.join(run_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    return traj_dir

def select_action(agent, env, obs, global_step):
    """Selects an action based on warmup or agent policy."""
    if global_step < WARMUP_STEPS:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return agent.act(obs, global_step, eval_mode=False)

def log_tensorboard_metrics(writer, reward, length, episode, global_step):
    """Logs basic episode metrics to TensorBoard."""
    writer.add_scalar("Train_Steps/Episode_Reward", reward, global_step)
    writer.add_scalar("Train_Steps/Episode_Length", length, global_step)
    writer.add_scalar("Train/Episode_Reward", reward, episode)
    writer.add_scalar("Train/Episode_Length", length, episode)

def save_trajectory_data(trajectory, episode_count, status_tag, gate_idx, save_folder):
    """Saves the full flight trajectory (pos + quat) to a .npy file."""
    if not trajectory:
        return
    filename_base = f"ep_{episode_count:05d}_G{gate_idx}_{status_tag}"
    npy_path = os.path.join(save_folder, filename_base + ".npy")
    np.save(npy_path, np.array(trajectory))

def save_episode_video(frames, episode_count, global_step, writer):
    """Saves local video frames and logs video to TensorBoard."""
    episode_video = np.array(frames)
    if len(episode_video) == 0:
        return

    print(f"[Train] Saving frames for episode {episode_count}...")
    
    # 1. Save Locally (Frames folder inside run_dir/frames/...)
    # Note: writer.log_dir is the base run directory
    ep_dir = os.path.join(writer.log_dir, "frames", f"episode_{episode_count}")
    os.makedirs(ep_dir, exist_ok=True)
    
    for i, frame in enumerate(episode_video):
        # Frame is (3, H, W) RGB -> Need (H, W, 3) BGR for OpenCV
        frame_hwc = np.transpose(frame, (1, 2, 0))
        frame_bgr = cv2.cvtColor(frame_hwc, cv2.COLOR_RGB2BGR)
        filename = os.path.join(ep_dir, f"{i:03d}.png")
        cv2.imwrite(filename, frame_bgr)
        
    print(f"[Train] Saved {len(episode_video)} frames to: {ep_dir}")

    # 2. Log to TensorBoard
    writer.add_video(
        "Train/Episode_Video", 
        episode_video[None, ...], # Add batch dim
        global_step, 
        fps=20
    )

def save_checkpoint(agent, global_step, run_dir):
    """Saves the agent weights."""
    filename = f"agent_step_{global_step}.pt"
    save_path = os.path.join(run_dir, "checkpoints", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True) 
    agent.save(save_path)
    return save_path

# --- MAIN LOOP ---

def run_training_loop(env, agent, replay_buffer, writer, debugger, start_step, run_dir):
    """The main training execution loop."""
    
    # 1. Setup: Create the single 'trajectories' folder inside the unique run dir
    traj_dir = setup_logging_dirs(run_dir)
    
    # Trackers
    global_step = start_step
    episode_reward = 0
    episode_step = 0
    episode_count = 0

    # Data Buffers
    current_episode_frames = []
    
    obs = env.reset()

    # Progress bar
    pbar = tqdm(total=TOTAL_STEPS, initial=start_step, desc="Training", unit="step")

    while global_step < TOTAL_STEPS:
        
        # --- 1. Capture Data ---
        frame = obs[:3, :, :].copy() 
        current_episode_frames.append(frame)

        # --- 2. Select Action ---
        action = select_action(agent, env, obs, global_step)

        # --- 3. Step Environment ---
        try:
            next_obs, reward, done, info = env.step(action)
            
            # Log Position & Trajectory
            if "pos" in info:
                debugger.log_step(info["pos"])
                
        except RuntimeError as e:
            print(f"[Python] ⚠️ Environment Error at Step {global_step}: {e}")
            next_obs = env.reset()
            reward = 0.0
            done = True
            
        # --- 4. Store & Update ---
        replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        global_step += 1
        pbar.update(1)

        if global_step >= WARMUP_STEPS and len(replay_buffer) > BATCH_SIZE:
            agent.update(replay_buffer, global_step)

        episode_reward += reward
        episode_step += 1

        # --- 5. Episode Complete Logic ---
        if done:

            #  Log Metrics
            log_tensorboard_metrics(writer, episode_reward, episode_step, episode_count, global_step)

            # Save Video (Periodic)
            if episode_count % NUM_EPISODE_BEFORE_VIDEO_SAVE == 0:
                save_episode_video(current_episode_frames, episode_count, global_step, writer)
            
            # Debug Plotting (Periodic)
            if episode_count % LOG_INTERVAL == 0:
                debugger.save_trajectory(episode_count, episode_reward)
                pbar.write(f"📸 Episode {episode_count} | Reward: {episode_reward:.2f}")

            # Reset
            debugger.reset()
            obs = env.reset()
            current_episode_frames = []
            
            episode_count += 1
            episode_reward = 0
            episode_step = 0

        # --- Save Checkpoint ---
        if global_step > 0 and global_step % SAVE_INTERVAL == 0:
            path = save_checkpoint(agent, global_step, run_dir)
            pbar.write(f"✅ Checkpoint saved: {path}")

    pbar.close()

def main():
    args = parse_args()
    
    # 1. Setup Device & Env
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Main] Training on device: {device}")
    env = setup_environment(args.seed)
    
    # 2. Create Unique Run Directory (based on timestamp)
    run_name = f"flight_racing_{int(time.time())}"
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    run_dir = writer.log_dir
    print(f"[Main] 📂 Run directory: {run_dir}")

    # 3. Setup Debugger (pointing to 'trajectories' inside run_dir)
    debugger = DebugLogger(log_dir=os.path.join(run_dir, "trajectories"))
    debugger.set_gates(env.gate_positions)
    print(f"[Main] Debugger loaded {len(env.gate_positions)} gates.")

    # 4. Agent & Buffer
    agent, replay_buffer = setup_agent_and_buffer(env, device)
    start_step = load_checkpoint_if_available(agent, args.resume)
    
    # 5. Start
    print("[Main] Connecting to Unity...")
    env.connect_and_warmup()
    
    print(f"[Main] Starting training loop from step {start_step}...")
    try:
        run_training_loop(env, agent, replay_buffer, writer, debugger, start_step, run_dir)
    except KeyboardInterrupt:
        print("\n[Main] Training interrupted. Saving emergency checkpoint...")
        save_checkpoint(agent, "interrupted", run_dir)
    finally:
        env.close()
        writer.close()
        print("[Main] Training Finished.")

if __name__ == "__main__":
    main()