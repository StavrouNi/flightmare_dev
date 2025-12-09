import os
import argparse
import re
import torch
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
TOTAL_STEPS = 1_000_000     # Total training duration
SAVE_INTERVAL = 10_000      # Save weights every N steps
LOG_INTERVAL = 1000          # Save trajectory plots every N episodes
BUFFER_SIZE = 100_000
BATCH_SIZE = 256
LEARNING_RATE = 1e-4

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
        num_expl_steps=2000,  # Pure random steps at start
        update_every_steps=2, # Update network every 2 env steps
        stddev_schedule='linear(1.0,0.1,100000)', # Decay noise over 100k steps
        stddev_clip=0.3       # Clip target noise for stability
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
        
        # Regex to find 'step_12345'
        match = re.search(r"step_(\d+)", resume_path)
        if match:
            start_step = int(match.group(1))
            print(f"[Train] ⏭️  Fast-forwarding global step to {start_step}")
        else:
            print("[Train] ⚠️ Checkpoint loaded, but step count not found in filename. Starting at 0.")
            
    elif resume_path:
        print(f"[Train] ❌ Checkpoint not found at {resume_path}. Starting fresh.")
    else:
        print("[Train] ✨ Starting fresh training.")

    return start_step

def run_training_loop(env, agent, replay_buffer, writer, debugger, start_step):
    """The main training execution loop."""
    
    # Initialize trackers
    global_step = start_step
    episode_reward = 0
    episode_step = 0
    episode_count = 0
    
    obs = env.reset()

    # Progress bar starting from where we left off
    pbar = tqdm(total=TOTAL_STEPS, initial=start_step, desc="Training", unit="step")

    while global_step < TOTAL_STEPS:
        
        # --- 1. Select Action ---
        # If we just resumed, we might want to warm up the buffer briefly even if step > 2000
        is_warmup = global_step < WARMUP_STEPS
        
        if is_warmup:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = agent.act(obs, global_step, eval_mode=False)

        # --- 2. Step Environment ---
        try:
            next_obs, reward, done, info = env.step(action)
            
            # Log real-world position for plotting
            if "pos" in info and not done:
                debugger.log_step(info["pos"])
                
        except RuntimeError as e:
            print(f"[Python] ⚠️ Environment Error at Step {global_step}: {e}")
            next_obs = env.reset()
            reward = 0.0
            done = True
            
        # --- 3. Store Data ---
        replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        global_step += 1
        pbar.update(1)

        # --- 4. Update Agent ---
        # Only update if we are past warmup AND have enough data in buffer
        if global_step >= WARMUP_STEPS and len(replay_buffer) > BATCH_SIZE:
            agent.update(replay_buffer, global_step)

        # --- 5. Episode Logging ---
        episode_reward += reward
        episode_step += 1

        if done:
            # TensorBoard
            writer.add_scalar("Train/Episode_Reward", episode_reward, episode_count)
            writer.add_scalar("Train/Episode_Length", episode_step, episode_count)

            # Debug Plotting
            if episode_count % LOG_INTERVAL == 0:
                debugger.save_trajectory(episode_count, episode_reward)
                pbar.write(f"📸 Episode {episode_count} | Reward: {episode_reward:.2f} | Steps: {episode_step}")

            # Reset Episode Trackers
            debugger.reset()
            obs = env.reset()
            episode_count += 1
            episode_reward = 0
            episode_step = 0

        # --- 6. Save Checkpoint ---
        if global_step > 0 and global_step % SAVE_INTERVAL == 0:
            save_path = f"agent_step_{global_step}.pt"
            agent.save(save_path)
            pbar.write(f"✅ Checkpoint saved: {save_path}")

    pbar.close()

def main():
    args = parse_args()
    
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Main] Training on device: {device}")
    
    env = setup_environment(args.seed)
    
    # 2. Tools
    writer = SummaryWriter(log_dir="runs/flight_racing_experiment")
    debugger = DebugLogger(log_dir="debug_plots")
    debugger.set_gates(env.gate_positions)
    print(f"[Main] Debugger loaded {len(env.gate_positions)} gates.")

    # 3. Agent & Buffer
    agent, replay_buffer = setup_agent_and_buffer(env, device)
    
    # 4. Resume Logic
    start_step = load_checkpoint_if_available(agent, args.resume)
    
    # 5. Start Simulation
    print("[Main] Connecting to Unity...")
    env.connect_and_warmup()
    
    # 6. Run Loop
    print(f"[Main] Starting training loop from step {start_step}...")
    try:
        run_training_loop(env, agent, replay_buffer, writer, debugger, start_step)
    except KeyboardInterrupt:
        print("\n[Main] Training interrupted by user. Saving emergency checkpoint...")
        agent.save("agent_interrupted.pt")
    finally:
        env.close()
        writer.close()
        print("[Main] Training Finished.")

if __name__ == "__main__":
    main()