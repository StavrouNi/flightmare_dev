import os
from torch.utils.tensorboard import SummaryWriter
from logger_utils import DebugLogger
import torch
from ruamel.yaml import YAML
from tqdm import tqdm
from wrapper import FlightmareDrQWrapper
from drqv2 import DrQV2Agent
from replay_buffer import ReplayBuffer

def main():

    # Setup Paths and configs
    fm_path = os.getenv("FLIGHTMARE_PATH")
    if fm_path is None:
        raise RuntimeError("FLIGHTMARE_PATH is not set")
    
    cfg_path = os.path.join(fm_path, "flightlib/configs/vec_env.yaml")
    cfg = YAML().load(open(cfg_path, "r"))

    # Initialize Environment
    # This loads Unity, Physics, and the Gate Track logic
    env = FlightmareDrQWrapper(cfg, stack_frames=3)

    # Observability loggers
    writer = SummaryWriter(log_dir="runs/flight_racing_experiment_1")
    debugger = DebugLogger(log_dir="debug_plots")
    debugger.set_gates(env.gate_positions) 
    print(f"[Main] Debugger loaded {len(env.gate_positions)} gates.")

    # Setup Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    action_shape = env.action_space.shape
    obs_shape = env.observation_space.shape # (9, 84, 84)
    
    # Initialize Agent (DrQ-v2)
    agent = DrQV2Agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        lr=1e-4,
        feature_dim=50,
        hidden_dim=1024,
        batch_size=256
    )

    # Initialize Replay Buffer (Stores 100k frames)
    # We store frames in RAM. 100k * 64*64*3 bytes is ~1.2GB. Safe.
    replay_buffer = ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=action_shape,
        capacity=100000,
        batch_size=256,
        device=device
    )

    print("4. Connecting to Unity...")
    env.connect_and_warmup()

    # 7. Training Loop
    print("5. Starting Training Loop!")

    max_steps = 1_000_000
    save_interval = 50_000 # 50.000 is good?
    episode_reward = 0
    episode_step = 0
    episode = 0
    
    pbar = tqdm(total=max_steps, desc="Training", unit="step")
    
    obs = env.reset()
    
    for step in range(max_steps):
        
        # Random exploration for the first 2000 steps to fill buffer
        if step < 2000:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = agent.act(obs, step, eval_mode=False)

        try:
            next_obs, reward, done, info = env.step(action)
            if "pos" in info:
                debugger.log_step(info["pos"])
        except RuntimeError as e:
            #### TODO We should not go inside here. This is the last resort. 
            # No image and other exceptions should be handled inside the wrapper.
            print(f"[Python] ⚠️ CAUGHT EXCEPTION at Step {step}: {e}")
            # Just try to reset once.
            next_obs = env.reset()
            # Keep the variables valid so the loop doesn't break # TODO What should we give here?
            reward = 0.0
            done = True
            info = {"timeout": True}


        # Add to Buffer
        # Note: We don't verify 'done' mask here because racing is continuous until crash 
        replay_buffer.add(obs, action, reward, next_obs, done)
        
        obs = next_obs
        step += 1

        # Update Agent
        if step >= 200:
            agent.update(replay_buffer, step)

        episode_reward += reward
        episode_step += 1
        pbar.update(1) 

        # Logging / Reset
        if done:
            # TENSORBOARD LOGGING
            writer.add_scalar("Train/Episode_Reward", episode_reward, episode)
            writer.add_scalar("Train/Episode_Length", episode_step, episode)
            
            # TRAJECTORY MAPPING (Save image every 20 episodes)
            if episode % 20 == 0:
                debugger.save_trajectory(episode, episode_reward)
                pbar.write(f"📸 Saved trajectory plot for Episode {episode}")

            # Reset
            debugger.reset()
            obs = env.reset()
            episode += 1
            episode_reward = 0
            episode_step = 0
            
        # Save checkpoint
        if step > 0 and step % save_interval == 0:
            save_path = f"agent_step_{step}.pt"
            agent.save(save_path) 
            pbar.write(f"✅ Saved weights to {save_path}")

    # Done
    env.close()
    pbar.close()
    print("Training Finished!")

if __name__ == "__main__":
    main()