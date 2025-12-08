import matplotlib.pyplot as plt
import os
import numpy as np

class DebugLogger:
    def __init__(self, log_dir="debug_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.positions = []
        self.gate_positions = None

    def set_gates(self, gates):
        self.gate_positions = gates

    def log_step(self, pos):
        """Record position [x, y, z]"""
        self.positions.append(pos)

    def save_trajectory(self, episode_idx, reward):
        """Draws the Top-Down view of the flight"""
        if not self.positions:
            return

        plt.figure(figsize=(10, 10))
        
        # 1. Plot Gates (Red Circles)
        if self.gate_positions is not None:
            gx = self.gate_positions[:, 0]
            gy = self.gate_positions[:, 1]
            plt.scatter(gx, gy, c='red', s=200, marker='X', label='Gates')
            # Number the gates
            for i, (x, y) in enumerate(zip(gx, gy)):
                plt.text(x+0.5, y+0.5, str(i+1), fontsize=12, color='red')

        # 2. Plot Drone Path (Blue Line)
        path = np.array(self.positions)
        plt.plot(path[:, 0], path[:, 1], c='blue', alpha=0.6, linewidth=2, label='Drone Path')
        
        # 3. Plot Start/End
        plt.scatter(path[0, 0], path[0, 1], c='green', s=100, label='Start')
        plt.scatter(path[-1, 0], path[-1, 1], c='black', s=100, label='Crash/End')

        plt.title(f"Episode {episode_idx} | Reward: {reward:.2f}")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal') # Keep aspect ratio square so circles look like circles

        # Save
        save_path = os.path.join(self.log_dir, f"traj_ep_{episode_idx}.png")
        plt.savefig(save_path)
        plt.close()
        
        # Reset for next episode
        self.positions = []
        
    def reset(self):
        """Call this every time the environment resets!"""
        self.positions = []