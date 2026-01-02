import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import sys
import os

# ==============================================================================
# 1. HARDCODED CONFIGURATION (The "Truth") FOR FIG 8 PLOTTING
# ==============================================================================
# [X, Y, Z, Yaw]
# GATE_POSES_REF = np.array([
#     [6.0, -3.0, 2.5, 1.57],      # Gate 0 (Target)
#     [8.0,  0.0, 2.5, 3.14159],   # Gate 1
#     [6.0,  3.0, 2.5, -1.57],     # Gate 2
#     [-2.0, -3.0, 2.5, -1.57],     # Gate 3
#     [-4.0, 0.0, 2.5, 3.14159],   # Gate 4
#     [-2.0, 3.0, 2.5, 1.57],      # Gate 5
# ])
# ==============================================================================
# 1. HARDCODED CONFIGURATION (The "Truth") FOR SPLINE PLOTTING
# ==============================================================================
GATE_POSES_REF = np.array([
    [ 0.0, -3.0, 2.5, 1.57],      # Gate 1
    [-3.0,  0.0, 2.5, 3.14159],   # Gate 2
    [ 0.0,  3.0, 2.5, 3.14159],   # Gate 3
    [-3.0,  6.0, 2.5, 3.14159],   # Gate 4
    [ 0.0,  9.0, 2.5, 1.57],      # Gate 5
    [ 4.0,  3.0, 2.5, 3.14159],   # Gate 6
])

# ==============================================================================
# 1. HARDCODED CONFIGURATION (The "Truth") FOR CIRCLE PLOTTING
# ==============================================================================
GATE_POSES_REF = np.array([
    [ -10.0,  3.0, 2.5, 0.0],      # Gate 1
    [ -5.0,  8.0, 2.5, 1.57],   # Gate 2
    [0.0,  3.0, 2.5, -0.0],     # Gate 3
    [ -5.0, -2.0, 2.5, 1.57],      # Gate 4
])

NUM_GATES = GATE_POSES_REF.shape[0]

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def rotate_and_draw_gate(ax, center, yaw, half_size, color, label_idx):
    """Draws the wireframe square of the gate rotated by Yaw."""
    x, y, z = center
    h = half_size
    
    # Visual Frame (Rotated 90 deg to match track)
    corners_local = np.array([
        [-h, 0, -h], [ h, 0, -h], [ h, 0,  h], [-h, 0,  h], [-h, 0, -h]
    ])
    
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s,  c, 0], [0,  0, 1]])
    corners_world = (R @ corners_local.T).T + np.array([x, y, z])
    
    ax.plot(corners_world[:, 0], corners_world[:, 1], corners_world[:, 2], 
            color=color, linewidth=2, linestyle='--')
    ax.text(x, y, z + h + 0.2, f'G{label_idx}', color=color, fontsize=10, ha='center', fontweight='bold')

def load_robust_trajectory(npy_path):
    """
    Loads data based on the specific structure in train.py:
    [State(7) | Reward(1) | Gates(...) | Prog(1) | Perc(1) | Pen(1) | GatePass(1)]
    """
    try:
        data = np.load(npy_path)
        
        # 1. Drone State (Cols 0-6)
        drone_states = data[:, 0:7]
        
        # 2. Total Reward (Col 7)
        total_rewards = data[:, 7]
        
        # 3. Components (Last 4 Cols)
        # saved as: [r_prog, r_perc, r_pen, r_gate]
        progress_rews = data[:, -4]
        perc_rewards  = data[:, -3]
        penalty_rews  = data[:, -2]
        gate_pass_flags = data[:, -1]

        return drone_states, total_rewards, progress_rews, perc_rewards, penalty_rews, GATE_POSES_REF

    except Exception as e:
        print(f"Error loading {npy_path}: {e}")
        return None, None, None, None, None, None

def plot_dashboard(drone_states, total_rewards, progress_rews, perc_rewards, gates_ref, episode_idx, run_dir):
    fig = plt.figure(figsize=(16, 12))
    
    # --- TOP PLOT: 3D Trajectory ---
    ax3d = fig.add_subplot(2, 1, 1, projection='3d')
    drone_pos = drone_states[:, 0:3]
    
    # Color path by Total Reward
    norm = Normalize(vmin=-5.0, vmax=10.0)
    colors = cm.RdYlGn(norm(total_rewards))
    
    for i in range(len(drone_pos) - 1):
        ax3d.plot(drone_pos[i:i+2, 0], drone_pos[i:i+2, 1], drone_pos[i:i+2, 2],
                color=colors[i], linewidth=2.0)
        
        # Draw Star if Gate Passed (Total reward spike > 8.0)
        if total_rewards[i] > 8.0:
             ax3d.scatter(drone_pos[i,0], drone_pos[i,1], drone_pos[i,2], c='magenta', s=200, marker='*', zorder=10)

    # Draw Gates
    for i in range(len(gates_ref)):
        rotate_and_draw_gate(ax3d, gates_ref[i, :3], gates_ref[i, 3], 1.0, 'red', i)
    
    # Start/End
    ax3d.scatter(drone_pos[0,0], drone_pos[0,1], drone_pos[0,2], c='green', s=100, label='Start')
    ax3d.scatter(drone_pos[-1,0], drone_pos[-1,1], drone_pos[-1,2], c='black', marker='s', s=100, label='End')

    # Formatting 3D
    ax3d.set_title(f"Episode {episode_idx} | Total Reward: {np.sum(total_rewards):.2f}")
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    
    # Limits
    all_x = np.concatenate((drone_pos[:,0], gates_ref[:,0]))
    all_y = np.concatenate((drone_pos[:,1], gates_ref[:,1]))
    all_z = np.concatenate((drone_pos[:,2], gates_ref[:,2]))
    max_range = np.array([all_x.ptp(), all_y.ptp(), all_z.ptp()]).max() / 2.0
    mid_x, mid_y, mid_z = all_x.mean(), all_y.mean(), all_z.mean()
    ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3d.set_zlim(mid_z - max_range, mid_z + max_range)

    # --- BOTTOM PLOT: Reward Components ---
    ax2 = fig.add_subplot(2, 1, 2)
    steps = np.arange(len(total_rewards))
    
    # 1. Total Reward (Left Axis - Black)
    ax2.plot(steps, total_rewards, color='black', alpha=0.2, label='Total Reward')
    ax2.set_ylabel('Total Reward', color='black', fontweight='bold')
    ax2.set_ylim(-5, 12) # Fixed scale to see spikes
    ax2.grid(True, alpha=0.3)

    # 2. Components (Right Axis - Colored)
    ax_right = ax2.twinx()
    
    # Progress (Green)
    ax_right.plot(steps, progress_rews, color='green', alpha=0.8, linewidth=1.5, label='Progress')
    
    # Perception (Blue)
    ax_right.plot(steps, perc_rewards, color='blue', linewidth=2.0, label='Perception')
    
    ax_right.set_ylabel('Shaping Rewards (Small Scale)', color='blue', fontweight='bold')
    ax_right.set_ylim(-0.1, 0.2) # Zoom in to see the 0.07 scale
    ax_right.tick_params(axis='y', labelcolor='blue')
    
    # Combined Legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax2.set_xlabel('Simulation Step')
    ax2.set_title("Reward Component Analysis (Stored Data)")
    
    # Save
    out_dir = os.path.join(run_dir, "3D_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"dashboard_ep_{episode_idx:05d}.png")
    plt.savefig(out_path)
    plt.close(fig)
    return out_path

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def run_analysis(log_dir, target_episode=None):
    traj_dir = os.path.join(log_dir, "npy_trajectories")
    
    if not os.path.exists(traj_dir):
        nested = os.path.join(log_dir, "npy_trajectories", "npy_trajectories")
        if os.path.exists(nested):
            traj_dir = nested
        else:
            print(f"❌ No trajectory data found in: {traj_dir}")
            return

    files = sorted([f for f in os.listdir(traj_dir) if f.endswith(".npy")])
    if target_episode:
        files = [f for f in files if f"ep_{int(target_episode):05d}" in f]

    print(f"🔍 Found {len(files)} files.")

    for f in files:
        try:
            ep_idx = int(f.split('_')[-1].split('.')[0])
        except:
            continue
            
        path = os.path.join(traj_dir, f)
        
        # Load data
        drone_s, total_r, prog_r, perc_r, pen_r, gates_ref = load_robust_trajectory(path)
        
        if drone_s is not None:
            save_p = plot_dashboard(drone_s, total_r, prog_r, perc_r, gates_ref, ep_idx, log_dir)
            print(f"   -> Dashboard saved: {save_p}")
            if total_r is not None and len(total_r) > 0:
                 print(f"      Reward Sum: {np.sum(total_r):.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_trajectories.py <RUN_DIR> [EPISODE]")
    else:
        run_analysis(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)