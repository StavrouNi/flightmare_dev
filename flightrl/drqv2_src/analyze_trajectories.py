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
# GATE_POSES_REF = np.array([
#     [ 0.0, -3.0, 2.5, 1.57],      # Gate 1
#     [-3.0,  0.0, 2.5, 3.14159],   # Gate 2
#     [ 0.0,  3.0, 2.5, 3.14159],   # Gate 3
#     [-3.0,  6.0, 2.5, 3.14159],   # Gate 4
#     [ 0.0,  9.0, 2.5, 1.57],      # Gate 5
#     [ 4.0,  3.0, 2.5, 3.14159],   # Gate 6
# ])

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

# ... (Imports and Config remain the same) ...

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
    Loads data matching the specific structure in train.py:
    [State(7) | Vel(3) | Act(4) | Reward(1) | Gates(...) | Prog(1) | Perc(1) | Pen(1) | GatePass(1)]
    """
    try:
        data = np.load(npy_path)
        
        # 1. Drone State (Cols 0-6)
        drone_states = data[:, 0:7]
        
        # 2. Linear Velocities (Cols 7-9) -> CORRECTED
        velocities = data[:, 7:10]

        # 3. Actions (Cols 10-13) -> CORRECTED
        actions = data[:, 10:14]
        
        # 4. Total Reward (Col 14) -> CORRECTED
        total_rewards = data[:, 14]
        
        # 5. Components (Last 4 Cols)
        # These are safe to access via negative indexing
        progress_rews = data[:, -4]
        perc_rewards  = data[:, -3]
        penalty_rews  = data[:, -2]
        gate_pass_flags = data[:, -1]

        return drone_states, total_rewards, actions, velocities, progress_rews, perc_rewards, penalty_rews, GATE_POSES_REF

    except Exception as e:
        print(f"Error loading {npy_path}: {e}")
        return None, None, None, None, None, None, None, None

def plot_dashboard(drone_states, total_rewards, actions, velocities, progress_rews, perc_rewards, gates_ref, episode_idx, run_dir):
    # Calculate derived speed (magnitude of position change)
    drone_pos = drone_states[:, 0:3]
    speeds = calculate_speed_proxy(drone_pos)

    # INCREASED FIG HEIGHT: 5 Rows now
    fig = plt.figure(figsize=(16, 20))
    
    # --- PLOT 1: 3D Trajectory (Row 1) ---
    ax3d = fig.add_subplot(5, 1, 1, projection='3d')
    norm = Normalize(vmin=-5.0, vmax=10.0)
    colors = cm.RdYlGn(norm(total_rewards))
    for i in range(len(drone_pos) - 1):
        ax3d.plot(drone_pos[i:i+2, 0], drone_pos[i:i+2, 1], drone_pos[i:i+2, 2],
                color=colors[i], linewidth=2.0)
        # Mark gate passes (reward spikes)
        if total_rewards[i] > 8.0:
             ax3d.scatter(drone_pos[i,0], drone_pos[i,1], drone_pos[i,2], c='magenta', s=200, marker='*', zorder=10)
    for i in range(len(gates_ref)):
        rotate_and_draw_gate(ax3d, gates_ref[i, :3], gates_ref[i, 3], 1.0, 'red', i)
    ax3d.set_title(f"Episode {episode_idx} | Total Reward: {np.sum(total_rewards):.2f}")


    # --- PLOT 2: Reward Components (Row 2) ---
    ax2 = fig.add_subplot(5, 1, 2)
    steps = np.arange(len(total_rewards))
    ax2.plot(steps, total_rewards, color='black', alpha=0.2, label='Total Reward')
    ax2.set_ylabel('Total Reward')
    ax2.set_ylim(-5, 12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    ax_right = ax2.twinx()
    ax_right.plot(steps, progress_rews, color='green', alpha=0.8, linewidth=1.5, label='Progress')
    ax_right.plot(steps, perc_rewards, color='blue', linewidth=2.0, label='Perception')
    ax_right.set_ylabel('Shaping Rewards', color='blue')
    ax_right.legend(loc='upper right')
    ax2.set_title("Reward Analysis")


    # # --- PLOT 3: SPEED ANALYSIS (Row 3) ---
    # ax3 = fig.add_subplot(5, 1, 3)
    # ax3.plot(steps, speeds, color='red', linewidth=2.0, label='Speed (Displacement/Step)')
    # gate_indices = np.where(total_rewards > 8.0)[0]
    # if len(gate_indices) > 0:
    #     ax3.scatter(gate_indices, speeds[gate_indices], color='magenta', marker='*', s=100, zorder=5, label='Gate Pass')
    # ax3.set_ylabel('Speed Proxy', color='red', fontweight='bold')
    # ax3.grid(True, alpha=0.3)
    # ax3.legend()


    # --- PLOT 4: ACTIONS (Row 4) ---
    # Actions: [collective_thrust, roll_rate, pitch_rate, yaw_rate]
    ax4 = fig.add_subplot(5, 1, 4)
    if actions is not None and actions.shape[1] >= 4:
        labels = ['Collective Thrust', 'Roll Rate', 'Pitch Rate', 'Yaw Rate']
        colors_act = ['purple', 'red', 'green', 'blue']
        for i in range(4):
            ax4.plot(steps, actions[:, i], label=labels[i], color=colors_act[i], alpha=0.8)
        ax4.set_title("Action Inputs (Thrust + Body Rates)")
        ax4.set_ylabel("Action Value")
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper right', ncol=4)
    else:
        ax4.text(0.5, 0.5, "Actions data shape incorrect or missing", ha='center')


    # --- PLOT 5: VELOCITIES (Row 5) ---
    # Assuming 3 velocities (Vx, Vy, Vz)
    ax5 = fig.add_subplot(5, 1, 5)
    if velocities is not None and velocities.shape[1] >= 3:
        v_labels = ['Vx', 'Vy', 'Vz']
        v_colors = ['r', 'g', 'b']
        for i in range(3):
            ax5.plot(steps, velocities[:, i], label=v_labels[i], color=v_colors[i])
        ax5.set_title("Drone Linear Velocities")
        ax5.set_ylabel("m/s")
        ax5.set_xlabel("Simulation Step")
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='upper right', ncol=3)
    else:
        ax5.text(0.5, 0.5, "Velocity data shape incorrect or missing", ha='center')


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
        
        # Load data (Updated unpack signature)
        drone_s, total_r, acts, vels, prog_r, perc_r, pen_r, gates_ref = load_robust_trajectory(path)
        
        if drone_s is not None:
            # Print shape for debugging
            if acts is not None:
                print(f"   [Debug] Loaded Actions shape: {acts.shape}")
            
            save_p = plot_dashboard(drone_s, total_r, acts, vels, prog_r, perc_r, gates_ref, ep_idx, log_dir)
            print(f"   -> Dashboard saved: {save_p}")

def calculate_speed_proxy(drone_positions):
    diffs = np.diff(drone_positions, axis=0)
    speeds = np.linalg.norm(diffs, axis=1)
    speeds = np.insert(speeds, 0, 0.0)
    return speeds

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_trajectories.py <RUN_DIR> [EPISODE]")
    else:
        run_analysis(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)