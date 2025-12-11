import matplotlib
matplotlib.use('Agg') # Headless mode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import glob
import os
import re
import csv
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
CRASH_FOLDER = "debug_plots/crashes"
OUTPUT_PLOT = "crash_orientation_map.png"
OUTPUT_CSV = "crash_report.csv"

GATE_POSITIONS = [
    [-10.0, 3.0, 2.5],  # G1
    [-5.0, 8.0, 2.5],   # G2
    [0.0, 3.0, 2.5],    # G3
    [-5.0, -2.0, 2.5]   # G4
]

# Colors for gates 0-3
GATE_COLORS = ['red', 'orange', 'gold', 'green']
# ---------------------

def get_gate_idx(filename):
    match = re.search(r"_G(\d+)_", filename)
    return int(match.group(1)) if match else 0

def get_episode_idx(filename):
    match = re.search(r"ep_(\d+)_", filename)
    return int(match.group(1)) if match else 0

def analyze_terminal_states():
    files = glob.glob(os.path.join(CRASH_FOLDER, "*.npy"))
    if not files:
        print("❌ No crash files found.")
        return

    print(f"📂 Processing {len(files)} crash files...")

    # Data Containers
    terminal_pos = []
    terminal_vec = [] # Direction drone was facing
    colors = []
    
    # Open CSV for writing
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Episode", "Target_Gate", "X", "Y", "Z", "Pitch_Deg", "Roll_Deg"])

        for f in files:
            try:
                # 1. Load Data
                data = np.load(f) # [Steps, 7]
                last_state = data[-1] # Only take the LAST row
                
                # Unpack
                pos = last_state[0:3]
                quat = last_state[3:7] # [qw, qx, qy, qz] (Check your C++ order!)
                
                # 2. Calculate Heading Vector (Where was I looking?)
                # Convert Quat to Rotation Matrix
                # Scipy expects [x, y, z, w], Flightmare usually gives [w, x, y, z]
                quat_scipy = np.roll(quat, -1) 
                r = R.from_quat(quat_scipy)
                
                # Get Body X-axis (Forward vector) in World Frame
                heading = r.apply([1, 0, 0]) 
                
                # Get Euler Angles for CSV
                euler = r.as_euler('xyz', degrees=True) # [Roll, Pitch, Yaw]
                
                # 3. Store for Plotting
                gate_idx = get_gate_idx(os.path.basename(f))
                c_idx = min(gate_idx, 3)
                
                terminal_pos.append(pos)
                terminal_vec.append(heading)
                colors.append(GATE_COLORS[c_idx])

                # 4. Write to CSV
                ep_idx = get_episode_idx(os.path.basename(f))
                writer.writerow([ep_idx, gate_idx, 
                                 f"{pos[0]:.2f}", f"{pos[1]:.2f}", f"{pos[2]:.2f}", 
                                 f"{euler[1]:.2f}", f"{euler[0]:.2f}"])

            except Exception as e:
                pass

    # --- PLOTTING ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    t_pos = np.array(terminal_pos)
    t_vec = np.array(terminal_vec)
    
    if len(t_pos) > 0:
        # 1. Draw Arrows (Quiver)
        # origin_x, origin_y, origin_z, dir_x, dir_y, dir_z
        # Length=1.0 makes arrows visible
        ax.quiver(t_pos[:,0], t_pos[:,1], t_pos[:,2], 
                  t_vec[:,0], t_vec[:,1], t_vec[:,2], 
                  color=colors, length=0.8, normalize=True, arrow_length_ratio=0.3)
        
        # 2. Draw Dots at base
        ax.scatter(t_pos[:,0], t_pos[:,1], t_pos[:,2], c=colors, s=20, alpha=0.6)

    # 3. Draw Gates
    gates = np.array(GATE_POSITIONS)
    ax.scatter(gates[:, 0], gates[:, 1], gates[:, 2], c='blue', marker='s', s=100, label='Gate')
    for i, g in enumerate(gates):
        ax.text(g[0], g[1], g[2]+1.5, f"G{i+1}", color='black', ha='center', weight='bold')

    ax.set_title("Crash Orientations (Arrows show facing direction)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlim(0, 8)
    ax.set_xlim(-15, 5)
    ax.set_ylim(-5, 10)

    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"✅ Orientation plot saved to: {OUTPUT_PLOT}")
    print(f"✅ Data report saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    analyze_terminal_states()