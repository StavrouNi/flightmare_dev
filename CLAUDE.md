# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Flightmare** is a flexible modular quadrotor simulator consisting of:
- **flightlib**: Core C++ library with physics simulation and Python bindings
- **flightrender**: Unity-based rendering engine (decoupled from physics)
- **flightrl**: Reinforcement learning environments and training scripts
- **flightros**: ROS integration (currently disabled via CATKIN_IGNORE)

The simulator uses a decoupled architecture where physics (flightlib) and rendering (Unity) communicate via ZMQ bridge and can run independently.

## Environment Setup

This repository requires the `FLIGHTMARE_PATH` environment variable:
```bash
export FLIGHTMARE_PATH=/home/ubuntu/Stavrou_PhD/flightmare_dev
```

**Required dependencies:**
- Python 3.7 (exact version required for pybind11 compatibility)
- CMake >= 3.0
- Eigen3 >= 3.3.4
- OpenCV
- ZMQ & ZMQPP
- yaml-cpp
- TensorFlow (for RL training)
- C++17 compiler

## Build Commands

### Building flightlib (Core C++ Library + Python Bindings)

```bash
# Standard build
cd $FLIGHTMARE_PATH/flightlib/build
cmake ..
make -j4

# Install Python bindings (flightgym)
cd $FLIGHTMARE_PATH/flightlib
pip install -e .
```

**Build options:**
- `-DBUILD_TESTS=ON` - Build C++ tests (default ON)
- `-DBUILD_UNITY_BRIDGE_TESTS=ON` - Build Unity bridge tests (default ON)
- `-DBUILD_DEMOS=ON` - Build demo executables (default OFF)
- `-DENABLE_FAST=ON` - Optimize for speed with -Ofast (default ON)
- `-DENABLE_PARALLEL=ON` - Enable OpenMP parallelization (default ON)

### Building flightrl (RL Training)

```bash
cd $FLIGHTMARE_PATH/flightrl
pip install -e .
```

### Building flightros (ROS - Currently Disabled)

ROS integration is currently disabled via `CATKIN_IGNORE` file. To enable:
```bash
rm $FLIGHTMARE_PATH/flightros/CATKIN_IGNORE
catkin build flightros
```

## Running Tests

```bash
# Run C++ tests
cd $FLIGHTMARE_PATH/flightlib/build
./test_lib           # Core library tests
./test_gym           # Python wrapper tests
./test_unity_bridge  # Unity bridge tests
```

## Running Examples

### C++ Demos (require BUILD_DEMOS=ON)

```bash
cd $FLIGHTMARE_PATH/flightlib/build
./spawn_gates_and_camera  # Spawn quadrotor with camera and gates
./demo_quad_env           # QuadrotorEnv demo
```

**Note:** Unity must be running with Flightmare environment loaded for these demos.

### Python RL Training

```bash
cd $FLIGHTMARE_PATH/flightrl/examples

# Train a PPO agent
python run_drone_control.py --train 1 --render 1

# Test trained model
python run_drone_control.py --train 0 --render 1 --weight ./saved/quadrotor_env.zip

# DrQv2 training (newer implementation)
cd $FLIGHTMARE_PATH/flightrl/drqv2_src
python train.py
```

## Code Architecture

### Core Components

**flightlib Structure:**
- `src/dynamics/` - Quadrotor dynamics models (modular system)
- `src/objects/` - Quadrotor, gates, camera objects
- `src/sensors/` - IMU, RGB camera sensors
- `src/envs/` - Environment interfaces (base, quadrotor, vectorized)
- `src/bridges/` - Unity ZMQ communication bridge
- `src/wrapper/` - Python bindings via pybind11
- `src/common/` - Utilities (math, types, integrators)

**Key Classes:**
- `QuadrotorDynamics`: Physics simulation (modular thrust/drag models)
- `Quadrotor`: Quadrotor object with sensors and state
- `QuadrotorEnv`: Single environment interface
- `VecEnv`: Vectorized environment for parallel training
- `UnityBridge`: ZMQ-based communication with Unity renderer

### Python RL Interface

**flightrl Structure:**
- `rpg_baselines/envs/` - Gym wrappers for flightlib environments
- `rpg_baselines/ppo/` - PPO implementation
- `examples/` - Training and testing scripts
- `drqv2_src/` - DrQv2 implementation (newer)

**Environment Flow:**
1. `flightgym.QuadrotorEnv_v1` (C++ via pybind11)
2. → `vec_env_wrapper.py` (wraps as Gym environment)
3. → RL algorithm (PPO2, DrQv2)

### Configuration System

Environments are configured via YAML files in `flightlib/configs/`:
- `vec_env.yaml` - Vectorized environment settings (num_envs, render, scene)
- `quadrotor_env.yaml` - Quadrotor parameters, gate positions, reward coefficients

**Gate tracks available:**
- Circle track
- Kidney track (currently active)
- Figure 8 track

### Unity Bridge Communication

**Architecture:**
- Physics (flightlib) runs independently from rendering (Unity)
- Communication via ZMQ sockets (default ports 10253/10254)
- Supports headless mode (render: no in configs)
- Can spawn objects dynamically (gates, cameras)

**Scene IDs:**
- 0: Industrial
- 1: Warehouse (default)
- 2+: Other scenes

## Recent Changes (Git Log Context)

Based on recent commits:
- **Modular dynamics system**: Physics models are now modular
- **RL action handling**: RL actions have been updated
- **Perception rewards**: Fixed bugs in perception reward calculation
- **Gate passing logic**: Bug investigation ongoing (hang when gate is passed)
- **Model architecture**: Corrections made to neural network architecture
- **Environment reset**: Investigating reset behavior when gates are passed

## Development Notes

### Working with Dynamics

The quadrotor dynamics system is **modular**. Key parameters in `quadrotor_env.yaml`:
- Mass and inertia configuration
- Motor positions (X-configuration)
- Thrust polynomial coefficients
- Motor RPM limits
- Drag coefficients

### Adding New Environments

1. Create C++ environment in `flightlib/src/envs/`
2. Inherit from `EnvBase` class
3. Implement required methods (reset, step, getObs, etc.)
4. Add Python bindings in `flightlib/src/wrapper/`
5. Create Gym wrapper in `flightrl/rpg_baselines/envs/`

### RL Reward Engineering

Reward coefficients are in `quadrotor_env.yaml` under `rl:` section:
- `pos_coeff`: Position error penalty
- `ori_coeff`: Orientation error penalty
- `lin_vel_coeff`: Linear velocity penalty
- `ang_vel_coeff`: Angular velocity penalty
- `act_coeff`: Action magnitude penalty

### Unity Rendering

Unity executable is in `flightrender/RPG_Flightmare/` (or custom builds in `custom_unity_build/`).
Start Unity before running any examples that require rendering.

## Common Issues

**"FLIGHTMARE_PATH not set" error:**
```bash
export FLIGHTMARE_PATH=/home/ubuntu/Stavrou_PhD/flightmare_dev
```

**Python version mismatch:**
The project requires **exactly Python 3.7** due to pybind11 compatibility in CMakeLists.txt.

**CMake external files error:**
The setup.py intentionally clears `flightlib/externals/` and `flightlib/build/` to avoid CMake cache issues.

**Unity connection fails:**
- Ensure Unity is running with Flightmare environment loaded
- Check ZMQ ports (10253/10254) are not blocked
- Verify `render: yes` in config if you want visualization

**Gate passing hangs environment:**
Known issue being investigated - related to environment reset logic when gates are passed.
