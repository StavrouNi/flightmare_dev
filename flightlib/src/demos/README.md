# Spawn Gates Demo

This demo spawns a quadrotor with an RGB camera and two static gates in the Unity Warehouse scene.

## Files

- `spawn_gates_and_camera.cpp` - Main demo implementation
- `spawn_gates_and_camera.hpp` - Header with function declarations
- `CMakeLists.txt` - Build configuration

## Building

From the flightlib root directory:

```bash
cd /home/ubuntu/Stavrou_PhD/flightmare_dev/flightlib/build
cmake .. -DBUILD_DEMOS=ON
make spawn_gates_and_camera
```

The executable will be created in the `build/` directory.

## Running

1. **Start Unity** with the Flightmare environment (Warehouse scene)

2. **Run the demo**:
```bash
cd /home/ubuntu/Stavrou_PhD/flightmare_dev/flightlib/build
./spawn_gates_and_camera
```

## What it does

1. Creates a quadrotor with an RGB camera attached
2. Spawns two static gates at positions:
   - Gate 1: (0, 10, 2.5)
   - Gate 2: (0, -10, 2.5)
3. Connects to Unity Warehouse scene via ZMQ bridge
4. Runs a render loop to keep the visualization active

## Dependencies

- flightlib (main library)
- Eigen3 (>= 3.3.4)
- OpenCV
- yaml-cpp
- ZMQ & ZMQPP
- C++17 compiler

## Quick Start

```bash
# 1. Build the demo
cd /home/ubuntu/Stavrou_PhD/flightmare_dev/flightlib/build
cmake .. -DBUILD_DEMOS=ON
make spawn_gates_and_camera

# 2. Start Unity (in separate terminal/window)
# Open Unity and load Flightmare with Warehouse scene

# 3. Run the demo
./spawn_gates_and_camera
```

## Troubleshooting

**Unity connection fails:**
- Make sure Unity is running with Flightmare environment loaded
- Check that the ZMQ ports (default 10253/10254) are not blocked

**Build errors:**
- Ensure flightlib is built first: `cd ../../build && make flightlib`
- Check that all dependencies are installed

