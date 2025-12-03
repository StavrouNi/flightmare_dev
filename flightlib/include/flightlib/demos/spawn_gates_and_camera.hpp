#pragma once

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/objects/static_gate.hpp"
#include "flightlib/sensors/rgb_camera.hpp"

namespace flightlib::demos {

// Runs a minimal demo: spawn quadrotor + two gates in Warehouse and render.
int runSpawnGatesMinimal();

}  // namespace flightlib::demos
