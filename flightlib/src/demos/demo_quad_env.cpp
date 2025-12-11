#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <thread>

// flightlib
#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/types.hpp"

namespace flightlib::demos {

int runQuadEnvDemo() {
  using namespace flightlib;

  std::cout << "=== QuadrotorEnv Demo ===" << std::endl;
  std::cout << "Initializing environment with camera and gates..." << std::endl;

  // Load environment config
  std::string cfg_path = std::string(std::getenv("FLIGHTMARE_PATH")) + 
                         "/flightlib/configs/quadrotor_env.yaml";
  
  // Create QuadrotorEnv (already has camera and gates from config)
  auto quad_env = std::make_shared<QuadrotorEnv>(cfg_path);
  
  // Create Unity bridge and connect
  auto unity_bridge = UnityBridge::getInstance();
  
  // Add environment objects (quadrotor, camera, gates) to Unity
  quad_env->addObjectsToUnity(unity_bridge);
  
  // Connect to Unity with WAREHOUSE scene
  std::cout << "Connecting to Unity (WAREHOUSE scene)..." << std::endl;
  bool connected = unity_bridge->connectUnity(UnityScene::WAREHOUSE);
  
  if (!connected) {
    std::cerr << "Failed to connect to Unity!" << std::endl;
    return 1;
  }
  
  std::cout << "Unity connected successfully!" << std::endl;

  // Reset environment
  Vector<quadenv::kNObs> obs;
  quad_env->reset(obs, false);  // non-random reset (uses init_pos from config)
  
  std::cout << "Initial observation: " << obs.transpose() << std::endl;
  std::cout << "  Position: [" << obs(0) << ", " << obs(1) << ", " << obs(2) << "]" << std::endl;
  std::cout << "  Orientation: [" << obs(3) << ", " << obs(4) << ", " << obs(5) << "]" << std::endl;

  // Action parameters
  // Actions are normalized: 0.0 = hover, positive = more thrust, negative = less
  const int n_steps = 300;
  const int save_every = 20;
  const Scalar sim_dt = 0.02;  // 20ms per step
  
  // Gate position (from config: first gate at [-10, 3, 2.5])
  // Drone starts at [-10, 0, 2.5], so gate is 3m forward in Y direction
  Vector<3> gate_pos(-10.0, 3.0, 2.5);
  
  // Simple control: gradual forward movement with stabilization
  Vector<quadenv::kNAct> action;
  
  std::cout << "\nStarting flight towards front gate at " << gate_pos.transpose() << std::endl;
  std::cout << "Running " << n_steps << " steps..." << std::endl;

  // Create output directory for images
  system("mkdir -p demo_quad_env_images");

  for (int step = 0; step < n_steps; step++) {
    // Get current position from observation
    Vector<3> current_pos(obs(0), obs(1), obs(2));
    Vector<3> to_gate = gate_pos - current_pos;
    Scalar distance = to_gate.norm();
    
    // Proportional control towards gate
    // Normalized actions in [-1, 1]: 0 = hover
    Scalar forward_thrust = 0.0;  // Y direction
    Scalar hover_adjust = 0.0;    // Z direction
    
    if (distance > 0.5) {
      // Move forward (Y+) towards gate
      forward_thrust = std::min(0.5, distance * 0.5);
      
      // Adjust altitude if needed
      Scalar height_error = gate_pos(2) - current_pos(2);
      hover_adjust = height_error * 0.5;
    }
    
    // Gradual ramp-up for first 30 steps
    Scalar ramp_factor = (step < 30) ? (step / 30.0) : 1.0;
    Scalar base_hover_offset = 10.0;  // base offset to maintain hover
    // Construct action: [motor0, motor1, motor2, motor3]
    // For forward movement: increase front motors, decrease rear
    action(0) = (base_hover_offset + forward_thrust + hover_adjust) * ramp_factor;  // front-left
    action(1) = (base_hover_offset + forward_thrust + hover_adjust) * ramp_factor;  // front-right
    action(2) = (base_hover_offset - forward_thrust * 0.5 + hover_adjust) * ramp_factor;  // rear-left
    action(3) = (base_hover_offset - forward_thrust * 0.5 + hover_adjust) * ramp_factor;  // rear-right
    
    // Clamp to [-1, 1]
    action = action.cwiseMax(-1.0).cwiseMin(1.0);
    
    // Step environment
    Scalar reward = quad_env->step(action, obs);
    
    // Render in Unity
    unity_bridge->getRender(step);
    unity_bridge->handleOutput();
    
    // Print progress
    if (step % 20 == 0) {
      std::cout << "Step " << step 
                << " | Pos: [" << current_pos.transpose() << "]"
                << " | Distance to gate: " << distance 
                << " | Reward: " << reward << std::endl;
    }
    
    // Save images periodically
    if (step % save_every == 0) {
      std::vector<uint8_t> img_buffer;
      int height, width;
      
      if (quad_env->getRGBImage(0, img_buffer, height, width)) {
        // Convert buffer to cv::Mat
        cv::Mat img(height, width, CV_8UC3, img_buffer.data());
        
        std::string filename = "demo_quad_env_images/frame_" + 
                              std::to_string(step) + ".png";
        cv::imwrite(filename, img);
        std::cout << "  -> Saved: " << filename << std::endl;
      }
    }
    
    // Check terminal state
    Scalar terminal_reward;
    if (quad_env->isTerminalState(terminal_reward)) {
      std::cout << "\n*** Terminal state reached at step " << step << " ***" << std::endl;
      std::cout << "Terminal reward: " << terminal_reward << std::endl;
      break;
    }
    
    // Small sleep to match real-time
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  
  std::cout << "\nDemo completed!" << std::endl;
  unity_bridge->disconnectUnity();
  
  return 0;
}

}  // namespace flightlib::demos

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  
  // Check FLIGHTMARE_PATH environment variable
  if (!std::getenv("FLIGHTMARE_PATH")) {
    std::cerr << "ERROR: FLIGHTMARE_PATH environment variable not set!" << std::endl;
    std::cerr << "Please set it to the flightmare root directory." << std::endl;
    return 1;
  }
  
  return flightlib::demos::runQuadEnvDemo();
}
