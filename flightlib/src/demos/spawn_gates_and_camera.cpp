#include "flightlib/demos/spawn_gates_and_camera.hpp"
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <sstream>

namespace flightlib::demos {

int runSpawnGatesMinimal() {
  using namespace flightlib;

  // --- 1. Create quadrotor and attach an RGB camera ---
  auto quad = std::make_shared<Quadrotor>();
  auto cam  = std::make_shared<RGBCamera>();

  // camera pose in body frame
  // The Unity Asset is rotated +90 (Left). 
  // We rotate the camera -90 (Right) to align it with Physics Forward (+X).
// 1. Yaw Fix: -90 deg around Z (Corrects the "Left-Looking" Asset)
  //    w = cos(-45) = 0.707, z = sin(-45) = -0.707
  // Quaternion q_yaw = Quaternion(0.707, 0.0, 0.0, -0.707);

  // 2. Pitch Fix: 30 deg Up-Tilt
  //    You found that rotating around X (0.259) gave the correct Up-Tilt.
  Quaternion q_pitch = Quaternion(0.966, 0.259, 0.0, 0.0);

  // 3. Combine: Apply Pitch first, then Yaw correction
  Quaternion q_cam_mount = q_pitch; 
  
  Matrix<3, 3> R_BC = q_cam_mount.toRotationMatrix();
  
  // Standard offset
  Vector<3> B_r_BC(0.0, 0.0, 0.3);
  cam->setFOV(90);
  cam->setWidth(720);
  cam->setHeight(480);
  cam->setRelPose(B_r_BC, R_BC);
  cam->setPostProcesscing(std::vector<bool>{false, false, false});
  quad->addRGBCamera(cam);
  Vector<3> scale_vec(2.0, 2.0, 2.0);
  quad->setSize(scale_vec);

  // initialize quad state (placed a bit behind first gate)
  QuadState qs;
  qs.setZero();
  qs.p = Vector<3>(-10.0, 0.0, 2.5);           // position in world
  qs.q() = Quaternion(1.0, 0.0, 0.0, 0.0);     // no rotation
  quad->reset(qs);

  // --- 2. Create a couple of static gates ---
  std::string prefab_id = "rpg_gate";  // must exist in Assets/Resources

  auto gate_1 = std::make_shared<StaticGate>("unity_gate_1", prefab_id);
  gate_1->setPosition(Eigen::Vector3f(0.0f, 2.0f, 2.5f));
  
  // Rotate gate_1 by +180 degrees around Z axis
  Scalar gate_yaw = M_PI;  // +180 degrees
  gate_1->setQuaternion(
    Quaternion(std::cos(0.5 * M_PI_2), 0.0, 0.0, std::sin(0.5 * M_PI_2)));

  // auto gate_2 = std::make_shared<StaticGate>("unity_gate_2", prefab_id);
  // // gate_2->setPosition(Eigen::Vector3f(0.0f, 2.0f, 2.5f));
  // gate_2->setQuaternion(
  //   Quaternion(std::cos(0.5 * M_PI_2), 0.0, 0.0, std::sin(0.5 * M_PI_2)));

  // --- 3. Unity bridge: connect, register quad + gates ---
  auto bridge = UnityBridge::getInstance();

  bridge->addQuadrotor(quad);
  bridge->addStaticObject(gate_1);
  // bridge->addStaticObject(gate_2);

  bool unity_ready = bridge->connectUnity(UnityScene::WAREHOUSE);
  if (!unity_ready) {
    std::cerr << "Failed to connect to Unity (WAREHOUSE scene)." << std::endl;
    return 1;
  }

  std::cout << "Connected to Unity. Spawning quad + gates in Warehouse."
            << std::endl;

  // --- 4. Simple render loop with image saving and gate tracking ---
  // Add a small movement over time so the drone is not static in Unity.
  // We'll translate it forward along X and add a slow yaw oscillation.
  double t = 0.0;
  const double dt_loop = 0.05;  // 50 ms per loop
  const double forward_speed = 0.5;  // m/s along Y
  const double yaw_amp = 0.6;  // radians amplitude for yaw oscillation
  
  // Image saving parameters
  const int save_every_n_frames = 10;  // Save image every 10 frames
  int frame_count = 0;
  std::string img_folder = "demo_images";
  
  // Create output folder for images
  system(("mkdir -p " + img_folder).c_str());

  while (unity_ready) {
    // move forward
    // qs.p[0] += forward_speed * dt_loop;  // increment X
    qs.p[1] += forward_speed * dt_loop;  // increment Y

// Log vectors every 10 frames
    if (frame_count % 10 == 0) {
        Quaternion q = qs.q();
        
        // Calculate Camera Forward Vector in World Frame
        // Rotate the Camera Mount (R_BC) by the Drone's Body Rotation (q)
        Vector<3> cam_vec_body(1, 0, 0); // Camera looks "Forward" relative to itself
        Vector<3> cam_vec_world = q.toRotationMatrix() * R_BC * cam_vec_body;

        // Calculate Vector to Gate
        Vector<3> gate_pos(0.0, 2.0, 2.5);;
        Vector<3> to_gate = gate_pos - qs.p;

        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Frame " << frame_count << std::endl;
        std::cout << "  Drone Yaw (Phys): 0.0 (Facing +X)" << std::endl;
        std::cout << "  Cam Vector:   [" << cam_vec_world.x() << ", " << cam_vec_world.y() << "]" << std::endl;
        std::cout << "  Gate Vector:  [" << to_gate.x() << ", " << to_gate.y() << "]" << std::endl;
        
        // CHECK ALIGNMENT
        if (cam_vec_world.x() > 0.9 && to_gate.x() > 0) {
             std::cout << "  STATUS: ALIGNED! Camera sees East (+X)." << std::endl;
        } else {
             std::cout << "  STATUS: MISALIGNED." << std::endl;
        }
        std::cout << "------------------------------------------------" << std::endl;
    }
    double yaw = yaw_amp * std::sin(t);
    qs.q() = Quaternion(std::cos(0.5 * yaw), 0.0, 0.0, std::sin(0.5 * yaw));

    quad->setState(qs);

    bridge->getRender(0);
    bridge->handleOutput();

    // --- Extract and save RGB image from camera every N frames ---
    if (frame_count % save_every_n_frames == 0) {
      cv::Mat img;
      cam->getRGBImage(img);
      
      if (!img.empty()) {
        std::ostringstream filename;
        filename << img_folder << "/frame_" 
                 << std::setw(6) << std::setfill('0') << frame_count 
                 << ".png";
        cv::imwrite(filename.str(), img);
        std::cout << "  -> Saved image: " << filename.str() << std::endl;
      } else {
        std::cout << "  -> Warning: Empty image at frame " << frame_count << std::endl;
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    t += dt_loop;
    frame_count++;
  }

  return 0;
}

}  // namespace flightlib::demos

// Standalone executable entry point
int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  return flightlib::demos::runSpawnGatesMinimal();
}
