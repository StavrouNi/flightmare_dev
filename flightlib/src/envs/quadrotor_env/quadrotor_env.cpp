#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"
#include <opencv2/core/core.hpp>
#include <cstring>

namespace flightlib {

QuadrotorEnv::QuadrotorEnv()
  : QuadrotorEnv(getenv("FLIGHTMARE_PATH") +
                 std::string("/flightlib/configs/quadrotor_env.yaml")) {}

QuadrotorEnv::QuadrotorEnv(const std::string &cfg_path)
  : EnvBase(),
    pos_coeff_(0.0),
    ori_coeff_(0.0),
    lin_vel_coeff_(0.0),
    ang_vel_coeff_(0.0),
    act_coeff_(0.0),
    goal_state_((Vector<quadenv::kNObs>() << 
             0.0, 0.0, 5.0,  // Goal Position
             0.0, 0.0, 0.0, 1.0, // Goal Quaternion (Identity)
             0.0, 0.0, 0.0,  // Goal Lin Vel
             0.0, 0.0, 0.0   // Goal Ang Vel
             ).finished())
    {
  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);

  // define a bounding box
  world_box_ << -20, 20, -20, 20, 0, 20;
  if (!quadrotor_ptr_->setWorldBox(world_box_)) {
    logger_.error("cannot set wolrd box");
  };

  // define input and output dimension for the environment
  obs_dim_ = quadenv::kNObs;
  act_dim_ = quadenv::kNAct;
  Vector<3> quad_size(0.2, 0.2, 0.2);
  quadrotor_ptr_->setSize(quad_size);
  Scalar mass = quadrotor_ptr_->getMass();
  
  // Action scaling for collective thrust + body rates control
  // Actions from wrapper: [collective_thrust, roll_rate, pitch_rate, yaw_rate]
  // collective_thrust in [0, 1] → maps to [0, max_collective_thrust] m/s²
  // body_rates in [-1, 1] → maps to [-omega_max, omega_max] rad/s
  // 
  // Scaling formula: scaled_action = action * act_std_ + act_mean_
  // For thrust [0,1]: scaled = action * max_thrust + 0 → [0, max_thrust] ✓
  // For rates [-1,1]: scaled = action * omega_max + 0 → [-omega_max, omega_max] ✓
  Scalar max_collective_thrust = quadrotor_ptr_->getDynamics().getForceMax() / mass;  // m/s²
  Vector<3> omega_max = quadrotor_ptr_->getDynamics().getOmegaMax();  // rad/s
  
  act_mean_ = Vector<quadenv::kNAct>::Zero();  // [0, 0, 0, 0]
  act_std_ << max_collective_thrust, omega_max(0), omega_max(1), omega_max(2);

  // in constructor
  // --- camera ---
  auto rgb_cam = std::make_shared<RGBCamera>();
  Vector<3> B_r_BC(0.0, 0.0, 0.3);
  // OLD (Flat):
  // Matrix<3, 3> R_BC = Quaternion(1.0, 0.0, 0.0, 0.0).toRotationMatrix();

  // NEW (30 deg Up-Tilt):
  // After ROS→Unity transform: ROS X→Unity X, ROS Y→Unity Z, ROS Z→Unity Y
  // For Unity pitch-up (around Unity X), we need ROS rotation around X-axis
  // Positive rotation around X pitches "up" in Unity (camera looks up)
  // 30° = 0.5236 rad: cos(15°)=0.966, sin(15°)=0.259
  // Matrix<3, 3> R_BC = Quaternion(0.966, 0.259, 0.0, 0.0).toRotationMatrix();
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
  
  rgb_cam->setFOV(90);
  rgb_cam->setWidth(84);
  rgb_cam->setHeight(84);
  rgb_cam->setRelPose(B_r_BC, R_BC);
  rgb_cam->setPostProcesscing(std::vector<bool>{false, false, false});
  quadrotor_ptr_->addRGBCamera(rgb_cam);
  rgb_cameras_.push_back(rgb_cam);
  // --- camera end ---

  // --- gates ---
  gates_.clear();
    std::string prefab_id = "rpg_gate"; 

    if (cfg_["quadrotor_env"]["gates"]) {
      const YAML::Node& gate_list = cfg_["quadrotor_env"]["gates"];
      for (size_t i = 0; i < gate_list.size(); i++) {
        std::vector<Scalar> pos = gate_list[i]["pos"].as<std::vector<Scalar>>();
        Scalar yaw = gate_list[i]["yaw"].as<Scalar>();
        
        std::string gate_name = "unity_gate_" + std::to_string(i);
        auto gate = std::make_shared<StaticGate>(gate_name, prefab_id);
        
        gate->setPosition(Eigen::Vector3f((float)pos[0], (float)pos[1], (float)pos[2]));
        gate->setQuaternion(Quaternion(std::cos(0.5 * yaw), 0.0, 0.0, std::sin(0.5 * yaw)));
        gates_.push_back(gate);
      }
    }
  // --- gates end---

  // load parameters
  loadParam(cfg_);

  QuadState initial_state;
  initial_state.setZero();
  
  // Use the position we just loaded from YAML
  initial_state.p = init_pos_.cast<Scalar>(); 
  initial_state.q(init_quat_);

  // Apply it to the internal physics object
  quadrotor_ptr_->reset(initial_state);
}

QuadrotorEnv::~QuadrotorEnv() {}

bool QuadrotorEnv::reset(Ref<Vector<>> obs, const bool random) {
  quad_state_.setZero();
  quad_act_.setZero();
  // quadrotor_ptr_->setCollision(false);
  // 1. First, set the drone to the safe center point (read from YAML)
  quad_state_.p = init_pos_.cast<Scalar>(); 

  // Always use the quaternion loaded from the YAML config
  quad_state_.q(init_quat_); 
  
  // Apply this calculated state to the dynamics simulator
  quadrotor_ptr_->reset(quad_state_);

  // Reset control commands
  cmd_.t = 0.0;
  cmd_.thrusts.setZero();

  // Obtain observations and return
  getObs(obs);
  return true;
}

bool QuadrotorEnv::getObs(Ref<Vector<>> obs) {
  quadrotor_ptr_->getState(&quad_state_);

  Quaternion q = quad_state_.q();
  Vector<4> q_vec(q.x(), q.y(), q.z(), q.w());
  
  quad_obs_ << quad_state_.p, q_vec, quad_state_.v, quad_state_.w;

  obs.segment<quadenv::kNObs>(quadenv::kObs) = quad_obs_;
  return true;
}

Scalar QuadrotorEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  quad_act_ = act.cwiseProduct(act_std_) + act_mean_;
  cmd_.t += sim_dt_;
  
  // Use collective thrust + body rates control
  cmd_.collective_thrust = quad_act_(0);  // Collective thrust (m/s²)
  cmd_.omega << quad_act_(1), quad_act_(2), quad_act_(3);  // Body rates (rad/s)
  
  // DEBUG: Log every 100 steps AFTER setting the command
  static int debug_counter = 0;
  if (debug_counter % 100 == 0) {
    std::cout << "\n[C++ DEBUG " << debug_counter << "]" << std::endl;
    std::cout << "  act (from wrapper): " << act.transpose() << std::endl;
    std::cout << "  act_std_: " << act_std_.transpose() << std::endl;
    std::cout << "  act_mean_: " << act_mean_.transpose() << std::endl;
    std::cout << "  quad_act_ (scaled): " << quad_act_.transpose() << std::endl;
    std::cout << "  cmd_.collective_thrust (AFTER assignment): " << cmd_.collective_thrust << std::endl;
    std::cout << "  cmd_.omega (AFTER assignment): " << cmd_.omega.transpose() << std::endl;
    std::cout << "  Drone mass: " << quadrotor_ptr_->getMass() << " kg" << std::endl;
    std::cout << "  Gravity accel: 9.81 m/s²" << std::endl;
    std::cout << "  Hover thrust needed: " << 9.81 << " m/s²" << std::endl;
  }
  debug_counter++;
  
  // CRITICAL: Invalidate thrusts vector so isSingleRotorThrusts() returns false
  // This ensures the code uses runFlightCtl() instead of direct motor control
  cmd_.thrusts.setConstant(NAN);
  
  // simulate quadrotor
  quadrotor_ptr_->run(cmd_, sim_dt_);

  // update observations
  getObs(obs);

  Matrix<3, 3> rot = quad_state_.q().toRotationMatrix();

  // ---------------------- reward function design
  // - position tracking
  Scalar pos_reward =
    pos_coeff_ * (quad_obs_.segment<quadenv::kNPos>(quadenv::kPos) -
                  goal_state_.segment<quadenv::kNPos>(quadenv::kPos))
                   .squaredNorm();
  // - orientation tracking
  Scalar ori_reward =
    ori_coeff_ * (quad_obs_.segment<quadenv::kNOri>(quadenv::kOri) -
                  goal_state_.segment<quadenv::kNOri>(quadenv::kOri))
                   .squaredNorm();
  // - linear velocity tracking
  Scalar lin_vel_reward =
    lin_vel_coeff_ * (quad_obs_.segment<quadenv::kNLinVel>(quadenv::kLinVel) -
                      goal_state_.segment<quadenv::kNLinVel>(quadenv::kLinVel))
                       .squaredNorm();
  // - angular velocity tracking
  Scalar ang_vel_reward =
    ang_vel_coeff_ * (quad_obs_.segment<quadenv::kNAngVel>(quadenv::kAngVel) -
                      goal_state_.segment<quadenv::kNAngVel>(quadenv::kAngVel))
                       .squaredNorm();

  // - control action penalty
  Scalar act_reward = act_coeff_ * act.cast<Scalar>().norm();

  Scalar total_reward =
    pos_reward + ori_reward + lin_vel_reward + ang_vel_reward + act_reward;

  // survival reward
  total_reward += 0.1;

  return total_reward;
}

bool QuadrotorEnv::isTerminalState(Scalar &reward) {
  // if (quadrotor_ptr_->getCollision()) { # Something weird happens on gates with this
  //     logger_.info("Quadrotor collided!");
  //     reward = -4.0; 
  //     return true;    
  // }
  Vector<3> p_crash = quad_state_.p;
  if (isCollisionCustomOriented()) {
      logger_.info("Quadrotor collided (Gate) at P=[%.2f, %.2f, %.2f]", p_crash(0), p_crash(1), p_crash(2));
      // logger_.info("Quadrotor collided with gate (OBB)!");
      reward = -4.0; 
      return true;    
  }
  if (quad_state_.x(QS::POSZ) <= 0.02) {
    logger_.info("Quadrotor crashed to the ground!");
    reward = -4.0;
    return true;
  }
  reward = 0.0;
  return false;
}

bool QuadrotorEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["quadrotor_env"]) {
    sim_dt_ = cfg["quadrotor_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["quadrotor_env"]["max_t"].as<Scalar>();
  } else {
    return false;
  }
  if (cfg["quadrotor_env"]["init_pos"]) {
      std::vector<Scalar> pos = cfg["quadrotor_env"]["init_pos"].as<std::vector<Scalar>>();
      init_pos_ << pos[0], pos[1], pos[2];
    } else {
      logger_.warn("No init_pos in YAML, using default [0,0,2.5]");
      init_pos_ << 0.0, 0.0, 2.5;
    }
  if (cfg["quadrotor_env"]["init_yaw"]) {
    Scalar init_yaw = cfg["quadrotor_env"]["init_yaw"].as<Scalar>();
    // Convert yaw (rotation around Z-axis) to Quaternion and store it
    init_quat_ = Quaternion(std::cos(0.5 * init_yaw), 0.0, 0.0, std::sin(0.5 * init_yaw));
    logger_.info("Initial Yaw loaded: %.2f rad. -> Quaternion: [w=%.3f, x=%.3f, y=%.3f, z=%.3f]",
        init_yaw, init_quat_.w(), init_quat_.x(), init_quat_.y(), init_quat_.z());
  } else {
    // Default to identity quaternion (no rotation)
    logger_.warn("No init_yaw in YAML, using default 0.0 rad.");
    init_quat_ = Quaternion(1.0, 0.0, 0.0, 0.0);
  }
  if (cfg["rl"]) {
    // load reinforcement learning related parameters
    pos_coeff_ = cfg["rl"]["pos_coeff"].as<Scalar>();
    ori_coeff_ = cfg["rl"]["ori_coeff"].as<Scalar>();
    lin_vel_coeff_ = cfg["rl"]["lin_vel_coeff"].as<Scalar>();
    ang_vel_coeff_ = cfg["rl"]["ang_vel_coeff"].as<Scalar>();
    act_coeff_ = cfg["rl"]["act_coeff"].as<Scalar>();
  } else {
    return false;
  }
  return true;
}

bool QuadrotorEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && quad_act_.allFinite()) {
    act = quad_act_;
    return true;
  }
  return false;
}

bool QuadrotorEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

bool QuadrotorEnv::getRGBImage(
    int cam_id,
    std::vector<uint8_t>& buffer,
    int& height,
    int& width) const {

  if (cam_id < 0 || cam_id >= static_cast<int>(rgb_cameras_.size())) {
    logger_.error("getRGBImage: invalid cam_id {} (have {} cameras)",
                  cam_id, rgb_cameras_.size());
    return false;
  }

  if (rgb_cameras_.empty() || !rgb_cameras_[cam_id]) {
    logger_.error("getRGBImage: RGB camera not initialized");
    return false;
  }

  cv::Mat img;
  rgb_cameras_[cam_id]->getRGBImage(img);

  if (img.empty()) {
    logger_.error("getRGBImage: received empty image");
    return false;
  }
  if (img.type() != CV_8UC3) {
    logger_.warn("getRGBImage: unexpected image type {}, expected CV_8UC3",
                 img.type());
  }

  height = img.rows;
  width  = img.cols;

  const std::size_t n_bytes =
      static_cast<std::size_t>(height) *
      static_cast<std::size_t>(width) * 3;

  buffer.resize(n_bytes);
  std::memcpy(buffer.data(), img.data, n_bytes);

  return true;
}

void QuadrotorEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
  for (auto& gate : gates_) {
    bridge->addStaticObject(gate);
  }
}

bool QuadrotorEnv::isCollisionCustomOriented() {
    // 1. Configuration
    const Scalar DRONE_RADIUS = 0.1; 
    const Scalar HOLE_HALF_SIZE = 1.0;          // 2m wide hole
    const Scalar OUTER_FRAME_HALF_SIZE = 1.15;  // 15cm thick rim
    const Scalar GATE_THICKNESS_HALF = 0.05;    // 10cm depth

    // 2. Thresholds
    const Scalar Y_DEPTH_THRESH = GATE_THICKNESS_HALF + DRONE_RADIUS;
    const Scalar SAFE_HOLE_LIMIT = HOLE_HALF_SIZE - DRONE_RADIUS;
    const Scalar OUTER_LIMIT = OUTER_FRAME_HALF_SIZE + DRONE_RADIUS;

    // 3. Current State
    const Vector<3> P_W = quad_state_.p;

    for (const auto& gate : gates_) {
        const Vector<3> G_pos_W = gate->getPosition().cast<Scalar>();
        // Using toRotationMatrix guarantees clean rotation handling
        const Matrix<3, 3> R_WG = gate->getQuaternion().cast<Scalar>().toRotationMatrix();
        
        // Transform P_W to Local Gate Frame (P_G)
        // Local X=Right, Y=Normal(Depth), Z=Up
        const Vector<3> P_G = R_WG.transpose() * (P_W - G_pos_W);
        const Vector<3> dist = P_G.cwiseAbs(); 

        // CHECK 1: Y-AXIS (Depth) - Are we inside the gate's "slice"?
        if (dist(1) < Y_DEPTH_THRESH) {
            
            // Condition A: Are we INSIDE the Safe Hole?
            bool inside_hole_x = (dist(0) < SAFE_HOLE_LIMIT);
            bool inside_hole_z = (dist(2) < SAFE_HOLE_LIMIT);

            if (inside_hole_x && inside_hole_z) {
                continue; // CLEAN PASS -> Next gate
            }

            // Condition B: Are we hitting the Frame?
            // We crash if we are OUTSIDE the hole, but INSIDE the outer dimensions
            bool inside_outer_x = (dist(0) < OUTER_LIMIT);
            bool inside_outer_z = (dist(2) < OUTER_LIMIT);
            
            if (inside_outer_x && inside_outer_z) {
                // LOG THE LOCAL COORDS so we know exactly where we hit
                logger_.info("Hit Gate Rim! Local: [%.2f, %.2f, %.2f]", P_G(0), P_G(1), P_G(2));
                return true; 
            }
        }
    }
    return false;
}

std::ostream &operator<<(std::ostream &os, const QuadrotorEnv &quad_env) {
  os.precision(3);
  os << "Quadrotor Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "act dim =            [" << quad_env.act_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n"
     << "act_mean =           [" << quad_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << quad_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << quad_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << quad_env.obs_std_.transpose() << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib