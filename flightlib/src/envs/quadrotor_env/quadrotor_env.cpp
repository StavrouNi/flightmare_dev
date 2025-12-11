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
    goal_state_((Vector<quadenv::kNObs>() << 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0)
                  .finished()) {
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
  act_mean_ = Vector<quadenv::kNAct>::Ones() * (-mass * Gz) / 4;
  act_std_ = Vector<quadenv::kNAct>::Ones() * (-mass * 2 * Gz) / 4;

  // in constructor
  // --- camera ---
  auto rgb_cam = std::make_shared<RGBCamera>();
  Vector<3> B_r_BC(0.0, 0.0, 0.3);
  Matrix<3, 3> R_BC = Quaternion(1.0, 0.0, 0.0, 0.0).toRotationMatrix();
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
  
  // Ensure rotation is identity (flat)
  initial_state.q() = Quaternion(1.0, 0.0, 0.0, 0.0);

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

  if (random) {
    // -----------------------------------------------------------------------
    // "NOISY BOX"  INITIALIZATION LOGIC
    // -----------------------------------------------------------------------
    
    // A. Define the Box Size (The "Start Zone")
    // This creates a box: 2m long (X), 2m wide (Y), 1m tall (Z)
    // Scalar box_x_width = 2.0; 
    // Scalar box_y_width = 2.0;
    // Scalar box_z_width = 1.0;

    // B. Create a local random distribution [-0.5, 0.5]
    // scale the box width.
    // std::uniform_real_distribution<Scalar> dist(-0.5, 0.5);

    // C. Apply Position Noise
    // quad_state_.x(QS::POSX) += dist(random_gen_) * box_x_width;
    // quad_state_.x(QS::POSY) += dist(random_gen_) * box_y_width;
    // quad_state_.x(QS::POSZ) += dist(random_gen_) * box_z_width;

    // Safety check: Don't let noise push it into the floor
    // if (quad_state_.x(QS::POSZ) < 0.1) {
    //     quad_state_.x(QS::POSZ) = 0.5; // Force minimum height
    // }

    // D. Apply Yaw (Rotation) Noise
    // Let the drone face roughly forward, but +/- 30 degrees (approx 0.5 radians)
    Scalar yaw_amplitude = 30.0 * M_PI / 180.0; 
    // Scalar random_yaw = dist(random_gen_) * 2.0 * yaw_amplitude; // dist gives -0.5 to 0.5, so *2 gives -1 to 1

    // Convert Yaw to Quaternion
    // formula: q = [cos(yaw/2), 0, 0, sin(yaw/2)] for pure Z-rotation
    quad_state_.q() = Quaternion(std::cos(0.5 * yaw_amplitude), 0.0, 0.0, std::sin(0.5 * yaw_amplitude));



  } else {
    // Non-random: Perfect, stable hover at the init position
    // Reset orientation to Identity (facing forward, flat)
    quad_state_.q() = Quaternion(1.0, 0.0, 0.0, 0.0);
  }

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

  // convert quaternion to euler angle
  Vector<3> euler_zyx = quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);
  // quaternionToEuler(quad_state_.q(), euler);
  quad_obs_ << quad_state_.p, euler_zyx, quad_state_.v, quad_state_.w;

  obs.segment<quadenv::kNObs>(quadenv::kObs) = quad_obs_;
  return true;
}

Scalar QuadrotorEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  quad_act_ = act.cwiseProduct(act_std_) + act_mean_;
  cmd_.t += sim_dt_;
  cmd_.thrusts = quad_act_;

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