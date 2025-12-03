// std
#include <cstring>

// pybind11
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// flightlib
#include "flightlib/envs/env_base.hpp"
#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"
#include "flightlib/envs/test_env.hpp"
#include "flightlib/envs/vec_env.hpp"

namespace py = pybind11;
using namespace flightlib;

PYBIND11_MODULE(flightgym, m) {
  py::class_<VecEnv<QuadrotorEnv>>(m, "QuadrotorEnv_v1")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, const bool>())
    .def("reset", &VecEnv<QuadrotorEnv>::reset)
    .def("step", &VecEnv<QuadrotorEnv>::step)
    .def("testStep", &VecEnv<QuadrotorEnv>::testStep)
    .def("setSeed", &VecEnv<QuadrotorEnv>::setSeed)
    .def("close", &VecEnv<QuadrotorEnv>::close)
    .def("isTerminalState", &VecEnv<QuadrotorEnv>::isTerminalState)
    .def("curriculumUpdate", &VecEnv<QuadrotorEnv>::curriculumUpdate)
    .def("connectUnity", &VecEnv<QuadrotorEnv>::connectUnity)
    .def("disconnectUnity", &VecEnv<QuadrotorEnv>::disconnectUnity)
    .def("getNumOfEnvs", &VecEnv<QuadrotorEnv>::getNumOfEnvs)
    .def("getObsDim", &VecEnv<QuadrotorEnv>::getObsDim)
    .def("getActDim", &VecEnv<QuadrotorEnv>::getActDim)
    .def("getExtraInfoNames", &VecEnv<QuadrotorEnv>::getExtraInfoNames)
    .def("get_rgb_image",
        [](VecEnv<QuadrotorEnv> &vec_env, int env_id, int cam_id) {
          int h = 0, w = 0;
          std::vector<uint8_t> buffer;
          if (!vec_env.getRGBImage(env_id, cam_id, buffer, h, w)) {
            throw std::runtime_error("getRGBImage failed in VecEnv");
          }

          if (h <= 0 || w <= 0 ||
              buffer.size() != static_cast<std::size_t>(h) *
                              static_cast<std::size_t>(w) * 3) {
            throw std::runtime_error("getRGBImage returned invalid size");
          }

          py::array_t<uint8_t> img({h, w, 3});
          std::memcpy(img.mutable_data(), buffer.data(), buffer.size());
          return img;
        },
        py::arg("env_id"), py::arg("cam_id") = 0)
    .def("__repr__", [](const VecEnv<QuadrotorEnv>&) {
      return "RPG Drone Racing Environment";
    });

  py::class_<TestEnv<QuadrotorEnv>>(m, "TestEnv_v0")
    .def(py::init<>())
    .def("reset", &TestEnv<QuadrotorEnv>::reset)
    .def("__repr__", [](const TestEnv<QuadrotorEnv>&) {
      return "Test Env";
    });
}
