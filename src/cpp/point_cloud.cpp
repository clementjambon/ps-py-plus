#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Eigen/Dense"

#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"

#include "utils.h"

namespace py = pybind11;
namespace ps = polyscope;

// For overloaded functions, with C++11 compiler only
template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

// Signal handler (makes ctrl-c work, etc)
// TODO: move that in utils
void checkSignalsPc() {
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();
  }


// clang-format off
void bind_point_cloud(py::module& m) {

  // == Helper classes
  py::class_<ps::PointCloudPickResult>(m, "PointCloudPickResult")
   .def(py::init<>())
   .def_readonly("index", &ps::PointCloudPickResult::index)
  ;

  // Scalar quantities
  bindScalarQuantity<ps::PointCloudScalarQuantity>(m, "PointCloudScalarQuantity");

  // Color quantities
  bindColorQuantity<ps::PointCloudColorQuantity>(m, "PointCloudColorQuantity");

  // Vector quantities
  bindVectorQuantity<ps::PointCloudVectorQuantity>(m, "PointCloudVectorQuantity");


  // == Main class
  bindStructure<ps::PointCloud>(m, "PointCloud")

    // basics
    .def("update_point_positions", &ps::PointCloud::updatePointPositions<Eigen::MatrixXf>, "Update point positions")
    .def("update_point_positions2D", &ps::PointCloud::updatePointPositions2D<Eigen::MatrixXf>, "Update point positions")
    .def("n_points", &ps::PointCloud::nPoints, "# points")

    // options
    .def("set_radius", &ps::PointCloud::setPointRadius, "Set radius")
    .def("get_radius", &ps::PointCloud::getPointRadius, "Get radius")
    .def("set_color", &ps::PointCloud::setPointColor, "Set color")
    .def("get_color", &ps::PointCloud::getPointColor, "Get color")
    .def("set_material", &ps::PointCloud::setMaterial, "Set material")
    .def("get_material", &ps::PointCloud::getMaterial, "Get material")
    .def("set_point_render_mode", &ps::PointCloud::setPointRenderMode, "Set point render mode")
    .def("get_point_render_mode", &ps::PointCloud::getPointRenderMode, "Get point render mode")

    // picking
    .def("interpret_pick_result", &ps::PointCloud::interpretPickResult)

    // variable radius
    .def("set_point_radius_quantity", 
        overload_cast_<std::string, bool>()(&ps::PointCloud::setPointRadiusQuantity), 
        "Use a scalar to set radius by name", py::arg("quantity_name"), py::arg("autoscale")=true)
    .def("clear_point_radius_quantity", &ps::PointCloud::clearPointRadiusQuantity, "Clear any quantity setting the radius")
    
    // scalar transparency
    .def("set_transparency_quantity", 
        overload_cast_<std::string>()(&ps::PointCloud::setTransparencyQuantity), py::arg("quantity_name"))
    .def("clear_transparency_quantity", &ps::PointCloud::clearTransparencyQuantity)
        
    // quantities
    .def("add_color_quantity", &ps::PointCloud::addColorQuantity<Eigen::MatrixXf>, "Add a color function at points",
        py::arg("name"), py::arg("values"), py::return_value_policy::reference)
    .def("add_scalar_quantity", &ps::PointCloud::addScalarQuantity<Eigen::VectorXf>, "Add a scalar function at points",
        py::arg("name"), py::arg("values"), py::arg("data_type")=ps::DataType::STANDARD, py::return_value_policy::reference)
    .def("add_vector_quantity", &ps::PointCloud::addVectorQuantity<Eigen::MatrixXf>, "Add a vector function at points",
        py::arg("name"), py::arg("values"), py::arg("vector_type")=ps::VectorType::STANDARD, py::return_value_policy::reference)
    .def("add_vector_quantity2D", &ps::PointCloud::addVectorQuantity2D<Eigen::MatrixXf>, "Add a vector function at points",
        py::arg("name"), py::arg("values"), py::arg("vector_type")=ps::VectorType::STANDARD, py::return_value_policy::reference)
    // Custom Callbacks
    .def("set_pick_callback", [](ps::PointCloud& s, const std::function<void(int)>& func) {
        // Create a wrapper which checks signals before calling the passed fuction
        // Captures by value, because otherwise func seems to become invalid. This is probably happening
        // on the Python side, and could be fixed with some Pybind11 keep_alive-s or something, but in
        // lieu of figuring that out capture by value seems fine.
        // See also the atexit() cleanup registered above, which is used to ensure any bound functions get deleted and
        // we can exit cleanly.
        auto wrapperFunc = [=](int surface_pick_id) {
            checkSignalsPc();
            func(surface_pick_id);
        };
        s.setUserPickCallback(wrapperFunc);
    })
    .def("set_hover_callback", [](ps::PointCloud& s, const std::function<void(int)>& func) {
        // Create a wrapper which checks signals before calling the passed fuction
        // Captures by value, because otherwise func seems to become invalid. This is probably happening
        // on the Python side, and could be fixed with some Pybind11 keep_alive-s or something, but in
        // lieu of figuring that out capture by value seems fine.
        // See also the atexit() cleanup registered above, which is used to ensure any bound functions get deleted and
        // we can exit cleanly.
        auto wrapperFunc = [=](int surface_pick_id) {
            checkSignalsPc();
            func(surface_pick_id);
        };
        s.setUserHoverCallback(wrapperFunc);
    });

  // Static adders and getters
  m.def("register_point_cloud", &ps::registerPointCloud<Eigen::MatrixXf>, 
      py::arg("name"), py::arg("values"), "Register a point cloud", py::return_value_policy::reference);
  m.def("register_point_cloud2D", &ps::registerPointCloud2D<Eigen::MatrixXf>, 
      py::arg("name"), py::arg("values"), "Register a point cloud", py::return_value_policy::reference);
  m.def("remove_point_cloud", &polyscope::removePointCloud, "Remove a point cloud by name");
  m.def("get_point_cloud", &polyscope::getPointCloud, "Get a point cloud by name", py::return_value_policy::reference);
  m.def("has_point_cloud", &polyscope::hasPointCloud, "Check for a point cloud by name");

}
// clang-format on
