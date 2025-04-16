#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Eigen/Dense"

#include "polyscope/polyscope.h"

#include "polyscope/floating_quantities.h"
#include "polyscope/image_quantity.h"

#include "utils.h"

namespace py = pybind11;
namespace ps = polyscope;

using Vec2T = std::tuple<float, float>;
ImVec2 imgui_to_vec2(const Vec2T& v) { return ImVec2(std::get<0>(v), std::get<1>(v)); }

// clang-format off
void bind_floating_quantities(py::module& m) {

  // == Global floating quantity management

  // global floating quantity structure
  bindStructure<ps::FloatingQuantityStructure >(m, "FloatingQuantityStructure");
  m.def("get_global_floating_quantity_structure", &ps::getGlobalFloatingQuantityStructure, py::return_value_policy::reference);

  m.def("remove_floating_quantity", &ps::removeFloatingQuantity);
  m.def("remove_all_floating_quantities", &ps::removeAllFloatingQuantities);

  // == Image floating quantities
  
  auto qScalarImage = bindScalarQuantity<ps::ScalarImageQuantity>(m, "ScalarImageQuantity");
  addImageQuantityBindings(qScalarImage);

  auto qColorImage = bindColorQuantity<ps::ColorImageQuantity>(m, "ColorImageQuantity");
  addImageQuantityBindings(qColorImage);
  qColorImage.def("set_is_premultiplied", &ps::ColorImageQuantity::setIsPremultiplied);
  qColorImage.def("imgui_image", 
        [](ps::ColorImageQuantity& quantity, int w, int h, const Vec2T& uv0, const Vec2T& uv1) {
            return quantity.imguiImage(w, h, imgui_to_vec2(uv0), imgui_to_vec2(uv1));
        },
        py::arg("w"),
        py::arg("h"),
        py::arg("uv0") = std::make_tuple(0.f, 0.f),
        py::arg("uv1") = std::make_tuple(1.f, 1.f)
    );


  // global / free-floating adders
  m.def("add_scalar_image_quantity", &ps::addScalarImageQuantity<Eigen::VectorXf>, 
        py::return_value_policy::reference);
  m.def("add_color_image_quantity", &ps::addColorImageQuantity<Eigen::MatrixXf>, 
        py::return_value_policy::reference);
  m.def("add_color_alpha_image_quantity", &ps::addColorAlphaImageQuantity<Eigen::MatrixXf>,
        py::return_value_policy::reference);

  // == Render image floating quantities
 
  auto qDepthRenderImage = bindQuantity<ps::DepthRenderImageQuantity>(m, "DepthRenderImageQuantity"); 
  qDepthRenderImage.def("set_material", &ps::DepthRenderImageQuantity::setMaterial, "Set material");
  qDepthRenderImage.def("set_color", &ps::DepthRenderImageQuantity::setColor, "Set color");
  qDepthRenderImage.def("set_transparency", &ps::DepthRenderImageQuantity::setTransparency, "Set transparency");
  qDepthRenderImage.def("set_allow_fullscreen_compositing", &ps::DepthRenderImageQuantity::setAllowFullscreenCompositing);

  auto qColorRenderImage = bindColorQuantity<ps::ColorRenderImageQuantity>(m, "ColorRenderImageQuantity");
  qColorRenderImage.def("set_material", &ps::ColorRenderImageQuantity::setMaterial, "Set material");
  qColorRenderImage.def("set_transparency", &ps::ColorRenderImageQuantity::setTransparency, "Set transparency");
  qColorRenderImage.def("set_allow_fullscreen_compositing", &ps::ColorRenderImageQuantity::setAllowFullscreenCompositing);

  auto qScalarRenderImage = bindScalarQuantity<ps::ScalarRenderImageQuantity>(m, "ScalarRenderImageQuantity");
  qScalarRenderImage.def("set_material", &ps::ScalarRenderImageQuantity::setMaterial, "Set material");
  qScalarRenderImage.def("set_transparency", &ps::ScalarRenderImageQuantity::setTransparency, "Set transparency");
  qScalarRenderImage.def("set_allow_fullscreen_compositing", &ps::ScalarRenderImageQuantity::setAllowFullscreenCompositing);
  
  auto qRawColorRenderImage = bindColorQuantity<ps::RawColorRenderImageQuantity>(m, "RawColorRenderImageQuantity");
  qRawColorRenderImage.def("set_transparency", &ps::RawColorRenderImageQuantity::setTransparency, "Set transparency");
  qRawColorRenderImage.def("set_allow_fullscreen_compositing", &ps::RawColorRenderImageQuantity::setAllowFullscreenCompositing);
  
  auto qRawColorAlphaRenderImage = bindColorQuantity<ps::RawColorAlphaRenderImageQuantity>(m, "RawColorAlphaRenderImageQuantity");
  qRawColorAlphaRenderImage.def("set_transparency", &ps::RawColorAlphaRenderImageQuantity::setTransparency, "Set transparency");
  qRawColorAlphaRenderImage.def("set_is_premultiplied", &ps::RawColorAlphaRenderImageQuantity::setIsPremultiplied);
  qRawColorAlphaRenderImage.def("set_allow_fullscreen_compositing", &ps::RawColorAlphaRenderImageQuantity::setAllowFullscreenCompositing);
  
  // global / free-floating adders
  m.def("add_depth_render_image_quantity", &ps::addDepthRenderImageQuantity<Eigen::VectorXf, Eigen::MatrixXf>, 
      py::return_value_policy::reference);
  m.def("add_color_render_image_quantity", &ps::addColorRenderImageQuantity<Eigen::VectorXf, Eigen::MatrixXf, Eigen::MatrixXf>, 
      py::return_value_policy::reference);
  m.def("add_scalar_render_image_quantity", &ps::addScalarRenderImageQuantity<Eigen::VectorXf, Eigen::MatrixXf, Eigen::VectorXf>, 
      py::return_value_policy::reference);
  m.def("add_raw_color_render_image_quantity", &ps::addRawColorRenderImageQuantity<Eigen::VectorXf, Eigen::MatrixXf>, 
      py::return_value_policy::reference);
  m.def("add_raw_color_alpha_render_image_quantity", &ps::addRawColorAlphaRenderImageQuantity<Eigen::VectorXf, Eigen::MatrixXf>, 
      py::return_value_policy::reference);

}

// clang-format on
