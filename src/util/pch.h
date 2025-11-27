#pragma once

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <limits>
#include <array>
#include <cassert>
#include <chrono>
#include <unordered_map>
#include <random>
#include <string>
#include <filesystem>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RIGHT_HANDED
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>   // glm::pi, glm::two_pi
#include <glm/gtc/matrix_transform.hpp> // glm::rotate
#include <glm/gtx/normalize_dot.hpp>
#include <glm/gtx/norm.hpp>

constexpr int      MAX_FRAMES_IN_FLIGHT = 2;

inline const std::vector<const char*> validation_layers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

#include <tiny_gltf.h>

#include <ktx.h>
