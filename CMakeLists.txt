cmake_minimum_required(VERSION 3.29)  # 3.28도 가능하나 3.29 권장
project(PowerEngine LANGUAGES CXX)

# C++20 + 모듈 스캔
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

# --- ImGui ---
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)

set(IMGUI_SOURCES
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/imgui_demo.cpp
  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)

add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui PUBLIC
  ${IMGUI_DIR}
  ${IMGUI_DIR}/backends
)

# 네가 이미 find_package로 잡은 glfw / Vulkan 사용
target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)

# (권장) PCH 충돌 피하려면 ImGui 소스만 PCH 스킵
set_source_files_properties(${IMGUI_SOURCES} PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
# Vulkan / GLFW 찾기
find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)

# 실행 파일
file(GLOB SRC CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(PowerEngine ${SRC})

# preamble.hpp를 PCH로 지정
target_precompile_headers(PowerEngine PRIVATE
    # 컴파일러가 C++일 때만 적용 (C/ObjC와 구분)
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/src/preamble.hpp>"
)

# Additional Options
if (MSVC)
  target_compile_options(PowerEngine PRIVATE 
    $<$<CONFIG:Debug>:/Od /RTC1>                         # Debug
    $<$<CONFIG:Release>:/O2 /Ob2 /DNDEBUG>               # Release
    $<$<CONFIG:RelWithDebInfo>:/O2 /Ob2 /DNDEBUG>        # RelWithDebInfo
    $<$<CONFIG:MinSizeRel>:/O1 /DNDEBUG>                 # MinSizeRel
  )
  add_custom_command(TARGET PowerEngine POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:PowerEngine>/textures"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_SOURCE_DIR}/textures"
          "$<TARGET_FILE_DIR:PowerEngine>/textures"  
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:PowerEngine>/shaders"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_SOURCE_DIR}/shaders"
          "$<TARGET_FILE_DIR:PowerEngine>/shaders"
          COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:PowerEngine>/models"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_SOURCE_DIR}/models"
          "$<TARGET_FILE_DIR:PowerEngine>/models"
)
endif()

# ---- Vulkan C++ Module 타깃 만들기 ----
# 비어있는 라이브러리 타깃을 만들고, 거기에 모듈 인터페이스 파일을 FILE_SET로 등록
add_library(VulkanCppModule STATIC)
add_library(Vulkan::cppm ALIAS VulkanCppModule)

# 모듈이 Vulkan C 헤더에 의존하므로 포함/링크 지정
target_link_libraries(VulkanCppModule PUBLIC Vulkan::Vulkan)
target_include_directories(VulkanCppModule PRIVATE "${Vulkan_INCLUDE_DIR}")

# 필요 시 HPP 설정 매크로들
target_compile_definitions(VulkanCppModule PUBLIC
  VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
  VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
)

# 가장 중요한 부분: 모듈 인터페이스 파일 등록
# vcpkg의 vulkan-headers가 제공하는 vulkan.cppm을 모듈로 등록
target_sources(VulkanCppModule
  PUBLIC
    FILE_SET cxx_modules TYPE CXX_MODULES
    BASE_DIRS "${Vulkan_INCLUDE_DIR}"
    FILES     "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm")

# ---- 실행 파일이 모듈을 실제로 사용하도록 링크 ----
target_link_libraries(PowerEngine
  PRIVATE
    Vulkan::cppm      # ← 이걸 잊으면 import 해석 안 됨
    glfw
    imgui
    tinyobjloader::tinyobjloader
)

function (add_slang_shader_target TARGET)
  cmake_parse_arguments ("SHADER" "" "" "SOURCES" ${ARGN})
  set (SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
  set (ENTRY_POINTS -entry vertMain -entry fragMain)
  add_custom_command (
          OUTPUT ${SHADERS_DIR}
          COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
  )
  add_custom_command (
          OUTPUT  ${SHADERS_DIR}/slang.spv
          COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
          WORKING_DIRECTORY ${SHADERS_DIR}
          DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
          COMMENT "Compiling Slang Shaders"
          VERBATIM
  )
  add_custom_target (${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()

set(SLANGC_EXECUTABLE "F:/VulkanSDK/1.4.321.1/Bin/slangc.exe")
set(SHADER_SLANG_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang
)
add_slang_shader_target( foo SOURCES ${SHADER_SLANG_SOURCES})
add_dependencies(PowerEngine foo)