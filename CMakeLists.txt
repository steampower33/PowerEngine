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
find_package(KTX CONFIG REQUIRED)

# 실행 파일
file(GLOB SRC CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(PowerEngine ${SRC})

target_precompile_headers(PowerEngine PRIVATE
    # 컴파일러가 C++일 때만 적용 (C/ObjC와 구분)
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/src/pch.h>"
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
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:PowerEngine>/assets/textures"
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
  "${CMAKE_SOURCE_DIR}/assets/textures"
  "$<TARGET_FILE_DIR:PowerEngine>/assets/textures"
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:PowerEngine>/assets/models"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  "${CMAKE_SOURCE_DIR}/assets/models"
  "$<TARGET_FILE_DIR:PowerEngine>/assets/models"
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:PowerEngine>/shaders"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  "${CMAKE_SOURCE_DIR}/shaders"
  "$<TARGET_FILE_DIR:PowerEngine>/shaders"
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
    Vulkan::cppm
    glfw
    imgui
    KTX::ktx
)

function(compile_slang_shaders TARGET)
  # 함수에 전달된 키워드 인자들을 파싱합니다.
  # SOURCES와 ENTRY_POINTS는 여러 개의 값을 가질 수 있는 리스트입니다.
  cmake_parse_arguments(ARG "" "" "SOURCES;ENTRY_POINTS" ${ARGN})

  # --- 설정 ---
  set(SHADERS_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)
  file(MAKE_DIRECTORY ${SHADERS_OUTPUT_DIR})

  # --- Slangc에 전달할 -entry 플래그 목록 생성 ---
  set(SLANG_ENTRY_FLAGS "")
  # 사용자가 넘겨준 ENTRY_POINTS 목록을 순회합니다.
  foreach(ENTRY_POINT IN LISTS ARG_ENTRY_POINTS)
    # 각 엔트리 포인트 이름 앞에 "-entry"를 붙여서 리스트에 추가합니다.
    # 예: "vertMain" -> "-entry" "vertMain"
    list(APPEND SLANG_ENTRY_FLAGS "-entry" ${ENTRY_POINT})
  endforeach()

  # 생성된 모든 .spv 파일의 경로를 저장할 리스트입니다.
  set(SPIRV_OUTPUTS "")

  # 모든 셰이더 소스 파일을 순회하는 루프
  foreach(SLANG_SOURCE IN LISTS ARG_SOURCES)
    get_filename_component(BASENAME ${SLANG_SOURCE} NAME_WE)
    set(SPIRV_OUTPUT ${SHADERS_OUTPUT_DIR}/${BASENAME}.spv)

    add_custom_command(
            OUTPUT ${SPIRV_OUTPUT}
            COMMAND ${SLANGC_EXECUTABLE}
                    ${SLANG_SOURCE}
                    -target spirv
                    -profile spirv_1_4
                    -emit-spirv-directly
                    -fvk-use-entrypoint-name
                    ${SLANG_ENTRY_FLAGS}   #<-- 여기서 동적으로 생성된 플래그 목록을 사용합니다.
                    -g3
                    -o ${SPIRV_OUTPUT}
            DEPENDS ${SLANG_SOURCE}
            COMMENT "Compiling [${ARG_ENTRY_POINTS}] -> ${BASENAME}.spv"
            VERBATIM
    )
    list(APPEND SPIRV_OUTPUTS ${SPIRV_OUTPUT})
  endforeach()

  # 모든 .spv 파일 생성에 의존하는 메타 타겟을 생성합니다.
  add_custom_target(${TARGET} DEPENDS ${SPIRV_OUTPUTS})
endfunction()

set(SLANGC_EXECUTABLE "F:/VulkanSDK/1.4.321.1/Bin/slangc.exe")

set(GRAPHICS_SHADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/model.slang
)

compile_slang_shaders(graphics_shaders_target
  SOURCES       ${GRAPHICS_SHADERS}
  ENTRY_POINTS  vertexMain fragmentMain
)

set(GRAPHICS_AND_COMPUTE_SHADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/cloth.slang
)
# 모든 엔트리 포인트를 지정합니다.
compile_slang_shaders(particle_shader_target
    SOURCES       ${GRAPHICS_AND_COMPUTE_SHADERS}
    ENTRY_POINTS  vertexMain fragmentMain computeMain
)

add_dependencies(PowerEngine 
    graphics_shaders_target
    particle_shader_target
)