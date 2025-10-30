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
# ---- GLSL -> SPIR-V 컴파일러 선택 ----
# Vulkan SDK 설치 시 보통 GLSLC_EXECUTABLE 변수 제공됨. 없으면 glslc 이름으로 탐색.
if (NOT GLSLC_EXECUTABLE)
  find_program(GLSLC_EXECUTABLE NAMES glslc HINTS "${Vulkan_GLSLC_EXECUTABLE}")
endif()

# glslc가 없으면 glslangValidator로 대체 (둘 중 하나만 쓰면 됨)
if (NOT GLSLC_EXECUTABLE)
  find_program(GLSLANG_VALIDATOR NAMES glslangValidator)
  if (NOT GLSLANG_VALIDATOR)
    message(FATAL_ERROR "glslc 또는 glslangValidator를 찾을 수 없습니다.")
  endif()
endif()

function(compile_glsl_shaders TARGET)
  # 사용법: compile_glsl_shaders(my_target SOURCES a.vert b.frag c.comp)
  cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})

  set(SHADERS_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)
  file(MAKE_DIRECTORY ${SHADERS_OUTPUT_DIR})

  set(SPIRV_OUTPUTS "")

  foreach(SRC IN LISTS ARG_SOURCES)
    get_filename_component(BASENAME ${SRC} NAME)       # 파일명.ext
    set(OUT ${SHADERS_OUTPUT_DIR}/${BASENAME}.spv)

    # 확장자로 스테이지 자동 인식 (glslc는 확장자 인식함)
    get_filename_component(EXT ${SRC} EXT)
    string(TOLOWER ${EXT} EXT)

    if (GLSLC_EXECUTABLE)
      add_custom_command(
        OUTPUT ${OUT}
        COMMAND ${GLSLC_EXECUTABLE}
                --target-env=vulkan1.3
                -O -g
                ${SRC}
                -o ${OUT}
        DEPENDS ${SRC}
        COMMENT "Compiling GLSL ${BASENAME} -> ${OUT}"
        VERBATIM
      )
    else()
      # glslangValidator 경로 (확장자 기반으로 자동 스테이지 인식)
      add_custom_command(
        OUTPUT ${OUT}
        COMMAND ${GLSLANG_VALIDATOR}
                -V --target-env vulkan1.3
                -g
                -o ${OUT} ${SRC}
        DEPENDS ${SRC}
        COMMENT "Compiling GLSL ${BASENAME} -> ${OUT}"
        VERBATIM
      )
    endif()

    list(APPEND SPIRV_OUTPUTS ${OUT})
  endforeach()

  add_custom_target(${TARGET} DEPENDS ${SPIRV_OUTPUTS})
endfunction()

# GLSL 소스 나열
set(GLSL_SHADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/model.vert
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/model.frag
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/cloth.vert
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/cloth.frag
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/cloth.comp
)

# 컴파일 타깃 생성
compile_glsl_shaders(glsl_shaders_target SOURCES ${GLSL_SHADERS})

# 실행 파일이 셰이더 빌드 결과에 의존하도록
add_dependencies(PowerEngine glsl_shaders_target)