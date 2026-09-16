cmake_minimum_required(VERSION 3.25)

project(gfx-research-radiance-rate VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GFX_RESEARCH_BASE_GIT_TAG main CACHE STRING "gfx-research-base Git revision")
set(GFX_RESEARCH_BASE_LOCAL_PATH "" CACHE PATH "Optional local gfx-research-base checkout")

if(GFX_RESEARCH_BASE_LOCAL_PATH)
    add_subdirectory("${GFX_RESEARCH_BASE_LOCAL_PATH}" "${CMAKE_BINARY_DIR}/gfx-research-base")
else()
    if(NOT DEFINED CPM_SOURCE_CACHE)
        set(CPM_SOURCE_CACHE "${CMAKE_CURRENT_SOURCE_DIR}/.cpm-cache" CACHE PATH "CPM source cache")
    endif()
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake")
    set(GFX_RESEARCH_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty" CACHE PATH "Third-party source directory")
    CPMAddPackage(
        NAME gfx_research_base
        GITHUB_REPOSITORY Terrasil/gfx-research-base
        GIT_TAG "${GFX_RESEARCH_BASE_GIT_TAG}"
        SOURCE_DIR "${GFX_RESEARCH_THIRD_PARTY_DIR}/gfx-research-base"
        OPTIONS "GFX_RESEARCH_BASE_BUILD_EXAMPLE OFF"
    )
endif()

if(TARGET gfx::research_base)
    set(GFX_RESEARCH_BASE_TARGET gfx::research_base)
elseif(TARGET gfx-research-base)
    set(GFX_RESEARCH_BASE_TARGET gfx-research-base)
elseif(TARGET gfx::research-base)
    set(GFX_RESEARCH_BASE_TARGET gfx::research-base)
else()
    message(FATAL_ERROR "gfx-research-base was added, but no supported target was exported")
endif()

if(NOT DEFINED GFX_RESEARCH_BASE_THIRD_PARTY_DIR)
    set(GFX_RESEARCH_BASE_THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/thirdparty")
endif()

add_executable(gfx-research-radiance-rate
    src/main.cpp
    src/textured_obj_scene.cpp
)
target_compile_features(gfx-research-radiance-rate PRIVATE cxx_std_20)
target_include_directories(gfx-research-radiance-rate PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${GFX_RESEARCH_BASE_THIRD_PARTY_DIR}/tinyobjloader"
)
target_link_libraries(gfx-research-radiance-rate PRIVATE ${GFX_RESEARCH_BASE_TARGET})
target_compile_definitions(gfx-research-radiance-rate PRIVATE
    GFX_RESEARCH_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(COMMAND gfx_research_configure_executable)
    gfx_research_configure_executable(gfx-research-radiance-rate)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gfx-research-radiance-rate PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(gfx-research-radiance-rate PRIVATE /W4)
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(
        NAME radiance-rate-contract
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_contract.py"
    )
endif()
