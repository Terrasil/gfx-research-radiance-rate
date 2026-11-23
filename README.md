# gfx-research-radiance-rate

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)](https://en.cppreference.com/w/cpp/20)
[![OpenGL 4.6](https://img.shields.io/badge/OpenGL-4.6-5586A4.svg)](https://www.opengl.org/)
[![CMake](https://img.shields.io/badge/CMake-3.25%2B-064F8C.svg)](https://cmake.org/)

A deferred GGX lighting experiment for selecting compute-shader shading rates within 8 × 8 pixel tiles.

[Method](#method) · [Build](#build) · [Controls](#controls) · [Experiments](#experiments) · [References](#references)

## Method

$L^*(x)$ is full-rate lighting with the same G-buffer, BRDF and light set. For a rate $r\in\{1,2,4\}$, each $r\times r$ block reuses the result at its top-left pixel $x_r$. The recorded tile error is the maximum absolute luminance residual.

```math
L_r(x)=L^*(x_r),\qquad
e_r(T)=\max_{x\in T}\left|Y\!\left(L_r(x)-L^*(x)\right)\right|.
```

The lighting proxy evaluates representative albedo, roughness, normal and position deviations using one light. It predicts $\widehat e_4$ and uses $\widehat e_2=0.42\widehat e_4$. The feature-only gate compares the same four G-buffer extents with fixed scaled thresholds. The parallel version uses the same gate and distributes the maximum reduction across the workgroup. Mixed or incomplete tiles fall back to rate 1 in both gates.

## Build

Requirements are CMake 3.25+, a C++20 compiler, Ninja, Python 3 with Jinja2, and a desktop OpenGL 4.6 Core driver. The initial CMake configuration downloads the pinned gfx-research-base revision and its dependencies. A local checkout can be supplied with `-DGFX_RESEARCH_BASE_LOCAL_PATH=../gfx-research-base`.

```powershell
python -m pip install jinja2
cmake -S . -B out/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build out/build --parallel
ctest --test-dir out/build --output-on-failure
```

On Linux, run the same build commands and omit `.exe` when launching an executable.

## Controls

Launch `gfx-research-radiance-rate.exe`. Use **Rate** for method and error budget, **Scene** for the light count.

**Tests** runs the smoke, quick or full publication benchmark. Each GUI run writes to a new subdirectory under `results/publication`.

## Experiments

```powershell
.\out\build\gfx-research-radiance-rate.exe --publication-smoke --output=results/rate-smoke
python tools/analyze_results.py results/rate-smoke
.\out\build\gfx-research-radiance-rate.exe --publication-suite --output=results/rate-paper
python tools/analyze_results.py results/rate-paper
```

The paper profile uses 720p and 1080p, 16, 64 and 256 lights, three camera poses and three error budgets. Both full-rate arms are timed in every block. The direct baseline bypasses the shared adaptive gate and its barriers. Lighting dispatch and its memory barrier are measured with the G-buffer frozen.

| Output | Contents |
|---|---|
| `timings.csv` | Individual elapsed queries for each method and block |
| `quality_cost.csv` | Matched HDR errors, time deltas and proxy error-budget failures on selected coarse tiles |
| `fixed_quality.csv` | Fastest measured setting within fixed relative-RMSE limits for each workload |

Analysis checks full-rate equivalence and the serial-versus-parallel gate equivalence before ranking operating points. The GGX proxy is an empirical error estimate. The study uses compute-shader sampling on regular tiles.

## References

1. Bruce Walter et al. [Microfacet Models for Refraction through Rough Surfaces](https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf). Eurographics Symposium on Rendering, 2007.
2. NVIDIA. [Variable Rate Shading](https://developer.nvidia.com/vrworks/graphics/variablerateshading). Hardware context for related shading-rate approaches.
