# gfx-research-radiance-rate

Research harness for **deferred shading-rate selection driven by a predicted GGX radiance-error budget**.

The project renders a textured Sponza G-buffer once and compares:

1. full-rate deferred PBR shading,
2. fixed 2x2 and fixed 4x4 shading baselines,
3. a conventional screen-space variation heuristic,
4. a proposed tile-local radiance-error estimator that selects 1x1, 2x2 or 4x4 shading blocks.

The method works entirely after rasterization. It does not alter geometry density and does not require a reference image at runtime.

## Build

```powershell
cmake -S . -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

With a local base checkout:

```powershell
cmake -S . -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DGFX_RESEARCH_BASE_LOCAL_PATH=E:/gfx-research-base
```

## Run

```powershell
.\cmake-build-release\gfx-research-radiance-rate.exe
```

The GUI exposes the method, light count, radiance budget, roughness scaling and a shading-rate visualization.

## Publication suites

Quick:

```powershell
.\cmake-build-release\gfx-research-radiance-rate.exe `
  --publication-quick --output=results/quick
python tools/analyze_results.py results/quick
```

Full:

```powershell
.\cmake-build-release\gfx-research-radiance-rate.exe `
  --publication-suite --output=results/publication
python tools/analyze_results.py results/publication
```

The full suite varies exact offscreen resolution, point-light count, camera preset, roughness scaling and the error budget.

## Method

Each 8x8 compute workgroup first loads G-buffer albedo/roughness, normal and world position into shared memory. Mixed-validity tiles are forced to full rate. For homogeneous tiles, the proposed path measures within-tile attribute excursions and evaluates a small GGX proxy at a representative light to obtain a predicted radiance-error signal. The tile then chooses 1x1, 2x2 or 4x4 shading.

The benchmark records:

- lighting-pass GPU time,
- linear-HDR RMSE and relative RMSE against full-rate shading,
- gradient RMSE,
- maximum error,
- fraction of pixels whose actual luminance error exceeds the requested budget,
- fraction of pixels shaded at each selected rate.

## Important scope

This project is deliberately a starting point, not a finished novelty claim. Multirate shading and error-controlled shading already have prior art. A paper must position the contribution narrowly around **post-raster deferred many-light rate selection from existing G-buffer data** and compare directly with the closest prior methods.
