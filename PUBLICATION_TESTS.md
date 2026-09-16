# Publication test plan

## Primary question

Can an image-space deferred renderer use G-buffer variation and a compact GGX sensitivity proxy to select shading rate while controlling final radiance error better than a conventional attribute-gradient heuristic?

## Minimum baselines

- full-rate deferred GGX reference,
- fixed 2x2,
- fixed 4x4,
- current screen-space heuristic,
- proposed radiance-budget selector,
- closest literature implementation after novelty review.

The current code contains the full-rate, fixed-rate, heuristic and proposed paths.

## Required experiments

1. 16 / 64 / 256 lights.
2. 1080p / 1440p / 4K.
3. rough and glossy material regimes.
4. three camera views.
5. multiple radiance budgets.
6. boundaries and thin geometry.
7. normal-map/high-frequency material stress.
8. camera motion and temporal rate stability.

## Primary metrics

- GPU time of the lighting compute pass,
- linear-HDR RMSE,
- gradient RMSE,
- maximum absolute error,
- budget-violation rate,
- 1x1 / 2x2 / 4x4 occupancy.

A strong result is not just a speedup. The key question is whether the requested budget predicts the *measured* error and whether the proposed selector has fewer violations at matched cost than the heuristic.

## Required ablations

- attribute variation only,
- representative-GGX sensitivity only,
- normal term removed,
- roughness term removed,
- position term removed,
- light-count scaling removed.

## Do not claim yet

- a formal hard upper bound,
- equivalence to hardware VRS,
- perceptual optimality,
- generality beyond the tested deferred PBR pipeline,
- hardware-general speedup from one GPU.
