from pathlib import Path

root = Path(__file__).resolve().parents[1]
cpp = (root / "src/main.cpp").read_text(encoding="utf-8")
comp = (root / "shaders/adaptive_lighting.comp").read_text(encoding="utf-8")

for token in [
    "Method::FullRate",
    "Method::Fixed2x2",
    "Method::Fixed4x4",
    "Method::Heuristic",
    "Method::RadianceBudget",
    "violationRate",
    "--publication-quick",
    "--publication-suite",
    "savePreviewPng(",
    "saveRgbPfm(",
    "captureIndex_.open(",
    "captureFullProfile_",
]:
    assert token in cpp, token

for token in [
    "shared vec4 sA[64]",
    "representativeErrorBound",
    "uErrorBudget",
    "sRate = 4",
    "sRate = 2",
    "shadeOne",
    "D_GGX",
]:
    assert token in comp, token

assert "validCount == 64" in comp, "geometry/material boundaries must force full rate"
assert "glClearNamedFramebufferfv(gbuffer_.id(), GL_DEPTH" in cpp, "depth-only FBO must be cleared as depth, not depth-stencil"
print("radiance-rate contract: PASS")
