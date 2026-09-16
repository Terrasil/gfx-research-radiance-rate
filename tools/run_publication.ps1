param(
    [string]$Exe = ".\cmake-build-release\gfx-research-radiance-rate.exe",
    [string]$Out = "results\publication",
    [switch]$Quick
)
$ErrorActionPreference = "Stop"
if ($Quick) {
    & $Exe --publication-quick "--output=$Out"
} else {
    & $Exe --publication-suite "--output=$Out"
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python tools\analyze_results.py $Out
