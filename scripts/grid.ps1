# The full main grid: 5 model/precision labels x toolsets A/B/C x 60 tasks.
# Meant to run DETACHED on the lab PC; resumable at every level (run.py skips
# finished instances, finished label/toolset cells rerun as no-ops).
$ErrorActionPreference = "Continue"
$repo = "D:\structure-over-scale"
Set-Location $repo

$pairs = @(
    @("e2b-q4",   "gemma-4-E2B-it-Q4_0.gguf"),
    @("e2b-q8",   "gemma-4-E2B-it-Q8_0.gguf"),
    @("e2b-bf16", "gemma-4-E2B-it-BF16.gguf"),
    @("e4b-q4",   "gemma-4-E4B-it-Q4_0.gguf"),
    @("12b-q4",   "gemma-4-12B-it-Q4_0.gguf")
)
foreach ($p in $pairs) {
    $label, $gguf = $p
    "=== [$(Get-Date -Format s)] serving $label ($gguf)"
    powershell -NoProfile -ExecutionPolicy Bypass -File $repo\scripts\serve.ps1 $gguf
    if ($LASTEXITCODE -ne 0) { "SERVE_FAILED $label"; continue }
    foreach ($ts in "A", "B", "C") {
        "=== [$(Get-Date -Format s)] cell $label/$ts"
        cmd /c ".venv\Scripts\python.exe -u run.py --label $label --toolset $ts --n 60 --out runs 2>&1"
        "CELL_DONE $label/$ts exit=$LASTEXITCODE"
    }
    "LABEL_DONE $label"
}
"=== final summary"
cmd /c ".venv\Scripts\python.exe summarize.py runs 2>&1"
"=== [$(Get-Date -Format s)] GRID_DONE"
