# Smoke test + remaining model downloads, meant to run DETACHED on the lab PC
# (launched via Win32_Process so it survives SSH disconnects). All output goes
# to the redirect set up by the launcher (D:\structure-over-scale\pilot.log).
$ErrorActionPreference = "Continue"
$repo = "D:\structure-over-scale"
Set-Location $repo

"=== [$(Get-Date -Format s)] serve E4B Q4"
powershell -NoProfile -ExecutionPolicy Bypass -File $repo\scripts\serve.ps1 gemma-4-E4B-it-Q4_0.gguf
if ($LASTEXITCODE -ne 0) { "SERVE_FAILED"; exit 1 }

"=== [$(Get-Date -Format s)] smoke run: 5 tasks, toolset B"
cmd /c ".venv\Scripts\python.exe -u run.py --label e4b-q4 --toolset B --n 5 --out runs_smoke 2>&1"
"RUN_EXIT: $LASTEXITCODE"
"=== summary"
cmd /c ".venv\Scripts\python.exe summarize.py runs_smoke 2>&1"

"=== [$(Get-Date -Format s)] downloading remaining models"
$ProgressPreference = "SilentlyContinue"
$files = @(
    @("gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-BF16.gguf"),
    @("gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q8_0.gguf"),
    @("gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_0.gguf"),
    @("gemma-4-12B-it-GGUF", "gemma-4-12B-it-Q4_0.gguf")
)
foreach ($f in $files) {
    $dst = "D:\models\$($f[1])"
    # curl -C - resumes a partial file from a killed earlier attempt
    curl.exe -sL -C - -o $dst "https://huggingface.co/ggml-org/$($f[0])/resolve/main/$($f[1])"
    "fetched $($f[1]): $([math]::Round((Get-Item $dst).Length/1GB,2)) GB"
}
"=== [$(Get-Date -Format s)] PILOT_DONE"
