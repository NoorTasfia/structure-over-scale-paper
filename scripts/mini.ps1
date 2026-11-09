$ErrorActionPreference = "Continue"
$repo = "D:\structure-over-scale"
Set-Location $repo
Remove-Item -Recurse -Force $repo\runs_smoke -ErrorAction SilentlyContinue
"=== [$(Get-Date -Format s)] serve E4B Q4 at 32k"
powershell -NoProfile -ExecutionPolicy Bypass -File $repo\scripts\serve.ps1 gemma-4-E4B-it-Q4_0.gguf
if ($LASTEXITCODE -ne 0) { "SERVE_FAILED"; exit 1 }
foreach ($ts in "A", "B", "C") {
    "=== [$(Get-Date -Format s)] toolset $ts"
    cmd /c ".venv\Scripts\python.exe -u run.py --label e4b-q4 --toolset $ts --n 5 --out runs_smoke 2>&1"
}
"=== summary"
cmd /c ".venv\Scripts\python.exe summarize.py runs_smoke 2>&1"
"=== [$(Get-Date -Format s)] MINI_DONE"
