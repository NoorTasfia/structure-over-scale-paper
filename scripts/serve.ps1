# Start (or restart) llama-server for one GGUF, surviving SSH disconnects.
# Usage: powershell -File scripts\serve.ps1 gemma-4-E4B-it-Q4_0.gguf [port]
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [int]$Port = 8000,
    [string]$Bin = "D:\llama\bin\llama-server.exe",
    [string]$ModelDir = "D:\models",
    [int]$Ctx = 32768
)
$ErrorActionPreference = "Stop"
$gguf = Join-Path $ModelDir $Model
if (-not (Test-Path $gguf)) { throw "model not found: $gguf" }

# Stop-Process, not taskkill: under EAP=Stop, PS 5.1 turns native stderr
# ("process not found") into a terminating error.
Stop-Process -Name llama-server -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Win32_Process Create detaches from the SSH session's job object, so the
# server keeps running after the SSH connection that launched it closes.
# No quotes in the command line: cmd /c mangles nested quoting, and none of
# these paths contain spaces. Keep it that way.
if ("$Bin$gguf" -match " ") { throw "paths with spaces are not supported here" }
$cmd = "cmd /c $Bin -m $gguf --n-gpu-layers 99 -c $Ctx --port $Port --jinja > D:\llama\server.out 2> D:\llama\server.err"
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd }
if ($r.ReturnValue -ne 0) { throw "Win32_Process Create failed: $($r.ReturnValue)" }

foreach ($i in 1..60) {
    Start-Sleep -Seconds 3
    try {
        if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2).status -eq "ok") {
            "READY $Model pid=$($r.ProcessId)"
            exit 0
        }
    } catch {}
}
Get-Content D:\llama\server.err -Tail 15 -ErrorAction SilentlyContinue
throw "server did not become healthy within 180s"
