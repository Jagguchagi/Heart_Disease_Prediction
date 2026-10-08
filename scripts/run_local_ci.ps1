$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $projectPython)) {
    $projectPython = (Get-Command python -ErrorAction Stop).Source
}

function Invoke-Check {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command $($Arguments -join ' ')"
    }
}

Push-Location $projectRoot
try {
    Invoke-Check -Command $projectPython -Arguments @("-m", "ruff", "check", "src", "tests")
    Invoke-Check -Command $projectPython -Arguments @("-m", "pytest", "-q")
    Invoke-Check -Command $projectPython -Arguments @("-m", "src.eda")
    Invoke-Check -Command $projectPython -Arguments @("-m", "src.train")
}
finally {
    Pop-Location
}