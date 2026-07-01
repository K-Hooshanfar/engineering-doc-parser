# Run the same quality checks as GitHub Actions CI (locally).
# Usage: .\scripts\check.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$env:PYTHONPATH = "src"

function Invoke-Step {
    param([string]$Name, [scriptblock]$Command)
    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed (exit $LASTEXITCODE)"
    }
}

Invoke-Step "Black (check formatting)" { python -m black --check . }
Invoke-Step "Ruff (lint & import order)" { python -m ruff check . }

Invoke-Step "Pylint (score >= 6.5)" {
    $pylintOut = python -m pylint src/engineering_doc_parser 2>&1
    $pylintOut | Select-Object -Last 15
    $match = $pylintOut | Select-String -Pattern "rated at ([0-9]+(?:\.[0-9]+)?)/10" | Select-Object -Last 1
    if (-not $match) { throw "Could not parse pylint score" }
    $score = [double]$match.Matches[0].Groups[1].Value
    Write-Host "Pylint score: $score/10"
    if ($score -lt 6.5) { throw "Pylint score below 6.5" }
    $global:LASTEXITCODE = 0
}

Invoke-Step "Mypy (type checking)" { python -m mypy src/engineering_doc_parser }

Invoke-Step "Pytest (coverage >= 68%)" {
    python -m pytest --cov=engineering_doc_parser --cov-report=term-missing --cov-fail-under=68
}

Write-Host ""
Write-Host "All CI checks passed." -ForegroundColor Green
