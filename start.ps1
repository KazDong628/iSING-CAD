param([int]$Port = 8769)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
python -m contour_agent serve --port $Port
