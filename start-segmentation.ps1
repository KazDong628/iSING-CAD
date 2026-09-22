param(
    [int]$Port = 8769,
    [string]$Checkpoint = 'runtime/segmentation/runs/unet-r18-dxf-v2/best.pt',
    [string]$Manifest = 'runtime/segmentation/gt-data-v3/manifest.json'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$segPython = Join-Path $PSScriptRoot '.venv-seg/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $segPython -PathType Leaf)) {
    throw 'Segmentation environment missing. Follow docs/deep-segmentation.md.'
}
if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw 'Segmentation checkpoint missing. Train a model first.'
}
$env:PYTHONIOENCODING = 'utf-8'
$env:CONTOUR_SEGMENTATION_CHECKPOINT = (Resolve-Path -LiteralPath $Checkpoint).Path
$env:CONTOUR_SEGMENTATION_MANIFEST = (Resolve-Path -LiteralPath $Manifest).Path
& $segPython -m contour_agent serve --port $Port
exit $LASTEXITCODE
