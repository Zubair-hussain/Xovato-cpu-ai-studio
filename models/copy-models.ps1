$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")
$backendModels = Join-Path $projectRoot "backend\data\models"

function Copy-IfPresent {
  param(
    [string]$Source,
    [string]$Destination
  )

  if (Test-Path -LiteralPath $Source) {
    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -Recurse -Force
    Write-Output "Copied: $Source -> $Destination"
  } else {
    Write-Output "Missing optional model: $Source"
  }
}

Copy-IfPresent `
  -Source (Join-Path $backendModels "segmentation\u2netp.onnx") `
  -Destination (Join-Path $PSScriptRoot "segmentation\u2netp.onnx")

Copy-IfPresent `
  -Source (Join-Path $backendModels "face\yunet.onnx") `
  -Destination (Join-Path $PSScriptRoot "face\yunet.onnx")

Copy-IfPresent `
  -Source (Join-Path $backendModels "face\gpen_bfr_256.onnx") `
  -Destination (Join-Path $PSScriptRoot "face\gpen_bfr_256.onnx")

$omnivoiceSource = Join-Path $backendModels "k2-fsa_OmniVoice"
$omnivoiceDestination = Join-Path $PSScriptRoot "omnivoice\k2-fsa_OmniVoice"
Copy-IfPresent -Source $omnivoiceSource -Destination $omnivoiceDestination

Write-Output "CPU model copy finished."
