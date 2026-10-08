$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$targetDirectory = Join-Path $projectRoot "data\heart+disease"
$targetDataset = Join-Path $targetDirectory "processed.cleveland.data"
$datasetUrl = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

if (Test-Path $targetDataset) {
    Write-Host "UCI Cleveland dataset already exists: $targetDataset"
    exit 0
}

if (Test-Path $targetDirectory) {
    throw "The target folder exists but the processed dataset is missing. Check $targetDirectory before downloading so existing files are not overwritten."
}

$tempDirectory = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
$archivePath = Join-Path $tempDirectory "heart-disease.zip"
$extractDirectory = Join-Path $tempDirectory "extracted"

try {
    New-Item -ItemType Directory -Path $extractDirectory -Force | Out-Null
    Invoke-WebRequest -Uri $datasetUrl -OutFile $archivePath
    Expand-Archive -LiteralPath $archivePath -DestinationPath $extractDirectory

    $datasetFile = Get-ChildItem -Path $extractDirectory -Filter "processed.cleveland.data" -Recurse | Select-Object -First 1
    if ($null -eq $datasetFile) {
        throw "The downloaded UCI archive did not contain processed.cleveland.data."
    }

    New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null
    Copy-Item -Path (Join-Path $datasetFile.DirectoryName "*") -Destination $targetDirectory -Recurse

    if (-not (Test-Path $targetDataset)) {
        throw "Dataset extraction completed but the expected file is missing: $targetDataset"
    }

    Write-Host "Downloaded and extracted the UCI Cleveland data to $targetDirectory"
}
finally {
    if (Test-Path $tempDirectory) {
        Remove-Item -Path $tempDirectory -Recurse -Force
    }
}
