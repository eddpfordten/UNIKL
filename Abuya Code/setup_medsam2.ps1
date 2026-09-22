param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$MedSAMRevision = "332f30d420f1d1b08e2a79b3ae6a602458808383"
$CheckpointSha256 = "c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060"
$CheckpointUrl = "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VendorDir = Join-Path $ProjectDir "vendor"
$MedSAMDir = Join-Path $VendorDir "MedSAM2"
$CheckpointDir = Join-Path $ProjectDir "outputs\checkpoints\medsam2"
$CheckpointPath = Join-Path $CheckpointDir "MedSAM2_latest.pt"

if (-not $Python) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $PythonCommand) {
        throw "Python was not found on PATH. Pass -Python with the full path to Python 3.12."
    }
    $Python = $PythonCommand.Source
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python 3.12 was not found at $Python. Pass -Python with its full path."
}
$PythonVersion = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or [string]$PythonVersion -ne "3.12") {
    throw "MedSAM2 setup requires Python 3.12; $Python reports Python $PythonVersion."
}

$TorchReady = & $Python -c "import torch; print(torch.__version__.startswith('2.5.1+cu121') and torch.cuda.is_available())" 2>$null
if ($LASTEXITCODE -ne 0 -or [string]$TorchReady -ne "True") {
    Write-Host "Installing CUDA-enabled PyTorch 2.5.1 (CUDA 12.1)..."
    & $Python -m pip install --upgrade --force-reinstall torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }
} else {
    Write-Host "Compatible CUDA PyTorch is already installed."
}

New-Item -ItemType Directory -Force -Path $VendorDir, $CheckpointDir | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $MedSAMDir ".git"))) {
    git clone https://github.com/bowang-lab/MedSAM2.git $MedSAMDir
    if ($LASTEXITCODE -ne 0) { throw "Could not clone MedSAM2." }
}
git -C $MedSAMDir fetch origin $MedSAMRevision
if ($LASTEXITCODE -ne 0) { throw "Could not fetch the pinned MedSAM2 revision." }
git -C $MedSAMDir checkout --detach $MedSAMRevision
if ($LASTEXITCODE -ne 0) { throw "Could not check out the pinned MedSAM2 revision." }

$env:SAM2_BUILD_CUDA = "0"
& $Python -m pip install -e $MedSAMDir
if ($LASTEXITCODE -ne 0) { throw "MedSAM2 installation failed." }
& $Python -m pip install -r (Join-Path $ProjectDir "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed." }

$ValidCheckpoint = $false
if (Test-Path -LiteralPath $CheckpointPath) {
    $ValidCheckpoint = (Get-FileHash -LiteralPath $CheckpointPath -Algorithm SHA256).Hash.ToLowerInvariant() -eq $CheckpointSha256
}
if (-not $ValidCheckpoint) {
    $PartialPath = "$CheckpointPath.partial"
    Invoke-WebRequest -Uri $CheckpointUrl -OutFile $PartialPath
    $ActualHash = (Get-FileHash -LiteralPath $PartialPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualHash -ne $CheckpointSha256) {
        Remove-Item -LiteralPath $PartialPath
        throw "Checkpoint checksum mismatch. Expected $CheckpointSha256, received $ActualHash."
    }
    Move-Item -Force -LiteralPath $PartialPath -Destination $CheckpointPath
}

& $Python -c "import torch; from sam2.build_sam import build_sam2_video_predictor_npz; assert torch.cuda.is_available(); print('MedSAM2 ready:', torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "MedSAM2 installed, but its CUDA smoke check failed." }

Write-Host "MedSAM2 setup complete."
