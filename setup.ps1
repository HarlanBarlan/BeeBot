# Setup script for a fresh machine (laptop, second desktop, etc.)
# Assumes Python 3.14 is already installed and available as `py` or `python`.
# Run this from the project root after `git clone`.

param(
    [string]$VenvName = ".venv",
    [switch]$SkipCudaTorch  # pass -SkipCudaTorch if machine has no NVIDIA GPU
)

$ErrorActionPreference = "Stop"

Write-Host "== BeeBot setup =="
Write-Host "Venv: $VenvName"
Write-Host ""

# 1. Find Python
$python = $null
foreach ($cmd in @("py", "python")) {
    try {
        $version = & $cmd --version 2>&1
        if ($version -match "Python 3\.1[4-9]") {
            $python = $cmd
            Write-Host "Found $cmd -> $version"
            break
        }
    } catch {}
}
if (-not $python) {
    Write-Error "No Python 3.14+ found. Install from https://python.org first (check 'Add to PATH')."
    exit 1
}

# 2. Create venv
if (Test-Path $VenvName) {
    Write-Host "Venv $VenvName already exists — reusing"
} else {
    Write-Host "Creating venv $VenvName..."
    & $python -m venv $VenvName
}

$pip = ".\$VenvName\Scripts\python.exe"

# 3. Upgrade pip, install requirements
Write-Host "Upgrading pip..."
& $pip -m pip install --upgrade pip

Write-Host "Installing requirements.txt (this can take several minutes)..."
& $pip -m pip install -r requirements.txt

# 4. PyTorch with CUDA (unless skipped)
if (-not $SkipCudaTorch) {
    Write-Host "Installing PyTorch + CUDA 12.6 wheel (2.6GB download)..."
    & $pip -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
} else {
    Write-Host "Skipping CUDA install (SkipCudaTorch set)"
}

# 5. Verify GPU
Write-Host ""
Write-Host "== GPU check =="
& $pip test_gpu.py

Write-Host ""
Write-Host "== Setup complete =="
Write-Host "To train: .\$VenvName\Scripts\python.exe -m rl.train_ppo"
Write-Host ""
Write-Host "Note: models/ is git-ignored. To use an existing trained model,"
Write-Host "manually copy the .zip checkpoint into models/ from wherever you"
Write-Host "have the latest one."
