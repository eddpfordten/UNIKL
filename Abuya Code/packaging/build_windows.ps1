$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  python -m venv .venv
  $py = Join-Path $Root ".venv\Scripts\python.exe"
  & $py -m pip install -U pip
  & $py -m pip install -r requirements.txt "pyinstaller==6.22.3"
}
& $py -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) { & $py -m pip install "pyinstaller==6.22.3" }

Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
& $py -m PyInstaller --noconfirm --distpath dist --workpath build packaging\BrainTumorViewer.spec

$exeDir = Join-Path $Root "dist\BrainTumorViewer"
Copy-Item (Join-Path $exeDir "BrainTumorViewer.exe") (Join-Path $Root "dist\BrainTumorViewer-Windows.exe") -Force
Write-Host "Built onedir at $exeDir (MedSAM2 not bundled in release)"
