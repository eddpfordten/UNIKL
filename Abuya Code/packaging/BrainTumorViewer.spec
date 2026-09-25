"""PyInstaller one-file build for the Abuya Code Windows desktop viewer."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


project = Path(SPEC).resolve().parent.parent
medsam_source = project / "vendor" / "MedSAM2"
sam_data, sam_binaries, sam_imports = collect_all("sam2", include_py_files=True)
data = [
    (str(project / "Interface" / "assets"), "Interface/assets"),
    (str(project / "outputs" / "checkpoints" / "best_model.pth"), "outputs/checkpoints"),
    (str(project / "outputs" / "checkpoints" / "survival_stats.json"), "outputs/checkpoints"),
    (str(project / "outputs" / "checkpoints" / "medsam2" / "MedSAM2_latest.pt"), "outputs/checkpoints/medsam2"),
    *sam_data,
]

a = Analysis(
    [str(project / "packaging" / "launch_viewer.py")],
    pathex=[str(project), str(medsam_source)],
    binaries=sam_binaries,
    datas=data,
    hiddenimports=sam_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "PyQt6", "PyQt5", "PySide2",
        "IPython", "ipywidgets", "jupyter", "notebook", "nbformat",
        "pandas", "pyarrow", "onnx", "onnxruntime", "transformers",
        "tensorflow", "torchaudio", "timm", "sklearn", "cv2",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="BrainTumorViewer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(project / "Interface" / "assets" / "app_icon.ico"),
)
