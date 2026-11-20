# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — UNIKL Brain Tumor Viewer (release build, no MedSAM2)."""

import sys
from pathlib import Path

project = Path(SPEC).resolve().parent.parent
packaging = Path(SPEC).resolve().parent

datas = [
    (str(project / "Interface" / "assets"), "Interface/assets"),
    (str(project / "outputs" / "checkpoints" / "best_model.pth"), "outputs/checkpoints"),
    (str(project / "outputs" / "checkpoints" / "survival_stats.json"), "outputs/checkpoints"),
]

a = Analysis(
    [str(packaging / "launch_viewer.py")],
    pathex=[str(project)],
    binaries=[],
    datas=datas,
    hiddenimports=["PySide6", "nibabel", "torch", "numpy", "matplotlib", "plotly", "skimage"],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "PyQt6", "PyQt5", "PySide2",
        "IPython", "ipywidgets", "jupyter", "notebook", "nbformat",
        "pandas", "pyarrow", "onnx", "onnxruntime", "transformers",
        "tensorflow", "torchaudio", "timm", "sklearn", "cv2",
        "sam2",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BrainTumorViewer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(
        packaging / "BrainTumorViewer.icns"
        if sys.platform == "darwin"
        else project / "Interface" / "assets" / "app_icon.ico"
    ),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="BrainTumorViewer",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="BrainTumorViewer.app",
        icon=str(packaging / "BrainTumorViewer.icns"),
        bundle_identifier="edu.unikl.braintumorviewer",
        version="1.0.0",
        info_plist={
            "NSPrincipalClass": "NSApplication",
            "NSHighResolutionCapable": True,
            "CFBundleName": "Brain Tumor Viewer",
            "CFBundleDisplayName": "Brain Tumor Viewer",
            "CFBundleShortVersionString": "1.0.0",
        },
    )
