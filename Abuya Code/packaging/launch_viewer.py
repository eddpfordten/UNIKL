"""Entry point for the frozen Windows desktop viewer."""

import json
import os
import sys
import traceback
from pathlib import Path


def bundle_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))


def configure_paths() -> None:
    if not getattr(sys, "frozen", False):
        return
    from brain_tumor_seg import config

    root = bundle_root()
    config.PROJECT_ROOT = root
    config.OUTPUT_DIR = root / "outputs"
    config.CHECKPOINT_DIR = root / "outputs" / "checkpoints"
    config.MEDSAM2_DIR = config.CHECKPOINT_DIR / "medsam2"
    config.MEDSAM2_CHECKPOINT = config.MEDSAM2_DIR / "MedSAM2_latest.pt"
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    config.REFINED_MASK_DIR = local_app_data / "UNIKL" / "BrainTumorViewer" / "refined_masks"
    external_data = os.environ.get("BRATS_DATA_ROOT")
    if external_data:
        config.DATA_ROOT = Path(external_data)
        config.TRAIN_DIR = config.DATA_ROOT / "Training"
        config.VAL_DIR = config.DATA_ROOT / "Validation"
        config.SURVIVAL_METADATA_PATH = config.DATA_ROOT / "BraTS-PEDs_metadata(Survival rate).tsv"


def configure_assets(viewer) -> None:
    if not getattr(sys, "frozen", False):
        return
    assets = bundle_root() / "Interface" / "assets"
    viewer._ASSETS_DIR = assets
    viewer._HEADER_BRAIN_PATH = assets / "header_brain.png"
    viewer._UNIKL_LOGO_PATH = assets / "unikl_logo.png"
    viewer._APP_ICON_PATH = assets / "app_icon.png"
    viewer._APP_ICO_PATH = assets / "app_icon.ico"
    viewer._RUN_ICON_PATH = assets / "icon_run.png"
    viewer.MODEL_PATH = bundle_root() / "outputs" / "checkpoints" / "best_model.pth"


def self_check(viewer, output_path: Path) -> None:
    import torch
    from brain_tumor_seg import config
    from brain_tumor_seg.sam import MedSAM2Refiner

    assets = [viewer._HEADER_BRAIN_PATH, viewer._UNIKL_LOGO_PATH,
              viewer._APP_ICON_PATH, viewer._RUN_ICON_PATH]
    for path in [*assets, viewer.MODEL_PATH, config.MEDSAM2_CHECKPOINT]:
        if not path.is_file():
            raise FileNotFoundError(path)
    viewer._ensure_model()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; MedSAM2 refinement cannot run")
    refiner = MedSAM2Refiner(config.MEDSAM2_CHECKPOINT)
    refiner._load()
    result = {
        "ui_imported": True,
        "assets_found": len(assets),
        "zaq_checkpoint_loaded": True,
        "medsam2_predictor_loaded": True,
        "cuda": torch.cuda.get_device_name(0),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    root = str(bundle_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    configure_paths()
    from Interface import ui_layout as viewer

    configure_assets(viewer)
    if "--self-check" in sys.argv:
        index = sys.argv.index("--self-check")
        if index + 1 >= len(sys.argv):
            raise ValueError("--self-check requires an output JSON path")
        output_path = Path(sys.argv[index + 1])
        try:
            self_check(viewer, output_path)
        except Exception as exc:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps({"error": str(exc), "traceback": traceback.format_exc()}, indent=2) + "\n", encoding="utf-8")
            raise
        return
    if "--smoke-ui" in sys.argv:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        app = QApplication([])
        window = viewer.MainWindow()
        window.show()
        QTimer.singleShot(3000, app.quit)
        sys.exit(app.exec())
    viewer.main()


if __name__ == "__main__":
    main()
