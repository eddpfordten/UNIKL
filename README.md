# UNIKL Brain Tumor Segmentation

This repository contains the Abuya Code desktop viewer, segmentation models,
training notebooks, and the Step 1 and Step 2 Dice comparison. The active app
is in [`Abuya Code/Interface/ui_layout.py`](Abuya%20Code/Interface/ui_layout.py).
The older root-level [`ui_layout.py`](ui_layout.py) is a separate standalone
viewer and remains in place.

## Where things are

| Location | Contents |
| --- | --- |
| [`Abuya Code/Interface/`](Abuya%20Code/Interface/) | Current PySide6 viewer and its images/icons |
| [`Abuya Code/brain_tumor_seg/`](Abuya%20Code/brain_tumor_seg/) | Data loading, model, inference, MedSAM2, and visualization code |
| [`Abuya Code/notebooks/`](Abuya%20Code/notebooks/) | Training notebooks and the saved-results notebook |
| [`Abuya Code/outputs/checkpoints/`](Abuya%20Code/outputs/checkpoints/) | Zaq checkpoints and survival statistics used by the app |
| [`Abuya Code/outputs/evaluation/`](Abuya%20Code/outputs/evaluation/) | Checked Dice CSV/JSON results, comparison HTML, ONNX models, and local evaluation assets |
| [`Abuya Code/outputs/plots/`](Abuya%20Code/outputs/plots/) | Saved training plots |
| [`Abuya Code/packaging/`](Abuya%20Code/packaging/) | Windows executable build recipe and instructions |
| [`Abuya Code/tests/`](Abuya%20Code/tests/) | Existing MedSAM2 checks |
| [`reference/legacy/`](reference/legacy/) | Old standalone reference files that the current app does not load |

The Google Drive BraTS-PEDs dataset is external to this repository. The
`Abuya Code/vendor/`, `.venv/`, `build_exe/`, and `dist/` directories are local
setup or build files. The source code expects the existing notebook, checkpoint,
and evaluation paths above, so they have not been reorganized.

## Open the viewer

From PowerShell in the repository root, use the project environment with the
packages in `Abuya Code/requirements.txt`:

```powershell
cd '.\Abuya Code'
python Interface\ui_layout.py
```

The viewer uses Zaq's `outputs/checkpoints/best_model.pth`. MedSAM2 is a
separate refinement model; its pinned setup is in
[`setup_medsam2.ps1`](Abuya%20Code/setup_medsam2.ps1). Its refinement requires
an NVIDIA CUDA GPU. The app does not bundle patient scans; select a patient
folder in the viewer. MedSAM2's local source and checkpoint are excluded from
Git.

The source viewer's report shows recorded survival from the BraTS-PEDs TSV
when that patient has a label. Its **Predicted Survival Days** display is a
model estimate and may differ from the recorded value. The dataset path can
also be set with `BRATS_DATA_ROOT`; the current `G:` Google Drive shortcut is
used when available. The existing executable is a snapshot of the earlier UI
and is not updated by source edits.

The **Run with** selector lets you try Zaq or Hayyi in the source viewer. Zaq
remains the default. Hayyi requires the local `outputs/evaluation/hayyi_best_model.pth`,
`hayyi_radiomics_features.csv`, and
`hayyi_branch/Abuya Code/outputs/checkpoints/radiomics_stats.json` files.
The supplied radiomics rows cover known training cases and were extracted
using their true tumor masks. The viewer refuses to run Hayyi for a case
without a matching row; its displayed result is a demonstration, not a
blind prediction. These large or teammate-provided assets are local and are
not included in Git or the current executable.

## Evaluation results

Open the checked [Step 1 and Step 2 comparison](Abuya%20Code/outputs/evaluation/step1_step2_comparison.html)
or the [results notebook](Abuya%20Code/notebooks/step1_step2_results.ipynb).
See [EVALUATION.md](Abuya%20Code/EVALUATION.md) for the per-patient CSVs,
rerun commands, data requirements, and the radiomics label-leakage limitation.
The reported 38-patient Dice scores use a labeled holdout from `Training`;
the provided `Validation` folder has no masks for calculating Dice.

## Windows executable

The local executable is `Abuya Code/dist/BrainTumorViewer.exe`. It is a large
one-file build and is excluded from Git. To rebuild or check it, follow
[`packaging/BUILD.md`](Abuya%20Code/packaging/BUILD.md). The desktop viewer
uses the PyTorch checkpoint; the quantized ONNX model is an evaluation result,
not the model loaded by this UI.

## Legacy reference files

The image, placeholder HTML, old installation commands, and standalone
confusion-metrics page are in [`reference/legacy/`](reference/legacy/README.md).
The original root-level `ui_layout.py` remains at its old path because it can
be launched directly.
