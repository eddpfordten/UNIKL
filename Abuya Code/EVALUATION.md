# Segmentation evaluation results

The checked results are in `outputs/evaluation/step1_step2_comparison.html` and
the accompanying CSV/JSON files. `notebooks/step1_step2_results.ipynb` reads
those files; it does not train or rescore either model. The same 38 labeled
patients from `Training` (15% split, seed 42) were used for both steps. The
provided `Validation` folder has no segmentation masks, so Dice cannot be
calculated for its cases.

| Model | Mean whole-tumor Dice (38 patients) |
| --- | ---: |
| Zaq PyTorch checkpoint | 0.848413 |
| Hayyi image + radiomics checkpoint | 0.841980 |
| Zaq regular ONNX export | 0.848413 |
| Zaq quantized ONNX | 0.849120 |

Hayyi's supplied radiomics features for the 38 cases were extracted from
ground-truth whole-tumor masks. Because the features enter the segmentation
model, the Step 1 comparison contains validation-label information and is not
a blind comparison of generalization. Do not infer training-time superiority
from the epoch at which each model reached its best checkpoint: elapsed time
per epoch and hardware were not controlled here.

## Reproduce on a machine with the data

Install `requirements-evaluation.txt` in a Python 3.12 environment. The
project's existing `outputs/checkpoints/best_model.pth` is Zaq's checkpoint.
Provide a local BraTS-PEDs-v1 dataset root containing `Training` and set
`--data-root` to that directory. The scripts also contain Zaq's original
Google Drive path as a convenience default, so users on another machine should
pass their own path explicitly.

Hayyi's checkpoint, radiomics CSV, source branch, and saved
`radiomics_stats.json` are **not committed** here. Obtain them from Hayyi and
pass their local paths to `evaluate_hayyi_dice.py`. `--hayyi-root` must point
to the directory containing Hayyi's `brain_tumor_seg` package and
`outputs/checkpoints/radiomics_stats.json`. His `Hayyi` GitHub branch provides
the source code. These assets must correspond to the same trained model and
386-feature CSV used in the checked results.

The committed CSV/JSON/HTML files are reference results. For a new run, use
fresh paths, for example:

```powershell
python evaluate_patient_dice.py --data-root "G:\path\to\BraTS-PEDs-v1" --output "outputs/evaluation/rerun/own_patient_dice.csv"
python evaluate_hayyi_dice.py --data-root "G:\path\to\BraTS-PEDs-v1" --hayyi-root "C:\path\to\Hayyi\Abuya Code" --checkpoint "C:\path\to\hayyi_best_model.pth" --radiomics-csv "C:\path\to\pyradiomics_3d_train_val_split.csv" --own-scores "outputs/evaluation/rerun/own_patient_dice.csv" --output "outputs/evaluation/rerun/hayyi_patient_dice.csv"
python compare_patient_dice.py --own-scores "outputs/evaluation/rerun/own_patient_dice.csv" --hayyi-scores "outputs/evaluation/rerun/hayyi_patient_dice.csv" --output-dir "outputs/evaluation/rerun"
python evaluate_onnx_quantized.py --data-root "G:\path\to\BraTS-PEDs-v1" --step1-scores "outputs/evaluation/rerun/own_patient_dice.csv" --output-dir "outputs/evaluation/rerun/onnx_step2"
python build_model_comparison_report.py --evaluation-dir "outputs/evaluation/rerun" --output "outputs/evaluation/rerun/comparison.html"
```

Each scoring script refuses to replace an existing result. Step 2 uses 16
training-subset patients for static signed INT8 QDQ calibration of convolution
layers, then ONNX Runtime's CPU provider for the 38 holdout patients. It does
not retrain the checkpoint. The saved quantized model is
`outputs/evaluation/onnx_step2/own_segmentation_int8.onnx`; operations outside
the quantized convolutions may remain floating point. The smaller file size
does not establish a speed improvement, which was not benchmarked.
