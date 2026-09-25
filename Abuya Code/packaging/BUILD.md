# Windows desktop executable

`BrainTumorViewer.spec` packages the PySide6 app at
`Interface/ui_layout.py` into `dist/BrainTumorViewer.exe`. The source branch
`zaq-segmentation-fix` is unchanged; packaging files live on `zaq`.

The executable bundles Zaq's existing `best_model.pth`, survival normalization,
the UI images, MedSAM2 source, and the pinned `MedSAM2_latest.pt` checkpoint.
The MedSAM2 source revision is `332f30d420f1d1b08e2a79b3ae6a602458808383`;
the checkpoint SHA-256 is
`c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060`.
The checkpoint is available from the official
[MedSAM2 model repository](https://huggingface.co/wanglab/MedSAM2).
See the [MedSAM2 source license](https://github.com/bowang-lab/MedSAM2/blob/main/LICENSE)
and [model repository license](https://huggingface.co/wanglab/MedSAM2).

## Build

Build on Windows with Python 3.12, CUDA PyTorch, the dependencies in
`requirements.txt`, PyInstaller 6.22.3, and the pinned MedSAM2 source installed.
Use `setup_medsam2.ps1` for the project's supported MedSAM2 setup, or provide
the same source and checkpoint at the paths expected by the spec file. From
`Abuya Code`:

```powershell
python -m PyInstaller --noconfirm --distpath dist --workpath build_exe packaging\BrainTumorViewer.spec
```

The executable runs Zaq's existing PyTorch checkpoint. The smaller ONNX model
from Step 2 is an evaluation artifact and is not used by this desktop viewer.
The current one-file build is about 2.96 GB and can take several minutes to
unpack on each launch. Keep enough free disk space in the user's temporary
directory for extraction.
MedSAM2 refinement requires an NVIDIA CUDA GPU; the regular U-Net path can run
on a CPU. The optional `BRATS_DATA_ROOT` environment variable points the app
at an external BraTS-PEDs-v1 dataset for survival metadata. The dataset itself
is not bundled. Refined masks are written under
`%LOCALAPPDATA%\UNIKL\BrainTumorViewer\refined_masks` so they persist after a
one-file executable exits.

## Check the build

```powershell
.\dist\BrainTumorViewer.exe --self-check .\build_exe\self-check.json
.\dist\BrainTumorViewer.exe --smoke-ui
```

The first command loads both model checkpoints and records the result in JSON.
The second opens the UI briefly and closes it automatically. A clean Windows
machine still needs a compatible NVIDIA driver for MedSAM2, and the executable
should be checked on the actual challenge laptop before presentation.
