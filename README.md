# UNIKL
UNIKL Studies
Assalamualaikum semua mari kita semua

## Nice to do

Ideas we want to grow into the A.I. Copilot next.

- 🗂️ **Batch volume load** — drop or pick several patient folders at once, not just one scan.
- 📋 **Collapsible batch drawer** — a slim panel that lists every loaded volume and tucks away when you do not need it.
- 🖱️ **Click-to-view cases** — each row in that list opens the matching case in the 2D / 3D viewers.
- 🎯 **SAM assist** — semi-automated tumor outline with Segment Anything, then refine by hand.
- 🧠 **Memory diet** — keep large NIfTI stacks and 3D meshes from eating RAM during long sessions.
- ⬡ **Viewer hive HUD** — a light, irregular honeycomb mark behind each viewer, not a full grid.
- 🌀 **Idle 3D motion** — a default orbit / pulse animation on the tumor and brain views before any image is loaded.
- 💾 **Save after segment** — export the overlay, mask, and 3D snapshots once a run finishes.
- 🧩 **Model picker** — choose which checkpoint / architecture to run before hitting segmentation.

## MedSAM2 refinement

The desktop viewer exposes the 3D U-Net and MedSAM2 as two independent segmentation models. On Windows, install the pinned official MedSAM2 model and CUDA dependencies with Python 3.12:

```powershell
& ".\Abuya Code\setup_medsam2.ps1"
```

After loading a BraTS folder, MedSAM2 can be used immediately without running the U-Net. Select **Refine** on any 2D panel: left-click adds a foreground point, right-click adds a background point, and left-drag draws a box. **Preview** runs MedSAM2; **Accept** updates the measurements and 3D views and writes a native-space NIfTI mask under `Abuya Code/outputs/refined_masks`. The brain icon remains available when you want to run the separate 3D U-Net model.

## To fix

Bugs and polish still sitting in the current viewer.

- 🟠 **Segmentation overlay animation** — the run-time scan effect in the 2D viewers glitches, and it still uses the old teal. Retheme it to yellowish-orange and make the motion clean.
- ⬛ **Full-black 2D viewers** — sagittal, axial, and coronal should be black across the whole panel. Right now only part of each viewer is black.
- ⬇️ **Survival Days vs remarks** — after survival is calculated the digits grow and shove the UniKL remarks off the window. Keep the credit strip inside the software.
