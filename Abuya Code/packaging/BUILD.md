# Brain Tumor Viewer packaging (release)

PyInstaller **6.22.3**. **MedSAM2 is not bundled** (keeps the artifact under GitHub’s ~2 GB release limit). U-Net + `best_model.pth` still run on CPU/GPU.

| Artifact | How |
| --- | --- |
| `dist/BrainTumorViewer-macOS.dmg` | `bash packaging/build_mac.sh` |
| `dist/BrainTumorViewer/` | Windows: `powershell -File packaging/build_windows.ps1` |

Tag `v1.0.0` runs `.github/workflows/release.yml` for the Windows zip. Mac DMG is built on a Mac and uploaded to the Release.
