"""
Brain Tumor Viewer — layout matches the wireframe:

    [Enter image]        [ 2D sagittal ]        [   3D tumor    ]
    [Run segmentation]   [ 2D axial    ]        [               ]
    [Patients record]    [ 2D coronal  ]        [  3D brain view]

Wires the PySide6 shell to `brain_tumor_seg`: the trained 3D U-Net (with
survival head), the Plotly 3D viewer, and the BraTS-PEDs survival metadata.

Run from this folder or the project root:
    python ui_layout.py
"""
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Interface/ sits next to brain_tumor_seg/; running this file directly would
# otherwise miss the package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _prefer_pyside6_dlls() -> None:
    """
    On Windows, PyQt6 also ships Qt6Core.dll. If that copy is found first,
    `from PySide6.QtCore import ...` fails with:
        ImportError: DLL load failed while importing QtCore
    Put PySide6's own folder at the front of the DLL search path first.
    """
    if sys.platform != "win32":
        return
    try:
        import PySide6
    except ImportError:
        return
    dll_dir = str(Path(PySide6.__file__).resolve().parent)
    os.add_dll_directory(dll_dir)
    os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("QT_API", "pyside6")


_prefer_pyside6_dlls()

from PySide6.QtCore import Qt, QUrl, QTimer, QPointF, QObject, QThread, Signal, QSize
from PySide6.QtGui import (
    QColor, QFont, QIcon, QImage, QLinearGradient, QPainter, QPainterPath,
    QPen, QPixmap, QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QFileDialog, QSizePolicy, QSlider, QMessageBox,
    QProgressBar, QStackedLayout, QScrollArea,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage

import numpy as np
import nibabel as nib
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom, label as ndi_label, center_of_mass as ndi_center_of_mass
import plotly.graph_objects as go

from brain_tumor_seg.config import (
    CHECKPOINT_DIR, MEDSAM2_CHECKPOINT, MODALITIES, REFINED_MASK_DIR, TARGET_SHAPE,
)
from brain_tumor_seg.data.dataset import _normalize
from brain_tumor_seg.data.survival import load_survival_days, load_survival_stats
from brain_tumor_seg.evaluation import predict_survival_days
from brain_tumor_seg.models import MultiTaskUNet3D
from brain_tumor_seg.models.multitask import run_model, split_model_outputs
from brain_tumor_seg.sam import MedSAM2Refiner, VolumeTransform
from brain_tumor_seg.sam.transforms import prompt_to_rotated_xy, rotated_to_prompt_xy
from brain_tumor_seg.visualization.survival import format_survival
from brain_tumor_seg.visualization.viewer3d import INTERACTION_CONFIG, show_volume_3d, brain_surface_level

# ---------------------------------------------------------------------------
# Dark theme palette — charcoal base with yellowish-orange as the
# high-tech accent (UniKL KL orange, tumor highlight, survival neon).
# Teal stays only on the 2D scan overlays so the amber chrome can read.
# ---------------------------------------------------------------------------
BG_PAGE = "#121214"
BG_PANEL = "#1c1c20"
BORDER_DIM = "#2e2a24"
ACCENT_AMBER = "#ff9f1c"
ACCENT_AMBER_SOFT = "#ffc14d"
ACCENT_AMBER_DEEP = "#e67a00"
ACCENT_TEAL = "#0a84ff"
ACCENT_TEAL_SOFT = "#409cff"
ACCENT_CORAL = ACCENT_AMBER
ACCENT_CORAL_SOFT = ACCENT_AMBER_SOFT
TEXT_LIGHT = "#f4f1ea"
TEXT_MUTED = "#9a9388"
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_HEADER_BRAIN_PATH = _ASSETS_DIR / "header_brain.png"
_UNIKL_LOGO_PATH = _ASSETS_DIR / "unikl_logo.png"
_APP_ICON_PATH = _ASSETS_DIR / "app_icon.png"
_APP_ICO_PATH = _ASSETS_DIR / "app_icon.ico"
_RUN_ICON_PATH = _ASSETS_DIR / "icon_run.png"
SURVIVAL_EDGE_GAP = 12
_ICON_CACHE: dict = {}
_HIVE_RGB_CACHE: dict = {}

# Distinct colors assigned to separate tumor components (a brain can have
# more than one lesion) — largest lesion gets the first color, and the
# same color is used consistently across the 2D heatmap, the report
# panel, and the PDF export so "the blue one" means the same tumor
# everywhere. Cycles if there are more components than colors.
COMPONENT_PALETTE = [
    ("Red", "#ff3b30"),
    ("Blue", "#0a84ff"),
    ("Green", "#30d158"),
    ("Yellow", "#ffd60a"),
    ("Purple", "#bf5af2"),
    ("Cyan", "#64d2ff"),
    ("Orange", "#ff9f0a"),
    ("Pink", "#ff375f"),
]


def _hex_to_rgb01(hex_color: str) -> tuple:
    """'#ff3b30' -> (1.0, 0.23, 0.19), for blending into a matplotlib RGB array."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)

# Single spacing unit used everywhere: gap between columns, gap between
# panels within a column, gap between buttons, and the window's outer
# margin. One number instead of several near-matching ones keeps the
# rhythm consistent across the whole interface.
SPACING = 16

APP_STYLESHEET = f"""
QMainWindow, QWidget#rootShell {{
    background-color: {BG_PAGE};
}}
QLabel {{
    color: {TEXT_LIGHT};
}}
QMessageBox {{
    background-color: {BG_PANEL};
}}
QMessageBox QLabel {{
    color: {TEXT_LIGHT};
}}
QSlider {{
    background: transparent;
    min-height: 18px;
    max-height: 18px;
}}
QSlider::groove:horizontal {{
    height: 3px;
    background: {BORDER_DIM};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT_AMBER};
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}
QSlider::handle:horizontal:disabled {{
    background: #48484a;
}}
QProgressBar {{
    border: none;
    border-radius: 5px;
    background-color: {BG_PANEL};
    height: 10px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT_AMBER};
    border-radius: 5px;
}}
QScrollBar:vertical {{
    background: #16161a;
    width: 8px;
    margin: 2px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {ACCENT_AMBER};
    border-radius: 4px;
    min-height: 22px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
"""


def _panel_style(border_color: Optional[str] = None) -> str:
    """
    Flat Apple-style card: no border by default (contrast comes from the
    background color alone, same as a macOS card on a page). Passing an
    accent color adds a thin 1.5px outline for the 'active/has data' state
    — no drop shadow, no glow, just a quiet color change.
    """
    border = f"1.5px solid {border_color}" if border_color else "none"
    return f"QFrame {{ border: {border}; border-radius: 16px; background-color: {BG_PANEL}; }}"


MAXIMIZE_BTN_STYLE = f"""
QPushButton {{
    border: none;
    border-radius: 6px;
    background-color: transparent;
    color: {ACCENT_AMBER};
    font-size: 13px;
    padding: 1px 5px;
}}
QPushButton:hover {{
    color: {ACCENT_AMBER_SOFT};
    background-color: #3a3224;
}}
"""

DROP_ZONE_STYLE = f"""
QFrame {{
    border: 1.5px dashed {BORDER_DIM};
    border-radius: 16px;
    background-color: {BG_PANEL};
}}
"""

DROP_ZONE_HOVER_STYLE = f"""
QFrame {{
    border: 1.5px dashed {ACCENT_AMBER};
    border-radius: 16px;
    background-color: #2a2114;
}}
"""

DROP_ZONE_LOADED_STYLE = f"""
QFrame {{
    border: 1.5px solid {ACCENT_AMBER};
    border-radius: 16px;
    background-color: {BG_PANEL};
}}
"""

# ---------------------------------------------------------------------------
# Model configuration — uses the project checkpoint and the same volume size
# the U-Net was trained at (mask is then resized back to native resolution).
# ---------------------------------------------------------------------------
def _resolve_checkpoint() -> Path:
    candidates = (
        CHECKPOINT_DIR / "best_model.pth",
        CHECKPOINT_DIR / "best_model_segmentation_only.pth",
        Path(__file__).resolve().parent / "checkpoints" / "best_model.pth",
    )
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


MODEL_PATH = _resolve_checkpoint()
INFERENCE_SIZE = TARGET_SHAPE
# 3D marching cubes on a native 240^3 volume stalls the UI; downsample first.
RENDER_MAX_DIM = 96
# Fixed inference/display grid. Direct resizing is intentional because it
# matches the preprocessing used to train the U-Net.
STANDARD_DISPLAY_SHAPE = (128, 128, 128)  # Direct resize, matching training preprocessing.

# Embedded default for the folder picker (BraTS-PEDs training cases).
EMBEDDED_TRAINING_DIR = Path(
    r"C:\Users\Haqkiem\OneDrive\UNIKL\July-2026\Competition"
    r"\PKG - BraTS-PEDs-v1\BraTS-PEDs-v1\Training"
)

_MODEL_CACHE = {"model": None, "device": None}


def _default_browse_dir() -> str:
    if EMBEDDED_TRAINING_DIR.is_dir():
        return str(EMBEDDED_TRAINING_DIR)
    for directory in [_PROJECT_ROOT, *_PROJECT_ROOT.parents]:
        candidate = directory / "PKG - BraTS-PEDs-v1" / "BraTS-PEDs-v1" / "Training"
        if candidate.is_dir():
            return str(candidate)
    return str(Path.home())


def _resize_exact(volume: np.ndarray, target_shape: tuple, order: int = 1) -> np.ndarray:
    target = tuple(int(s) for s in target_shape)
    if volume.shape == target:
        return volume
    zoom_factors = [t / s for t, s in zip(target, volume.shape)]
    resampled = zoom(volume, zoom_factors, order=order)
    if resampled.shape != target:
        slices = tuple(slice(0, min(s, t)) for s, t in zip(resampled.shape, target))
        cropped = resampled[slices]
        pad_widths = [(0, t - c) for c, t in zip(cropped.shape, target)]
        resampled = np.pad(cropped, pad_widths, mode="constant")
    return resampled


def _standardize_volume(volume: np.ndarray, affine: np.ndarray, order: int = 1):
    """
    Resample to STANDARD_DISPLAY_SHAPE exactly as the training dataset does.

    The returned affine reflects the resized voxel spacing, keeping volume and
    3D measurements correct on the display grid.
    """
    native_spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    native_shape = np.array(volume.shape, dtype=np.float64)
    display_shape = np.array(STANDARD_DISPLAY_SHAPE, dtype=np.float64)
    resampled = _resize_exact(volume, STANDARD_DISPLAY_SHAPE, order=order)
    new_spacing = native_spacing * (native_shape / display_shape)
    new_affine = np.eye(4)
    new_affine[0, 0] = new_spacing[0]
    new_affine[1, 1] = new_spacing[1]
    new_affine[2, 2] = new_spacing[2]
    return resampled, new_affine


def _content_bbox(volume: np.ndarray, thresh: float = 0.04, margin: float = 0.05):
    """Inclusive (lo, hi) of tissue voxels, with a small margin. None if empty."""
    hits = np.argwhere(volume > thresh)
    if hits.size == 0:
        return None
    lo = hits.min(axis=0)
    hi = hits.max(axis=0)
    pad = np.maximum(2, ((hi - lo + 1) * margin).astype(int))
    lo = np.maximum(0, lo - pad)
    hi = np.minimum(np.array(volume.shape) - 1, hi + pad)
    return lo, hi


class _BackgroundJob(QObject):
    """Runs a callable off the GUI thread so loading overlays can keep painting."""

    ok = Signal(object)
    err = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            self.ok.emit(self._fn())
        except Exception as exc:
            self.err.emit(str(exc))


def _ensure_model():
    cached = _MODEL_CACHE["model"]
    if cached is not None:
        return cached, _MODEL_CACHE["device"]
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model = MultiTaskUNet3D(in_channels=4, out_channels=1)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_segmentation_weights(state_dict)
    model.to(device)
    model.eval()
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["device"] = device
    return model, device


def _column_separator() -> QFrame:
    line = QFrame()
    line.setFixedWidth(1)
    line.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
    line.setStyleSheet("QFrame { background-color: #4a3a22; border: none; }")
    return line


def _hex_points(cx: float, cy: float, radius: float) -> QPolygonF:
    pts = QPolygonF()
    for i in range(6):
        angle = math.radians(60 * i - 30)
        pts.append(QPointF(cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return pts


def _paint_hive_chrome(painter: QPainter, rect, color: QColor) -> None:
    """Amber honeycomb corners + edge ticks around a card."""
    painter.setRenderHint(QPainter.Antialiasing)
    inset = 9
    radius = 6.5
    corners = (
        (rect.left() + inset, rect.top() + inset),
        (rect.right() - inset, rect.top() + inset),
        (rect.left() + inset, rect.bottom() - inset),
        (rect.right() - inset, rect.bottom() - inset),
    )
    glow = QColor(color)
    glow.setAlpha(28)
    edge = QColor(color)
    edge.setAlpha(72)
    painter.setBrush(Qt.NoBrush)
    for cx, cy in corners:
        painter.setPen(QPen(glow, 2.4))
        painter.drawPolygon(_hex_points(cx, cy, radius + 1.5))
        painter.setPen(QPen(edge, 1.15))
        painter.drawPolygon(_hex_points(cx, cy, radius))
    painter.setPen(QPen(edge, 1.0))
    left, top, right, bottom = rect.left() + 6, rect.top() + 6, rect.right() - 6, rect.bottom() - 6
    painter.drawLine(left + 14, top, left + 30, top)
    painter.drawLine(right - 30, top, right - 14, top)
    painter.drawLine(left + 14, bottom, left + 30, bottom)
    painter.drawLine(right - 30, bottom, right - 14, bottom)
    painter.drawLine(left, top + 14, left, top + 28)
    painter.drawLine(right, top + 14, right, top + 28)
    painter.drawLine(left, bottom - 28, left, bottom - 14)
    painter.drawLine(right, bottom - 28, right, bottom - 14)


def _paint_sparse_hives(painter: QPainter, rect, seeds) -> None:
    """A few irregular hex marks — not a filled honeycomb grid."""
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(Qt.NoBrush)
    for cx, cy, radius, alpha, sides in seeds:
        x = rect.left() + cx * rect.width()
        y = rect.top() + cy * rect.height()
        color = QColor(ACCENT_AMBER)
        color.setAlpha(alpha)
        painter.setPen(QPen(color, 1.15))
        pts = _hex_points(x, y, radius)
        if sides >= 6:
            painter.drawPolygon(pts)
            continue
        for i in range(min(sides, 6)):
            painter.drawLine(pts[i], pts[(i + 1) % 6])


def _tight_pixmap(path: Path, pad: int = 8) -> QPixmap:
    pix = QPixmap(str(path))
    if pix.isNull():
        return pix
    image = pix.toImage()
    width, height = image.width(), image.height()
    min_x, min_y, max_x, max_y = width, height, -1, -1
    step = max(1, min(width, height) // 240)
    for y in range(0, height, step):
        for x in range(0, width, step):
            if image.pixelColor(x, y).lightness() > 14:
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y
    if max_x < 0:
        return pix
    return QPixmap.fromImage(image.copy(
        max(0, min_x - pad),
        max(0, min_y - pad),
        min(width - max(0, min_x - pad), max_x - min_x + 2 * pad),
        min(height - max(0, min_y - pad), max_y - min_y + 2 * pad),
    ))


def _asset_icon(path: Path) -> QIcon:
    key = str(path)
    icon = _ICON_CACHE.get(key)
    if icon is None:
        icon = QIcon(key)
        _ICON_CACHE[key] = icon
    return icon


def _paint_hive_field(painter: QPainter, rect, alpha: float = 0.20, radius: float = 13.0) -> None:
    """Even honeycomb HUD — same opacity across the whole rectangle."""
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(Qt.NoBrush)
    color = QColor(ACCENT_AMBER)
    color.setAlphaF(alpha)
    painter.setPen(QPen(color, 1.05))
    dx = radius * 1.75
    dy = radius * 1.52
    row = 0
    y = rect.top() + radius * 0.35
    while y < rect.bottom() + radius:
        x = rect.left() + radius * (0.85 if row % 2 else 0.1)
        while x < rect.right() + radius:
            painter.drawPolygon(_hex_points(x, y, radius))
            x += dx
        y += dy
        row += 1


def _hive_backdrop_rgb(height: int, width: int) -> np.ndarray:
    """Soft amber honeycomb field used behind 2D slices."""
    key = (height, width)
    cached = _HIVE_RGB_CACHE.get(key)
    if cached is not None:
        return cached
    image = QImage(max(1, width), max(1, height), QImage.Format_RGBA8888)
    image.fill(QColor("#08080c"))
    painter = QPainter(image)
    _paint_hive_field(painter, image.rect(), alpha=0.20, radius=max(9.0, min(height, width) / 12.0))
    painter.end()
    buf = bytes(image.constBits())
    rgba = np.frombuffer(buf, dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
    rgba = rgba[:, : width * 4].reshape(height, width, 4)
    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    _HIVE_RGB_CACHE[key] = rgb
    return rgb


def _fullscreen_button(tooltip: str = "Maximize") -> QPushButton:
    button = QPushButton("⛶")
    button.setFixedSize(22, 20)
    button.setStyleSheet(MAXIMIZE_BTN_STYLE)
    button.setToolTip(tooltip)
    button.setCursor(Qt.PointingHandCursor)
    button.setFocusPolicy(Qt.NoFocus)
    return button


class HiveField(QWidget):
    """Full-rectangle honeycomb behind a 2D viewer."""

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#08080c"))
        _paint_hive_field(painter, self.rect(), alpha=0.20, radius=13.0)
        painter.end()


class HiveOverlay(QWidget):
    """Corner hive drawn above child viewers so hexes are not clipped."""

    def __init__(self, host):
        super().__init__(host)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent; border: none;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        _paint_hive_chrome(painter, self.rect().adjusted(1, 1, -1, -1), QColor(ACCENT_AMBER))
        painter.end()


class HivePanel(QFrame):
    """Card with amber honeycomb corners layered above inner content."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hive_color = QColor(ACCENT_AMBER)
        self._hive_overlay = HiveOverlay(self)
        self._hive_overlay.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_hive_overlay()

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_hive_overlay()

    def _sync_hive_overlay(self):
        self._hive_overlay.setGeometry(self.rect())
        self._hive_overlay.raise_()
        self._hive_overlay.show()


class RootShell(QWidget):
    """Dark page shell. Object name keeps the global stylesheet off child widgets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rootShell")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(BG_PAGE))
        painter.end()


class AppHeader(QWidget):
    """Circuit / hive header with generated cancer-detection brain art."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(86)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._brain = QPixmap(str(_HEADER_BRAIN_PATH))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        rect = self.rect()

        wash = QLinearGradient(0, 0, rect.width(), 0)
        wash.setColorAt(0.0, QColor("#0c0a08"))
        wash.setColorAt(0.55, QColor("#16120e"))
        wash.setColorAt(1.0, QColor("#0a0a0c"))
        painter.fillRect(rect, wash)

        dim = QColor(ACCENT_AMBER)
        dim.setAlpha(55)
        lit = QColor(ACCENT_AMBER)
        lit.setAlpha(130)
        painter.setPen(QPen(dim, 1))
        painter.drawLine(18, 16, int(rect.width() * 0.48), 16)
        painter.drawLine(18, rect.height() - 14, int(rect.width() * 0.40), rect.height() - 14)
        painter.setPen(QPen(lit, 1.1))
        painter.drawLine(28, 16, 28, rect.height() - 14)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(lit, 1))
        painter.drawPolygon(_hex_points(28, 16, 4.5))
        painter.drawPolygon(_hex_points(28, rect.height() - 14, 4.5))

        brain_w = 0
        bx = rect.right()
        if not self._brain.isNull():
            brain = self._brain.scaled(
                min(280, int(rect.width() * 0.30)),
                rect.height() - 4,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            bx = rect.width() - brain.width() - 4
            by = (rect.height() - brain.height()) // 2
            painter.drawPixmap(bx, by, brain)
            brain_w = brain.width()

        _paint_sparse_hives(
            painter,
            rect,
            (
                (0.58, 0.28, 9, 70, 6),
                (0.64, 0.62, 7, 46, 4),
                (0.70, 0.22, 11, 38, 3),
                (0.74, 0.70, 8, 58, 5),
                (0.52, 0.72, 6, 40, 6),
                (0.80, 0.40, 5, 52, 4),
            ),
        )
        painter.setPen(QPen(QColor(255, 159, 28, 70), 1))
        painter.drawLine(int(rect.width() * 0.58), int(rect.height() * 0.28), bx + 10, rect.height() // 2)

        title_end = self._draw_tech_title(painter, rect.adjusted(44, 0, -brain_w - 16, 0))
        painter.setPen(QPen(QColor(255, 159, 28, 160), 1.3))
        painter.drawLine(48, rect.height() - 18, max(title_end + 8, 220), rect.height() - 18)

        painter.setPen(QPen(QColor(ACCENT_AMBER), 1.4))
        painter.drawLine(0, rect.height() - 1, rect.width(), rect.height() - 1)
        painter.end()

    def _draw_tech_title(self, painter: QPainter, rect) -> int:
        parts = (
            ("A.I. ", True),
            ("COPILOT FOR ", False),
            ("CANCER DETECTION", True),
            (" AND ", False),
            ("PATIENT SURVIVAL PREDICTION", True),
        )
        hot = QColor(ACCENT_AMBER)
        glow = QColor(255, 176, 50, 110)
        cream = QColor("#ffd27a")
        size = 13
        text_width = rect.width()
        while size >= 9:
            font = QFont("Segoe UI", size, QFont.DemiBold)
            font.setLetterSpacing(QFont.AbsoluteSpacing, 1.1)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            full = "".join(text for text, _ in parts)
            if metrics.horizontalAdvance(full) + 22 <= text_width:
                break
            size -= 1
        metrics = painter.fontMetrics()
        y = rect.center().y() + metrics.ascent() / 2 - 2
        x = rect.left()
        painter.setPen(QPen(QColor(ACCENT_AMBER), 1.2))
        painter.setBrush(QColor(255, 159, 28, 40))
        painter.drawPolygon(_hex_points(x + 6, y - metrics.ascent() * 0.35, 6))
        x += 18
        for text, accent in parts:
            color = hot if accent else cream
            painter.setPen(QColor(255, 159, 28, 70))
            painter.drawText(int(x + 1), int(y + 1), text)
            painter.setPen(color)
            painter.drawText(int(x), int(y), text)
            x += metrics.horizontalAdvance(text)
        return int(x)


class TechRule(QWidget):
    """Horizontal circuit separator used above the UniKL credit."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(14)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        y = self.height() / 2
        w = self.width()
        amber = QColor(ACCENT_AMBER)
        amber.setAlpha(160)
        dim = QColor(ACCENT_AMBER)
        dim.setAlpha(60)
        painter.setPen(QPen(dim, 1))
        painter.drawLine(8, y, w - 8, y)
        painter.setPen(QPen(amber, 1.2))
        painter.drawLine(28, y, w - 28, y)
        painter.setBrush(QColor(ACCENT_AMBER))
        for x in (10, w / 2, w - 10):
            painter.setPen(QPen(amber, 1))
            painter.drawPolygon(_hex_points(x, y, 4.2))
        painter.end()


class SquadCredit(QWidget):
    """Developed-by strip: circuit rule, UniKL logo, A.I. Squad remark."""

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 18, 2, 4)
        layout.setSpacing(8)
        layout.addWidget(TechRule())

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        logo = QLabel()
        logo.setStyleSheet("background: transparent; border: none;")
        pix = _tight_pixmap(_UNIKL_LOGO_PATH)
        if not pix.isNull():
            logo.setPixmap(pix.scaled(108, 44, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        row.addWidget(logo, 0, Qt.AlignVCenter)

        copy = QLabel("Developed by\nUniKL MIIT A.I. Squad")
        copy.setWordWrap(True)
        copy.setStyleSheet(
            f"border: none; background: transparent; color: {ACCENT_AMBER_SOFT}; "
            f"font-size: 9px; font-weight: 600; letter-spacing: 0.4px;"
        )
        row.addWidget(copy, 1)
        layout.addLayout(row)


class TechLabel(QLabel):
    """Yellowish-orange HUD label for the 3D panels."""

    def __init__(self, text: str):
        super().__init__(f"⬡  {text.upper()}")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            f"border: none; color: {ACCENT_AMBER}; font-size: 11px; "
            f"font-weight: 700; letter-spacing: 1.5px; background: transparent;"
        )


class _RecordRow(QWidget):
    """Bullet row: small muted label, larger value underneath."""

    def __init__(self, caption: str):
        super().__init__()
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        bullet = QLabel("•")
        bullet.setFixedWidth(12)
        bullet.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        bullet.setStyleSheet(
            f"border: none; color: {ACCENT_AMBER}; font-size: 16px; "
            f"font-weight: 700; background: transparent; padding-top: 1px;"
        )
        outer.addWidget(bullet)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)

        cap = QLabel(caption)
        cap.setStyleSheet(
            f"border: none; color: {TEXT_MUTED}; font-size: 10px; "
            f"font-weight: 500; letter-spacing: 0.3px; background: transparent;"
        )
        text_col.addWidget(cap)

        self.value = QLabel("—")
        self.value.setWordWrap(True)
        self.value.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._apply_value_style(filled=False)
        text_col.addWidget(self.value)
        outer.addLayout(text_col, stretch=1)

    def _apply_value_style(self, filled: bool) -> None:
        color = TEXT_LIGHT if filled else "#636366"
        self.value.setStyleSheet(
            f"border: none; color: {color}; font-size: 13px; font-weight: 600; "
            f"background: transparent; padding: 0;"
        )

    def set_value(self, text: str, filled: bool) -> None:
        self.value.setText(text)
        self.value.setToolTip(text if filled else "")
        self._apply_value_style(filled)


class PatientRecordPanel(HivePanel):
    """Compact card: distinct header bar + bulleted facts."""

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(
            f"QFrame {{ border: none; border-radius: 14px; background-color: {BG_PANEL}; }}"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(132)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setStyleSheet(
            f"QFrame {{ background-color: #3a3a3c; border: none; "
            f"border-top-left-radius: 14px; border-top-right-radius: 14px; "
            f"border-bottom: 2px solid {ACCENT_AMBER}; }}"
        )
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(14, 9, 14, 8)
        title = QLabel("PATIENT RECORD")
        title.setStyleSheet(
            f"border: none; color: {ACCENT_AMBER_SOFT}; font-size: 11px; "
            f"font-weight: 700; letter-spacing: 1.4px; background: transparent;"
        )
        header_layout.addWidget(title)
        layout.addWidget(header)

        body = QWidget()
        body.setStyleSheet("background: transparent;")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(14, 10, 10, 12)
        body_layout.setSpacing(8)

        self.case_field = _RecordRow("Patient ID")
        self.survival_field = _RecordRow("Overall survival")
        self.volume_field = _RecordRow("Tumor volume")
        body_layout.addWidget(self.case_field)
        body_layout.addWidget(self.survival_field)
        body_layout.addWidget(self.volume_field)
        body.setMinimumHeight(210)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setWidget(body)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        scroll.viewport().setStyleSheet("background: transparent;")
        layout.addWidget(scroll, 1)
        self._record_scroll = scroll

    def set_case(self, case_id: str, survival_days: Optional[float]) -> None:
        self.case_field.set_value(case_id, True)
        if survival_days is None:
            self.survival_field.set_value("No metadata label", False)
        else:
            self.survival_field.set_value(format_survival(survival_days), True)
        self.volume_field.set_value("—", False)

    def set_volume(self, tumor_cm3: Optional[float]) -> None:
        if tumor_cm3 is None:
            self.volume_field.set_value("—", False)
        else:
            self.volume_field.set_value(f"{tumor_cm3:.2f} cm\u00b3", True)


class DropVolumeZone(HivePanel):
    """Click or drag-and-drop a patient folder (or a NIfTI file inside one)."""

    folder_chosen = Signal(object)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(96)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._loaded = False
        self._set_style(hover=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 14, 12, 14)
        layout.setSpacing(6)

        self._title = QLabel("Drop volume")
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setWordWrap(True)
        self._title.setStyleSheet(
            f"border: none; color: {TEXT_LIGHT}; font-size: 13px; font-weight: 600;"
        )
        layout.addWidget(self._title)

        self._hint = QLabel("Drop a patient folder here\nor click to browse")
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(
            f"border: none; color: {TEXT_MUTED}; font-size: 11px;"
        )
        layout.addWidget(self._hint)

    def _set_style(self, hover: bool) -> None:
        if hover:
            self.setStyleSheet(DROP_ZONE_HOVER_STYLE)
        elif self._loaded:
            self.setStyleSheet(DROP_ZONE_LOADED_STYLE)
        else:
            self.setStyleSheet(DROP_ZONE_STYLE)

    def set_case(self, case_id: str) -> None:
        self._loaded = True
        self._title.setText(case_id)
        self._hint.setText("Drop or click to replace")
        self._set_style(hover=False)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            folder = QFileDialog.getExistingDirectory(
                self.window(), "Select patient folder", _default_browse_dir()
            )
            if folder:
                self.folder_chosen.emit(Path(folder))
        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        self._set_style(hover=True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._set_style(hover=False)
        super().leaveEvent(event)

    @staticmethod
    def _path_from_mime(event) -> Optional[Path]:
        urls = event.mimeData().urls() if event.mimeData() else []
        if not urls:
            return None
        path = Path(urls[0].toLocalFile())
        if path.is_file() and re.search(r"\.nii(\.gz)?$", path.name, re.IGNORECASE):
            return path.parent
        if path.is_dir():
            return path
        return None

    def dragEnterEvent(self, event):
        if self._path_from_mime(event) is not None:
            event.acceptProposedAction()
            self._set_style(hover=True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._set_style(hover=False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        folder = self._path_from_mime(event)
        self._set_style(hover=False)
        if folder is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.folder_chosen.emit(folder)


_NEON_SEGMENTS = {
    "0": "abcdef",
    "1": "bc",
    "2": "abged",
    "3": "abcdg",
    "4": "fgbc",
    "5": "afgcd",
    "6": "afgcde",
    "7": "abc",
    "8": "abcdefg",
    "9": "abcdfg",
}

_NEON_OFF = QColor(52, 28, 4, 80)
_NEON_SEG_SHADES = {
    "a": (QColor(180, 80, 0, 50), QColor(255, 140, 20, 90), QColor("#d97800"), QColor("#ffb020"), QColor("#ffe7b0")),
    "b": (QColor(160, 70, 0, 45), QColor(240, 130, 10, 85), QColor("#c86800"), QColor("#ff9f1c"), QColor("#ffd88a")),
    "c": (QColor(140, 60, 0, 40), QColor(220, 120, 8, 80), QColor("#b85c00"), QColor("#f59212"), QColor("#ffd070")),
    "d": (QColor(190, 90, 0, 50), QColor(255, 160, 30, 95), QColor("#e88800"), QColor("#ffc14d"), QColor("#fff4d2")),
    "e": (QColor(120, 50, 0, 40), QColor(200, 110, 0, 75), QColor("#a85000"), QColor("#e67a00"), QColor("#ffc868")),
    "f": (QColor(150, 65, 0, 45), QColor(230, 125, 10, 80), QColor("#c06000"), QColor("#ffaa28"), QColor("#ffe0a0")),
    "g": (QColor(200, 100, 0, 55), QColor(255, 180, 40, 100), QColor("#ff9f1c"), QColor("#ffd060"), QColor("#ffffff")),
}


class NeonDigits(QWidget):
    """Seven-segment readout with layered yellowish-orange neon glow."""

    def __init__(self, digits: str = "000"):
        super().__init__()
        self._digits = digits
        self.setFixedHeight(102)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent; border: none;")

    def set_value(self, days: Optional[int]) -> None:
        if days is None:
            self._digits = "000"
        else:
            n = max(0, int(days))
            self._digits = f"{n:03d}" if n < 1000 else str(n)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        count = max(3, len(self._digits))
        text = self._digits.rjust(count, "0")
        pad = 8
        gap = 8
        available = max(1, self.width() - pad * 2 - gap * (count - 1))
        digit_w = available / count
        digit_h = self.height() - 12
        y = 6

        for i, ch in enumerate(text):
            x = pad + i * (digit_w + gap)
            self._draw_digit(painter, x, y, digit_w, digit_h, ch)
        painter.end()

    def _draw_digit(self, painter: QPainter, x: float, y: float, w: float, h: float, ch: str):
        thickness = max(6.0, min(w, h) * 0.20)
        inset = 2.0
        active = set(_NEON_SEGMENTS.get(ch, ""))

        def horiz(px, py, length):
            path = QPainterPath()
            t = thickness / 2
            path.moveTo(px + t, py)
            path.lineTo(px + length - t, py)
            path.lineTo(px + length, py + t)
            path.lineTo(px + length - t, py + thickness)
            path.lineTo(px + t, py + thickness)
            path.lineTo(px, py + t)
            path.closeSubpath()
            return path

        def vert(px, py, length):
            path = QPainterPath()
            t = thickness / 2
            path.moveTo(px + t, py)
            path.lineTo(px + thickness, py + t)
            path.lineTo(px + thickness, py + length - t)
            path.lineTo(px + t, py + length)
            path.lineTo(px, py + length - t)
            path.lineTo(px, py + t)
            path.closeSubpath()
            return path

        mid_y = y + (h - thickness) / 2
        inner_w = w - thickness
        upper_h = mid_y - y - inset
        lower_h = (y + h) - (mid_y + thickness) - inset

        segs = {
            "a": horiz(x + thickness * 0.4, y, inner_w),
            "d": horiz(x + thickness * 0.4, y + h - thickness, inner_w),
            "g": horiz(x + thickness * 0.4, mid_y, inner_w),
            "f": vert(x, y + thickness * 0.5, upper_h),
            "b": vert(x + w - thickness, y + thickness * 0.5, upper_h),
            "e": vert(x, mid_y + thickness * 0.4, lower_h),
            "c": vert(x + w - thickness, mid_y + thickness * 0.4, lower_h),
        }

        for name, path in segs.items():
            if name not in active:
                painter.setPen(Qt.NoPen)
                painter.setBrush(_NEON_OFF)
                painter.drawPath(path)
                continue
            halo, glow, deep, mid, hot = _NEON_SEG_SHADES[name]
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(halo, thickness * 2.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPath(path)
            painter.setPen(QPen(glow, thickness * 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPath(path)
            painter.setPen(Qt.NoPen)
            painter.setBrush(deep)
            painter.drawPath(path)
            painter.setBrush(mid)
            painter.drawPath(path)
            painter.setBrush(hot)
            painter.drawPath(path)


class SurvivalDaysPanel(HivePanel):
    """Predicted overall survival as a neon seven-segment number."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet(
            "QFrame { border: none; border-radius: 16px; background-color: #050608; }"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, SURVIVAL_EDGE_GAP, 12, SURVIVAL_EDGE_GAP)
        layout.setSpacing(8)

        title = QLabel("Survival Days")
        title.setAlignment(Qt.AlignCenter)
        title.setFixedHeight(18)
        title.setStyleSheet(
            f"border: none; background: transparent; color: {ACCENT_AMBER_SOFT}; "
            f"font-size: 13px; font-weight: 600;"
        )
        layout.addWidget(title)

        self.digits = NeonDigits("000")
        layout.addWidget(self.digits)
        self.setFixedHeight(SURVIVAL_EDGE_GAP + 18 + 8 + 102 + SURVIVAL_EDGE_GAP)

    def set_days(self, days: Optional[float]) -> None:
        if days is None:
            self.digits.set_value(None)
        else:
            self.digits.set_value(int(round(float(days))))


class PixelateEffect(QWidget):
    """
    Renders a pixelated/blocky version of a snapshot of the actual slice
    image (not an abstract shape), with the block size animating between
    fine and coarse over time — reads as 'the system is actively
    processing this specific image' rather than a generic overlay.
    """

    def __init__(self, color: str = ACCENT_TEAL, parent=None):
        super().__init__(parent)
        self._tint = QColor(color)
        self._source: Optional[QPixmap] = None
        self._phase = 0.0
        self._interval_ms = 60
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    def set_source(self, pixmap: QPixmap):
        """Call this right before start() with a fresh grab() of the
        canvas — the effect pixelates whatever was last actually shown."""
        self._source = pixmap
        self.update()

    def start(self):
        self._timer.start(self._interval_ms)

    def stop(self):
        self._timer.stop()
        self._phase = 0.0

    def _advance(self):
        self._phase = (self._phase + 0.05) % (2 * math.pi)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)  # crisp blocks, no smoothing

        w, h = self.width(), self.height()
        if self._source is None or self._source.isNull() or w <= 0 or h <= 0:
            painter.end()
            return

        # Block size oscillates between fine and coarse pixelation —
        # downscale-then-upscale with no smoothing is the classic
        # pixelation trick.
        wave = (math.sin(self._phase) + 1) / 2  # 0..1
        block = max(2, int(4 + wave * 24))
        small_w = max(1, w // block)
        small_h = max(1, h // block)

        scaled_down = self._source.scaled(
            small_w, small_h, Qt.IgnoreAspectRatio, Qt.FastTransformation
        )
        pixelated = scaled_down.scaled(
            w, h, Qt.IgnoreAspectRatio, Qt.FastTransformation
        )

        painter.setOpacity(0.9)
        painter.drawPixmap(0, 0, pixelated)

        # Faint accent tint so the pixelated frame reads as "processing,"
        # not just a low-res copy of the image.
        tint = QColor(self._tint)
        tint.setAlphaF(0.12)
        painter.setOpacity(1.0)
        painter.fillRect(self.rect(), tint)

        painter.end()


class ScanLineSweep(QWidget):
    """
    A glowing horizontal line that sweeps top-to-bottom across the whole
    panel on a loop, with a short fading trail behind it — a literal
    'scanning the image' cue. Unlike the small fixed-size loaders before
    this one, it stretches to fill whatever space it's given.
    """

    def __init__(self, color: str = ACCENT_TEAL, parent=None):
        super().__init__(parent)
        self._base_color = QColor(color)
        self._progress = 0.0  # 0..1, position from top to bottom
        self._interval_ms = 20
        self._speed = 0.012  # progress added per tick
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    def start(self):
        self._timer.start(self._interval_ms)

    def stop(self):
        self._timer.stop()
        self._progress = 0.0

    def _advance(self):
        self._progress = (self._progress + self._speed) % 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        y = self._progress * h

        # Main line plus a few fainter copies trailing behind it, standing
        # in for a soft glow/motion-blur without needing a real blur effect.
        trail = [(0, 1.0, 3.0), (9, 0.5, 2.2), (18, 0.25, 1.6), (28, 0.1, 1.0)]
        for offset, opacity, pen_width in trail:
            trail_y = y - offset
            if trail_y < 0:
                continue
            color = QColor(self._base_color)
            color.setAlphaF(opacity)
            painter.setPen(QPen(color, pen_width))
            painter.drawLine(QPointF(0, trail_y), QPointF(w, trail_y))

        painter.end()


class ArcSpinner(QWidget):
    """
    Small segmented rotating arc — the secondary 'processing' cue paired
    with the scan line sweep above, meant to sit tucked in a corner
    rather than as the main focal point.
    """

    def __init__(self, diameter: int = 26, color: str = ACCENT_TEAL, parent=None):
        super().__init__(parent)
        self._base_color = QColor(color)
        self._angle = 0.0
        self.setFixedSize(diameter, diameter)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        self._segments = [
            (0, 60, 1.00),
            (75, 40, 0.60),
            (130, 25, 0.30),
        ]

    def start(self):
        self._timer.start(16)

    def stop(self):
        self._timer.stop()

    def _advance(self):
        self._angle = (self._angle + 8) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(3, 3, -3, -3)
        pen_width = 2.5

        for offset, span, opacity in self._segments:
            color = QColor(self._base_color)
            color.setAlphaF(opacity)
            painter.setPen(QPen(color, pen_width, Qt.SolidLine, Qt.RoundCap))
            start_angle = int((self._angle + offset) * 16)
            span_angle = int(span * 16)
            painter.drawArc(rect, start_angle, span_angle)

        painter.end()


class SlicePanel(HivePanel):
    """
    A bordered panel that shows one anatomical plane of a 3D volume as a
    matplotlib image, with a slider underneath to scroll through slices
    along that axis, and an optional tumor mask overlaid in red.

    `axis` is which array axis is sliced to produce this 2D view:
        0 -> sagittal, 1 -> coronal, 2 -> axial
    """

    refinement_requested = Signal(object)
    accept_requested = Signal()
    discard_requested = Signal()
    mask_edited = Signal(object)
    mask_undo_requested = Signal()

    def __init__(self, title: str, axis: int, min_height: int = 200):
        super().__init__()
        self.axis = axis
        self.volume = None      # grayscale background, 3D numpy array, 0-1 range
        self.mask = None        # optional binary mask, same shape as volume
        self.probs = None       # optional continuous 0-1 probability map, same shape
        self.labels = None      # optional int array, same shape: 0=background, 1..N=component id
        self.color_map = None   # optional {component_id: hex_color}, matching self.labels
        self._bbox = None       # inclusive (lo, hi) of brain tissue, for zoomed square crop
        self._sam_enabled = False
        self._sam_busy = False
        self._sam_preview_available = False
        self._sam_editing = False
        self._sam_points = []   # [(x, y, label)] in the unrotated slice
        self._sam_box = None    # (x0, y0, x1, y1) in the unrotated slice
        self._drag_start = None
        self._sam_prompt_slice = None
        self._resize_redraw_pending = False
        self._mask_edit_enabled = False
        self._mask_edit_mode = None  # "eraser" or None
        self._mask_stroke = []
        self._mask_emitted_count = 0
        self._mask_dragging = False

        self.setFrameShape(QFrame.Box)
        # Pure black background (not the charcoal BG_PANEL used elsewhere)
        # — MRI slices have their own black background outside the brain,
        # so matching that exactly makes the panel disappear into the
        # image instead of showing as a separate charcoal-colored frame
        # around it.
        self.setStyleSheet(
            "QFrame { border: 1.5px solid #1a140c; border-radius: 16px; "
            "background-color: #08080c; }"
        )
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        header.addStretch(1)
        self._base_title = title.upper()
        self.title_label = QLabel(self._base_title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(
            f"border: none; color: {ACCENT_AMBER}; font-size: 11px; "
            f"font-weight: 700; letter-spacing: 1.4px;"
        )
        header.addWidget(self.title_label)
        header.addStretch(1)
        self.maximize_btn = _fullscreen_button("Maximize")
        header.addWidget(self.maximize_btn)
        layout.addLayout(header)

        self.figure = Figure(figsize=(3, 3))
        self.figure.patch.set_facecolor("#000000")
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setStyleSheet("background-color: #000000;")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#000000")
        self.ax.axis("off")
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)

        # Canvas + loading overlay share the same space via a stacked
        # layout, so the spinner appears directly on top of the slice
        # image instead of needing a separate panel or popup.
        self.canvas_stack = QStackedLayout()
        self.canvas_stack.setStackingMode(QStackedLayout.StackAll)
        canvas_container = QWidget()
        canvas_container.setLayout(self.canvas_stack)
        self.canvas_stack.addWidget(self.canvas)

        self.loading_overlay = QWidget()
        # Lighter than before — the pixelated image itself now carries most
        # of the visual weight, so a heavy flat black backdrop would just
        # compete with it instead of framing it.
        self.loading_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 90);")
        overlay_stack = QStackedLayout(self.loading_overlay)
        overlay_stack.setStackingMode(QStackedLayout.StackAll)

        # Base layer: a pixelated snapshot of whatever was last actually
        # shown in this panel, with block size animating — "the system is
        # processing this specific image," not just a generic dimmed panel.
        self.pixelate_effect = PixelateEffect(color=ACCENT_TEAL)
        overlay_stack.addWidget(self.pixelate_effect)

        # Middle effect: the scan line sweeps top-to-bottom across the
        # (now pixelated) slice image.
        self.scan_line = ScanLineSweep(color=ACCENT_TEAL)
        overlay_stack.addWidget(self.scan_line)

        # Top effect: small arc spinner tucked in the top-right corner,
        # with the caption anchored at the bottom — this layer's own
        # background stays transparent so the layers underneath stay
        # visible everywhere except where these small widgets sit.
        corner_layer = QWidget()
        corner_layer.setStyleSheet("background: transparent;")
        corner_layout = QVBoxLayout(corner_layer)
        corner_layout.setContentsMargins(10, 10, 10, 10)

        spinner_row = QHBoxLayout()
        spinner_row.addStretch(1)
        self.loading_animation = ArcSpinner(diameter=26, color=ACCENT_TEAL_SOFT)
        spinner_row.addWidget(self.loading_animation)
        corner_layout.addLayout(spinner_row)
        corner_layout.addStretch(1)

        self.loading_caption = QLabel("SCANNING")
        self.loading_caption.setAlignment(Qt.AlignCenter)
        self.loading_caption.setStyleSheet(
            f"color: {ACCENT_TEAL_SOFT}; font-size: 10px; font-weight: 600; "
            f"letter-spacing: 2px; background: transparent; border: none;"
        )
        corner_layout.addWidget(self.loading_caption)

        overlay_stack.addWidget(corner_layer)

        self.canvas_stack.addWidget(self.loading_overlay)
        self.loading_overlay.hide()

        layout.addWidget(canvas_container, stretch=1)

        slider_wrap = QWidget()
        slider_wrap.setFixedHeight(28)
        slider_wrap.setStyleSheet("background: transparent;")
        slider_layout = QVBoxLayout(slider_wrap)
        slider_layout.setContentsMargins(4, 6, 4, 2)
        slider_layout.setSpacing(0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.slider)
        layout.addWidget(slider_wrap)

        # Manual mask correction for removing false-positive tumor voxels.
        # An eraser stroke is committed on mouse-up so
        # all three planes and both 3D views stay synchronized.
        self.mask_tools = QWidget()
        self.mask_tools.setStyleSheet("background: transparent; border: none;")
        mask_tools_layout = QHBoxLayout(self.mask_tools)
        mask_tools_layout.setContentsMargins(2, 0, 2, 0)
        mask_tools_layout.setSpacing(5)
        mask_label = QLabel("DRAG IMAGE TO ERASE")
        mask_label.setStyleSheet(
            f"color: {TEXT_MUTED}; border: none; font-size: 8px; font-weight: 700;"
        )
        self.eraser_size = QSlider(Qt.Horizontal)
        self.eraser_size.setRange(1, 20)
        self.eraser_size.setValue(5)
        self.eraser_size.setFixedWidth(65)
        self.eraser_size.setToolTip("Eraser radius in voxels")
        self.eraser_undo_btn = QPushButton("UNDO")
        self.eraser_undo_btn.setEnabled(False)
        self.eraser_undo_btn.setCursor(Qt.PointingHandCursor)
        self.eraser_undo_btn.setStyleSheet(
            f"QPushButton {{ color: {TEXT_MUTED}; background: #121216; "
            f"border: 1px solid {BORDER_DIM}; border-radius: 6px; "
            "padding: 3px 7px; font-size: 8px; font-weight: 700; }"
            f"QPushButton:hover {{ color: {TEXT_LIGHT}; border-color: {ACCENT_AMBER}; }}"
            "QPushButton:disabled { color: #4c494d; border-color: #29272a; }"
        )
        mask_tools_layout.addWidget(mask_label)
        mask_tools_layout.addStretch(1)
        mask_tools_layout.addWidget(QLabel("SIZE"))
        mask_tools_layout.addWidget(self.eraser_size)
        mask_tools_layout.addWidget(self.eraser_undo_btn)
        layout.addWidget(self.mask_tools)
        self.eraser_undo_btn.clicked.connect(self.mask_undo_requested.emit)

        # MedSAM2 uses progressive disclosure: the normal scan view gets one
        # clean action row, while prompt and review tools appear only when they
        # are relevant. This avoids the cramped six-button strip used by the
        # first integration and follows the amber card language of the app.
        self.sam_controls = QWidget()
        self.sam_controls.setObjectName("samControls")
        self.sam_controls.setStyleSheet(
            f"QWidget#samControls {{ background: #121216; border: 1px solid {BORDER_DIM}; "
            "border-radius: 9px; }}"
            "QWidget#samControls QLabel { border: none; background: transparent; }"
        )
        sam_layout = QVBoxLayout(self.sam_controls)
        sam_layout.setContentsMargins(8, 6, 8, 7)
        sam_layout.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.setSpacing(6)
        sam_title = QLabel("MEDSAM2")
        sam_title.setStyleSheet(
            f"color: {ACCENT_AMBER}; font-size: 9px; font-weight: 700; letter-spacing: 1.4px;"
        )
        self.sam_status = QLabel("LOCKED")
        self.sam_status.setAlignment(Qt.AlignCenter)
        self.sam_status.setStyleSheet(
            f"color: {TEXT_MUTED}; background: #202024; border: none; border-radius: 7px; "
            "padding: 2px 6px; font-size: 8px; font-weight: 700;"
        )
        header_row.addWidget(sam_title)
        header_row.addWidget(self.sam_status)
        header_row.addStretch(1)

        secondary_style = (
            f"QPushButton {{ color: {TEXT_MUTED}; background: transparent; border: 1px solid {BORDER_DIM}; "
            "border-radius: 6px; padding: 4px 8px; font-size: 9px; font-weight: 600; }}"
            f"QPushButton:hover {{ color: {TEXT_LIGHT}; border-color: {ACCENT_AMBER_DEEP}; "
            "background: #2a2114; }}"
            "QPushButton:disabled { color: #565159; border-color: #29272a; background: transparent; }"
        )
        primary_style = (
            f"QPushButton {{ color: #17120a; background: {ACCENT_AMBER}; border: none; "
            "border-radius: 6px; padding: 5px 10px; font-size: 9px; font-weight: 800; }}"
            f"QPushButton:hover {{ background: {ACCENT_AMBER_SOFT}; }}"
            "QPushButton:disabled { color: #77716a; background: #343036; }"
        )
        self.sam_refine_btn = QPushButton("REFINE")
        self.sam_refine_btn.setCheckable(True)
        self.sam_refine_btn.setEnabled(False)
        self.sam_refine_btn.setCursor(Qt.PointingHandCursor)
        self.sam_refine_btn.setStyleSheet(
            secondary_style
            + f"QPushButton:checked {{ color: {ACCENT_AMBER_SOFT}; border-color: {ACCENT_AMBER}; "
            "background: #2a2114; }}"
        )
        header_row.addWidget(self.sam_refine_btn)
        sam_layout.addLayout(header_row)

        self.sam_editor = QWidget()
        self.sam_editor.setStyleSheet("background: transparent; border: none;")
        editor_layout = QVBoxLayout(self.sam_editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(5)
        self.sam_prompt_summary = QLabel("FG LEFT  /  BG RIGHT  /  DRAG BOX")
        self.sam_prompt_summary.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 8px; letter-spacing: .4px;"
        )
        self.sam_prompt_summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.sam_prompt_summary.setWordWrap(True)
        editor_layout.addWidget(self.sam_prompt_summary)

        tool_row = QHBoxLayout()
        tool_row.setSpacing(5)
        self.sam_undo_btn = QPushButton("Undo")
        self.sam_clear_btn = QPushButton("Clear")
        self.sam_run_btn = QPushButton("PREVIEW")
        for button in (self.sam_undo_btn, self.sam_clear_btn):
            button.setStyleSheet(secondary_style)
            tool_row.addWidget(button)
        tool_row.addStretch(1)
        self.sam_run_btn.setStyleSheet(primary_style)
        self.sam_run_btn.setCursor(Qt.PointingHandCursor)
        tool_row.addWidget(self.sam_run_btn)
        editor_layout.addLayout(tool_row)
        sam_layout.addWidget(self.sam_editor)

        self.sam_review = QWidget()
        self.sam_review.setStyleSheet("background: transparent; border: none;")
        review_row = QHBoxLayout(self.sam_review)
        review_row.setContentsMargins(0, 0, 0, 0)
        review_row.setSpacing(5)
        review_label = QLabel("PREVIEW READY")
        review_label.setStyleSheet(
            f"color: {ACCENT_TEAL_SOFT}; font-size: 8px; font-weight: 700; letter-spacing: 1px;"
        )
        self.sam_discard_btn = QPushButton("DISCARD")
        self.sam_accept_btn = QPushButton("ACCEPT")
        self.sam_discard_btn.setStyleSheet(secondary_style)
        self.sam_accept_btn.setStyleSheet(primary_style)
        review_row.addWidget(review_label)
        review_row.addStretch(1)
        review_row.addWidget(self.sam_discard_btn)
        review_row.addWidget(self.sam_accept_btn)
        sam_layout.addWidget(self.sam_review)

        self.sam_refine_btn.toggled.connect(self._set_sam_editing)
        self.sam_undo_btn.clicked.connect(self._undo_sam_prompt)
        self.sam_clear_btn.clicked.connect(self.clear_sam_prompts)
        self.sam_run_btn.clicked.connect(self._request_sam_refinement)
        self.sam_accept_btn.clicked.connect(self.accept_requested.emit)
        self.sam_discard_btn.clicked.connect(self.discard_requested.emit)
        self.sam_editor.hide()
        self.sam_review.hide()
        self.set_sam_preview_available(False)
        layout.addWidget(self.sam_controls)
        self._show_hive_empty()
        self._sync_hive_overlay()

    def set_loading(self, active: bool):
        """Shows/hides the pixelate + scan-line + corner-arc loading
        animation over the slice image — used while Run segmentation is
        processing this panel's data."""
        if active:
            # Grab whatever's currently on screen (the last-shown slice)
            # right now, before any of it changes — that's what gets
            # pixelated for the duration of this loading pass.
            self.pixelate_effect.set_source(self.canvas.grab())
            self.loading_overlay.show()
            self.loading_overlay.raise_()
            self.pixelate_effect.start()
            self.scan_line.start()
            self.loading_animation.start()
        else:
            self.pixelate_effect.stop()
            self.scan_line.stop()
            self.loading_animation.stop()
            self.loading_overlay.hide()
        self._sync_hive_overlay()

    def set_volume(self, volume: np.ndarray):
        """volume: 3D numpy array (already normalized for display, 0-1 range)."""
        self.volume = volume
        self.mask = None   # new volume invalidates any previous overlay
        self.probs = None
        self.labels = None
        self.color_map = None
        self._bbox = _content_bbox(volume)
        self.clear_sam_prompts()
        self.enable_sam(False)
        self.enable_mask_editing(False)
        n_slices = volume.shape[self.axis]
        self.slider.setEnabled(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(n_slices - 1, 0))
        self.slider.setValue(n_slices // 2)
        self._draw_slice(n_slices // 2)

    def set_mask(self, mask: np.ndarray):
        """
        mask: binary 3D numpy array, same shape as the current volume.
        Renders as a hard on/off red overlay. Superseded by
        set_detection() when a probability map + per-component labels
        are available (see on_run_segmentation) — kept as a fallback API
        for callers that only have a plain binary mask.
        """
        if self.volume is not None and mask.shape != self.volume.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match volume shape {self.volume.shape}"
            )
        self.mask = mask
        self.probs = None
        self.labels = None
        self.color_map = None
        self.enable_mask_editing(True)
        self._draw_slice(self.slider.value())

    def clear_detection(self):
        """Remove segmentation overlays while keeping the loaded MRI volume."""
        self.mask = None
        self.probs = None
        self.labels = None
        self.color_map = None
        self.enable_mask_editing(False)
        if self.volume is not None:
            self._draw_slice(self.slider.value())

    def set_refined_detection(self, mask: np.ndarray, labels: np.ndarray, color_map: dict):
        """Show an accepted/previewed SAM mask with solid component colors."""
        if self.volume is not None and mask.shape != self.volume.shape:
            raise ValueError(f"mask shape {mask.shape} does not match volume shape {self.volume.shape}")
        self.mask = mask
        self.probs = None
        self.labels = labels
        self.color_map = color_map
        self.enable_mask_editing(True)
        self._draw_slice(self.slider.value())

    @property
    def plane(self) -> str:
        return {0: "sagittal", 1: "coronal", 2: "axial"}[self.axis]

    def enable_sam(self, enabled: bool):
        self._sam_enabled = bool(enabled)
        if not enabled:
            self.sam_refine_btn.setChecked(False)
        self._sync_sam_controls()

    def set_sam_busy(self, busy: bool):
        self._sam_busy = bool(busy)
        self._sync_sam_controls()

    def set_sam_preview_available(self, available: bool):
        self._sam_preview_available = bool(available)
        self._sync_sam_controls()

    def _sync_sam_controls(self):
        ready = self._sam_enabled and not self._sam_busy
        has_prompt = self._sam_box is not None or bool(self._sam_points)
        self.sam_refine_btn.setEnabled(ready)
        self.sam_undo_btn.setEnabled(ready and has_prompt)
        self.sam_clear_btn.setEnabled(ready and has_prompt)
        self.sam_run_btn.setEnabled(ready and has_prompt)
        self.sam_accept_btn.setEnabled(ready and self._sam_preview_available)
        self.sam_discard_btn.setEnabled(ready and self._sam_preview_available)
        self.sam_review.setVisible(self._sam_preview_available)

        if self._sam_busy:
            status, color = "PROCESSING", ACCENT_TEAL_SOFT
        elif self._sam_preview_available:
            status, color = "REVIEW", ACCENT_TEAL_SOFT
        elif self._sam_enabled:
            status, color = "READY", ACCENT_AMBER_SOFT
        else:
            status, color = "LOAD MRI", TEXT_MUTED
        self.sam_status.setText(status)
        self.sam_status.setStyleSheet(
            f"color: {color}; background: #202024; border: none; border-radius: 7px; "
            "padding: 2px 6px; font-size: 8px; font-weight: 700;"
        )

        count = len(self._sam_points) + int(self._sam_box is not None)
        self.sam_prompt_summary.setText(
            f"{count} PROMPT{'S' if count != 1 else ''}  ·  FG LEFT / BG RIGHT / DRAG BOX"
            if count else "FG LEFT  /  BG RIGHT  /  DRAG BOX"
        )

    def _set_sam_editing(self, editing: bool):
        if editing:
            self._set_mask_edit_mode(None)
        self._sam_editing = editing and self._sam_enabled
        if not self._sam_editing and self._mask_edit_enabled:
            self._set_mask_edit_mode("eraser")
        self.canvas.setCursor(
            Qt.CrossCursor if self._sam_editing or self._mask_edit_mode else Qt.ArrowCursor
        )
        self.sam_editor.setVisible(self._sam_editing)
        self.sam_refine_btn.setText("EDITING" if self._sam_editing else "REFINE")
        self._sync_sam_controls()

    def clear_sam_prompts(self):
        self._sam_points = []
        self._sam_box = None
        self._drag_start = None
        self._sam_prompt_slice = None
        self._sync_sam_controls()
        if self.volume is not None:
            self._draw_slice(self.slider.value())

    def _undo_sam_prompt(self):
        if self._sam_points:
            self._sam_points.pop()
        elif self._sam_box is not None:
            self._sam_box = None
        self._sync_sam_controls()
        self._draw_slice(self.slider.value())

    def _event_to_prompt_xy(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return None
        current = self._slice_along_axis(self.volume, self.slider.value())
        rows, cols = current.shape
        return rotated_to_prompt_xy(event.xdata, event.ydata, rows, cols)

    def enable_mask_editing(self, enabled: bool):
        self._mask_edit_enabled = bool(enabled)
        self.eraser_size.setEnabled(enabled)
        if not enabled:
            self.eraser_undo_btn.setEnabled(False)
        self._set_mask_edit_mode("eraser" if enabled and not self._sam_editing else None)

    def set_mask_undo_available(self, available: bool):
        # Undo remains usable while a previous 3D refresh is running; a newer
        # mask refresh will supersede the stale result when it completes.
        self.eraser_undo_btn.setEnabled(available)

    def _set_mask_edit_mode(self, mode):
        self._mask_edit_mode = mode
        if mode is not None and self.sam_refine_btn.isChecked():
            self.sam_refine_btn.setChecked(False)
        self.canvas.setCursor(Qt.CrossCursor if mode or self._sam_editing else Qt.ArrowCursor)

    def _append_mask_stroke_point(self, event):
        xy = self._event_to_prompt_xy(event)
        if xy is None:
            return
        point = (int(round(xy[0])), int(round(xy[1])))
        if not self._mask_stroke or point != self._mask_stroke[-1]:
            self._mask_stroke.append(point)

    def _emit_mask_stroke(self, final: bool = False, start_of_stroke: bool = False):
        """Send only newly sampled points; `final` requests the costly full refresh."""
        # Include the previously emitted endpoint so the receiver can fill
        # the line to the first new sample even when mouse events are sparse.
        start = max(0, self._mask_emitted_count - 1)
        new_points = self._mask_stroke[start:]
        if new_points or final:
            self.mask_edited.emit({
                "plane": self.plane,
                "slice_index": self.slider.value(),
                "points": list(new_points),
                "radius": self.eraser_size.value(),
                "final": final,
                "start": start_of_stroke,
            })
            self._mask_emitted_count = len(self._mask_stroke)

    def _on_canvas_press(self, event):
        if self._mask_edit_mode is not None and self.volume is not None and event.button == 1:
            self._mask_stroke = []
            self._mask_emitted_count = 0
            self._mask_dragging = True
            self._append_mask_stroke_point(event)
            self._emit_mask_stroke(start_of_stroke=True)
            return
        if not self._sam_editing or self.volume is None:
            return
        xy = self._event_to_prompt_xy(event)
        if xy is None:
            return
        current_slice = self.slider.value()
        if self._sam_prompt_slice is not None and self._sam_prompt_slice != current_slice:
            self.clear_sam_prompts()
        self._sam_prompt_slice = current_slice
        if event.button == 1:
            self._drag_start = xy
        elif event.button == 3:
            self._sam_points.append((xy[0], xy[1], 0))
            self._sync_sam_controls()
            self._draw_slice(self.slider.value())

    def _on_canvas_release(self, event):
        if self._mask_dragging and event.button == 1:
            self._append_mask_stroke_point(event)
            self._mask_dragging = False
            self._emit_mask_stroke(final=True)
            self._mask_stroke = []
            self._mask_emitted_count = 0
            return
        if not self._sam_editing or event.button != 1 or self._drag_start is None:
            return
        end = self._event_to_prompt_xy(event)
        start = self._drag_start
        self._drag_start = None
        if end is None:
            return
        # A small amount of movement is normal during a click. Requiring a
        # larger drag prevents an intended foreground point becoming a tiny,
        # misleading box prompt.
        if abs(end[0] - start[0]) >= 5 or abs(end[1] - start[1]) >= 5:
            self._sam_box = (
                min(start[0], end[0]), min(start[1], end[1]),
                max(start[0], end[0]), max(start[1], end[1]),
            )
        else:
            self._sam_points.append((end[0], end[1], 1))
        self._sync_sam_controls()
        self._draw_slice(self.slider.value())

    def _on_canvas_motion(self, event):
        if self._mask_dragging:
            self._append_mask_stroke_point(event)
            self._emit_mask_stroke()

    def _request_sam_refinement(self):
        if self._sam_box is None and not self._sam_points:
            return
        if self._sam_prompt_slice != self.slider.value():
            self.clear_sam_prompts()
            return
        self.refinement_requested.emit({
            "plane": self.plane,
            "slice_index": self.slider.value(),
            "box": self._sam_box,
            "points": [(x, y) for x, y, _label in self._sam_points],
            "point_labels": [label for _x, _y, label in self._sam_points],
        })

    def set_detection(self, probs: np.ndarray, labels: np.ndarray, color_map: dict):
        """
        probs: continuous 0-1 probability array (the model's raw sigmoid
        output, before thresholding). labels: same-shape int array from
        connected-component labeling (0=background, 1..N=component id —
        a brain can have more than one separate lesion). color_map:
        {component_id: hex_color}, assigning each distinct tumor its own
        color, matching the same colors used in the report/PDF.

        Blend strength still scales with confidence per voxel (see
        _draw_slice), but the color itself now depends on which
        component that voxel belongs to, instead of every tumor
        rendering in the same flat red.
        """
        if self.volume is not None and probs.shape != self.volume.shape:
            raise ValueError(
                f"probability map shape {probs.shape} does not match volume shape {self.volume.shape}"
            )
        self.probs = probs
        self.labels = labels
        self.color_map = color_map
        self.mask = None  # heatmap supersedes the binary overlay for display
        self.enable_mask_editing(True)
        self._draw_slice(self.slider.value())

    def _on_slider_changed(self, index: int):
        # Points and boxes are 2D prompts and are valid only on the slice on
        # which they were drawn. Never silently reuse them on another slice.
        if self._sam_prompt_slice is not None and self._sam_prompt_slice != index:
            self.clear_sam_prompts()
        self._draw_slice(index)

    def _slice_along_axis(self, array: np.ndarray, index: int) -> np.ndarray:
        if self.axis == 0:
            return array[index, :, :]
        elif self.axis == 1:
            return array[:, index, :]
        else:
            return array[:, :, index]

    def _draw_slice(self, index: int):
        if self.volume is None:
            return
        img_slice = self._slice_along_axis(self.volume, index)
        rgb = np.stack([img_slice, img_slice, img_slice], axis=-1)

        if self.probs is not None:
            # Continuous heatmap: blend strength scales with the model's
            # actual confidence at each voxel, rather than a flat on/off
            # color. Three things make the gradient actually visible
            # instead of a flat wash:
            #   1. A higher floor (0.2) drops low-confidence background
            #      noise entirely, instead of tinting the whole image
            #      faintly and diluting the contrast that matters.
            #   2. A gamma < 1 stretches the remaining 0.2-1.0 range so
            #      mid-confidence differences are visually distinguishable,
            #      not compressed into a narrow band near full opacity.
            #   3. A higher max alpha (0.95) lets fully-confident voxels
            #      read as strong, clearly-visible color.
            prob_slice = self._slice_along_axis(self.probs, index)
            confidence = np.clip(prob_slice, 0.0, 1.0)

            floor = 0.2
            gamma = 0.55
            max_alpha = 0.95

            normalized = np.zeros_like(confidence)
            visible = confidence > floor
            normalized[visible] = (confidence[visible] - floor) / (1.0 - floor)
            alpha_all = np.power(normalized, gamma) * max_alpha

            if self.labels is not None and self.color_map:
                # Color each connected component with its own assigned
                # color, so separate tumors are visually distinguishable
                # and match the same colors used in the report.
                label_slice = self._slice_along_axis(self.labels, index)
                for label_id, hex_color in self.color_map.items():
                    comp_rgb = _hex_to_rgb01(hex_color)
                    comp_region = label_slice == label_id
                    if not comp_region.any():
                        continue
                    a = alpha_all * comp_region
                    for c in range(3):
                        rgb[..., c] = np.where(
                            comp_region, (1 - a) * rgb[..., c] + a * comp_rgb[c], rgb[..., c]
                        )
                # Any visible-confidence voxels that ended up outside every
                # labeled component (shouldn't normally happen, since
                # labels come from the same mask) still show up in the
                # default red rather than silently vanishing.
                labeled_anywhere = label_slice > 0
                leftover = visible & (~labeled_anywhere)
                if leftover.any():
                    a = alpha_all * leftover
                    default_rgb = (1.0, 0.15, 0.15)
                    for c in range(3):
                        rgb[..., c] = np.where(
                            leftover, (1 - a) * rgb[..., c] + a * default_rgb[c], rgb[..., c]
                        )
            else:
                # No per-component labels available — fall back to the
                # single flat red heatmap.
                red = (1.0, 0.15, 0.15)
                for c in range(3):
                    rgb[..., c] = (1 - alpha_all) * rgb[..., c] + alpha_all * red[c]
        elif self.mask is not None:
            mask_slice = self._slice_along_axis(self.mask, index)
            hit = mask_slice > 0.5
            if self.labels is not None and self.color_map:
                label_slice = self._slice_along_axis(self.labels, index)
                for label_id, hex_color in self.color_map.items():
                    region = label_slice == label_id
                    color = _hex_to_rgb01(hex_color)
                    for channel in range(3):
                        rgb[..., channel] = np.where(
                            region, 0.2 * rgb[..., channel] + 0.8 * color[channel], rgb[..., channel]
                        )
            else:
                rgb[hit, 0] = 1.0
                rgb[hit, 1] = 0.15
                rgb[hit, 2] = 0.15

        shown = np.rot90(rgb)
        self.ax.clear()
        self.ax.imshow(shown, aspect="equal", interpolation="bilinear")
        self._draw_sam_prompts(rgb.shape[1])
        self.ax.set_aspect("equal", adjustable="datalim")
        self._apply_brain_zoom(shown.shape[0], shown.shape[1], rgb.shape[1])
        self.ax.axis("off")
        self.canvas.draw_idle()

        self.title_label.setText(f"{self._base_title}  (SLICE {index})")
        self._sync_hive_overlay()

    def _draw_sam_prompts(self, pre_rot_cols: int):
        for x, y, label in self._sam_points:
            shown_x, shown_y = prompt_to_rotated_xy(x, y, pre_rot_cols)
            color = "#30d158" if label == 1 else "#ff3b30"
            marker = "+" if label == 1 else "x"
            self.ax.plot(shown_x, shown_y, marker=marker, color=color, markersize=9, markeredgewidth=2)
        if self._sam_box is not None:
            x0, y0, x1, y1 = self._sam_box
            shown_x0, shown_x1 = y0, y1
            shown_y0, shown_y1 = pre_rot_cols - 1 - x1, pre_rot_cols - 1 - x0
            self.ax.add_patch(Rectangle(
                (shown_x0, shown_y0), shown_x1 - shown_x0, shown_y1 - shown_y0,
                fill=False, edgecolor=ACCENT_AMBER_SOFT, linewidth=1.8,
            ))

    def _apply_brain_zoom(self, shown_h: int, shown_w: int, pre_rot_cols: int):
        """
        Crop the view to the brain bounding box after rot90 so the head
        fills more of the panel without changing voxel aspect ratio.
        """
        if self._bbox is None:
            return
        lo, hi = self._bbox
        if self.axis == 0:
            r0, r1 = lo[1], hi[1]
            c0, c1 = lo[2], hi[2]
        elif self.axis == 1:
            r0, r1 = lo[0], hi[0]
            c0, c1 = lo[2], hi[2]
        else:
            r0, r1 = lo[0], hi[0]
            c0, c1 = lo[1], hi[1]
        # np.rot90: source (row, col) -> displayed
        # (x=row, y=pre_rot_cols-1-col).
        xs = [r0, r1]
        ys = [pre_rot_cols - 1 - c0, pre_rot_cols - 1 - c1]
        x0, x1 = min(xs) - 0.5, max(xs) + 0.5
        y0, y1 = min(ys) - 0.5, max(ys) + 0.5
        self.ax.set_xlim(max(-0.5, x0), min(shown_w - 0.5, x1))
        self.ax.set_ylim(min(shown_h - 0.5, y1), max(-0.5, y0))

    def _show_hive_empty(self):
        self.ax.clear()
        self.ax.set_facecolor("#000000")
        self.ax.axis("off")
        self.canvas.draw_idle()

    def set_maximized(self, is_max: bool):
        self.maximize_btn.setText("🗗" if is_max else "⛶")
        self.maximize_btn.setToolTip("Restore" if is_max else "Maximize")
        self._sync_hive_overlay()

    def refresh_after_layout_change(self):
        """Redraw the slice after Qt has applied the panel's new geometry.

        An idle Matplotlib repaint can remain pending when the other dashboard
        columns are hidden. A later mouse event then flushes that paint, which
        makes the slice appear to expand only after the cursor is moved.
        """
        self.updateGeometry()
        self.canvas.updateGeometry()
        if self.volume is not None:
            self._draw_slice(self.slider.value())
        else:
            self.canvas.draw()

    def resizeEvent(self, event):
        """Re-render the current slice when this panel changes dimensions."""
        super().resizeEvent(event)
        if not hasattr(self, "canvas") or self._resize_redraw_pending:
            return
        self._resize_redraw_pending = True
        QTimer.singleShot(0, self._redraw_after_resize)

    def _redraw_after_resize(self):
        self._resize_redraw_pending = False
        self.refresh_after_layout_change()


def _inject_dark_page_style(html_path: str):
    """
    Plotly's write_html doesn't expose the page <body> background directly
    — only the plot's own paper/plot background, which leaves the HTML
    page's default white margin visible around the figure if the plot
    doesn't exactly fill the QWebEngineView. This patches a small <style>
    block into the generated file so the whole page matches the dark
    panel instead.
    """
    try:
        path = Path(html_path)
        content = path.read_text(encoding="utf-8")
    except OSError:
        return
    style_block = (
        f"<style>html,body{{margin:0;padding:0;background:{BG_PANEL};"
        f"overflow:hidden;}}</style>"
    )
    if "<head>" in content:
        content = content.replace("<head>", f"<head>{style_block}", 1)
        try:
            path.write_text(content, encoding="utf-8")
        except OSError:
            pass


def _inject_middle_click_pan(html_path: str):
    """
    Plotly's 3D scenes support panning natively via right-click-drag, but
    that fights with the browser's own context menu inside a
    QWebEngineView. This injects a small script that instead pans on
    middle-mouse-button drag (holding the scroll wheel down and moving
    the mouse) — left-drag still rotates and the wheel still zooms,
    exactly as before, this only adds the missing pan gesture.

    The math: compute the camera's current right and up vectors from its
    eye/center/up, then translate both eye and center together along
    those vectors by an amount proportional to the mouse movement — a
    true pan that keeps the viewing direction unchanged, rather than a
    rotation.
    """
    try:
        path = Path(html_path)
        content = path.read_text(encoding="utf-8")
    except OSError:
        return

    script_block = """
<script>
(function() {
    function setupMiddleClickPan() {
        var gd = document.querySelector('.plotly-graph-div');
        if (!gd) { setTimeout(setupMiddleClickPan, 100); return; }

        var dragging = false;
        var lastX = 0, lastY = 0;
        var panScale = 0.0022;

        gd.addEventListener('mousedown', function(e) {
            if (e.button === 1) {
                dragging = true;
                lastX = e.clientX;
                lastY = e.clientY;
                e.preventDefault();
            }
        });

        window.addEventListener('mouseup', function(e) {
            if (e.button === 1) { dragging = false; }
        });

        window.addEventListener('mousemove', function(e) {
            if (!dragging) return;
            e.preventDefault();

            var dx = e.clientX - lastX;
            var dy = e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;

            var layout = gd._fullLayout;
            var scene = layout && layout.scene;
            var camera = scene && scene.camera;
            if (!camera) return;

            var eye = camera.eye;
            var center = camera.center || {x: 0, y: 0, z: 0};
            var up = camera.up || {x: 0, y: 0, z: 1};

            var fx = center.x - eye.x, fy = center.y - eye.y, fz = center.z - eye.z;
            var flen = Math.sqrt(fx * fx + fy * fy + fz * fz) || 1;
            fx /= flen; fy /= flen; fz /= flen;

            var rx = fy * up.z - fz * up.y;
            var ry = fz * up.x - fx * up.z;
            var rz = fx * up.y - fy * up.x;
            var rlen = Math.sqrt(rx * rx + ry * ry + rz * rz) || 1;
            rx /= rlen; ry /= rlen; rz /= rlen;

            var ux = ry * fz - rz * fy;
            var uy = rz * fx - rx * fz;
            var uz = rx * fy - ry * fx;

            var dxp = -dx * panScale;
            var dyp = dy * panScale;

            var moveX = rx * dxp + ux * dyp;
            var moveY = ry * dxp + uy * dyp;
            var moveZ = rz * dxp + uz * dyp;

            Plotly.relayout(gd, {
                'scene.camera.eye': {x: eye.x + moveX, y: eye.y + moveY, z: eye.z + moveZ},
                'scene.camera.center': {x: center.x + moveX, y: center.y + moveY, z: center.z + moveZ}
            });
        });
    }
    setupMiddleClickPan();
})();
</script>
"""
    if "</body>" in content:
        content = content.replace("</body>", f"{script_block}</body>", 1)
    else:
        content += script_block

    try:
        path.write_text(content, encoding="utf-8")
    except OSError:
        pass


def _inject_tumor_selection(html_path: str):
    """Send tumor/background clicks to Qt through a private console prefix."""
    try:
        path = Path(html_path)
        content = path.read_text(encoding="utf-8")
    except OSError:
        return

    script_block = """
<script>
(function() {
    function setupTumorSelection() {
        var gd = document.querySelector('.plotly-graph-div');
        if (!gd || !gd.on) { setTimeout(setupTumorSelection, 100); return; }

        var downX = 0, downY = 0, plotlyHit = false;
        gd.addEventListener('mousedown', function(e) {
            if (e.button !== 0) return;
            downX = e.clientX; downY = e.clientY; plotlyHit = false;
        });
        gd.on('plotly_click', function(data) {
            plotlyHit = true;
            var point = data && data.points && data.points[0];
            var meta = point && point.data && point.data.meta;
            var id = meta && meta.tumor_component_id;
            console.log('__TUMOR_SELECT__:' + (id == null ? 'all' : id));
        });
        gd.addEventListener('mouseup', function(e) {
            if (e.button !== 0) return;
            var moved = Math.hypot(e.clientX - downX, e.clientY - downY) > 4;
            setTimeout(function() {
                if (!moved && !plotlyHit) console.log('__TUMOR_SELECT__:all');
            }, 80);
        });
    }
    setupTumorSelection();
})();
</script>
"""
    if "</body>" in content:
        content = content.replace("</body>", f"{script_block}</body>", 1)
    else:
        content += script_block
    try:
        path.write_text(content, encoding="utf-8")
    except OSError:
        pass


class _PlotlyPage(QWebEnginePage):
    tumor_selected = Signal(object)

    def javaScriptConsoleMessage(self, level, message, line_number, source_id):
        prefix = "__TUMOR_SELECT__:"
        if message.startswith(prefix):
            value = message[len(prefix):]
            self.tumor_selected.emit(None if value == "all" else int(value))
            return
        super().javaScriptConsoleMessage(level, message, line_number, source_id)


class Panel3D(HivePanel):
    """
    A bordered panel that shows an interactive 3D Plotly figure (rotate,
    zoom, pan) via an embedded web view. Qt has no native Plotly renderer,
    so the figure is written to a temporary standalone HTML file and loaded
    into a QWebEngineView — the same approach as save_figure_html() in the
    existing viewer3d.py, just pointed at a temp file instead of a
    user-chosen path.
    """

    tumor_selected = Signal(object)

    def __init__(self, title: str, min_height: int = 200, accent: str = ACCENT_TEAL):
        super().__init__()
        self.accent = accent
        self.setFrameShape(QFrame.Box)
        # No border at all, active or not — unlike the 2D panels, the 3D
        # panels stay borderless; the accent color is used elsewhere
        # (e.g. tumor-panel labeling) instead of as a frame outline here.
        self.setStyleSheet(_panel_style())
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        header.addStretch(1)
        self.title_label = TechLabel(title)
        header.addWidget(self.title_label)
        header.addStretch(1)
        self.maximize_btn = _fullscreen_button("Maximize")
        header.addWidget(self.maximize_btn)
        layout.addLayout(header)

        view_host = QWidget()
        view_stack = QStackedLayout(view_host)
        view_stack.setStackingMode(QStackedLayout.StackAll)

        self.web_view = QWebEngineView()
        self._plotly_page = _PlotlyPage(self.web_view)
        self._plotly_page.tumor_selected.connect(self.tumor_selected.emit)
        self.web_view.setPage(self._plotly_page)
        self.web_view.setStyleSheet(f"background-color: {BG_PANEL}; border: none;")
        self.web_view.page().setBackgroundColor(QColor(BG_PANEL))
        self.web_view.setHtml(
            f"<html><body style='margin:0;background:{BG_PANEL};'></body></html>"
        )
        view_stack.addWidget(self.web_view)

        self.placeholder_label = QLabel("(run segmentation or SAM to render)")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet(
            f"border: none; color: {TEXT_MUTED}; font-size: 12px; background: {BG_PANEL};"
        )
        view_stack.addWidget(self.placeholder_label)
        self.placeholder_label.raise_()
        layout.addWidget(view_host, stretch=1)

        self._layout = layout
        self._temp_files = []  # keep references so temp files aren't GC'd/deleted early

    def set_figure(self, fig: go.Figure):
        self.placeholder_label.hide()

        # Hide Plotly's icon toolbar (zoom/pan/camera/reset buttons) — the
        # panel is still fully interactive via mouse drag/scroll, just
        # without the visible icon strip.
        config = {**INTERACTION_CONFIG, "displayModeBar": False}

        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        fig.write_html(
            tmp.name,
            include_plotlyjs=True,
            full_html=True,
            config=config,
            default_width="100%",
            default_height="100%",
        )
        _inject_dark_page_style(tmp.name)
        _inject_middle_click_pan(tmp.name)
        _inject_tumor_selection(tmp.name)
        self._temp_files.append(tmp.name)
        self.web_view.load(QUrl.fromLocalFile(tmp.name))
        QTimer.singleShot(0, self._sync_hive_overlay)

    def select_tumor(self, component_id: Optional[int]):
        """Show one tumor trace, or all tumor traces when component_id is None."""
        selected = "null" if component_id is None else str(int(component_id))
        script = f"""
        (function() {{
            var gd = document.querySelector('.plotly-graph-div');
            if (!gd || !gd.data) return;
            var selected = {selected};
            var visibility = gd.data.map(function(trace) {{
                var meta = trace.meta || {{}};
                var id = meta.tumor_component_id;
                return id == null || selected == null || id === selected;
            }});
            Plotly.restyle(gd, {{visible: visibility}});
        }})();
        """
        self.web_view.page().runJavaScript(script)

    def clear_figure(self):
        """Restore the empty state when there is no tumor mask to render."""
        self.web_view.setHtml(
            f"<html><body style='margin:0;background:{BG_PANEL};'></body></html>"
        )
        self.placeholder_label.show()
        self.placeholder_label.raise_()
        self._sync_hive_overlay()

    def set_maximized(self, is_max: bool):
        self.maximize_btn.setText("🗗" if is_max else "⛶")
        self.maximize_btn.setToolTip("Restore" if is_max else "Maximize")
        self._sync_hive_overlay()


def _style_ui_figure(fig: go.Figure) -> go.Figure:
    """Strip notebook chrome so the mesh fills the dark Qt panel."""
    fig.update_layout(
        title=None,
        width=None,
        height=None,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PANEL,
        scene_bgcolor=BG_PANEL,
        showlegend=False,
        font=dict(color=TEXT_LIGHT),
    )
    return fig


def _prepare_3d(volume: np.ndarray, mask: Optional[np.ndarray], affine: np.ndarray):
    """Downsample for marching cubes and convert the affine into voxel spacing."""
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).astype(np.float64)
    longest = float(max(volume.shape))
    factor = min(1.0, RENDER_MAX_DIM / longest) if longest else 1.0
    if factor < 0.99:
        new_shape = tuple(max(8, int(round(size * factor))) for size in volume.shape)
        zooms = [new / old for new, old in zip(new_shape, volume.shape)]
        volume = zoom(volume, zooms, order=1)
        if mask is not None:
            mask = (zoom(mask.astype(np.float32), zooms, order=0) > 0.5).astype(np.float32)
        spacing = spacing / factor
    return volume, mask, tuple(float(s) for s in spacing)


def build_tumor_figure(mask: np.ndarray, spacing: tuple) -> go.Figure:
    """Tumor mesh alone, from the shared 3D viewer."""
    fig = show_volume_3d(
        mask,
        mask,
        show_brain=False,
        level=0.5,
        min_voxels=25,
        spacing=spacing,
        title="3D tumor",
        clean=True,
        show_legend=False,
    )
    return _style_ui_figure(_tag_tumor_traces(fig))


def _tag_tumor_traces(fig: go.Figure) -> go.Figure:
    """Attach stable component IDs used by the embedded Plotly click bridge."""
    for trace in fig.data:
        match = re.match(r"^#(\d+)\b", str(trace.name or ""))
        trace.meta = (
            {"tumor_component_id": int(match.group(1))}
            if match else {"surface": "brain"}
        )
    return fig


def build_brain_figure(
    volume: np.ndarray,
    mask: Optional[np.ndarray],
    spacing: tuple,
) -> go.Figure:
    """Brain shell, with the tumor inside it when a mask is available."""
    fig = show_volume_3d(
        volume,
        mask,
        show_brain=True,
        min_voxels=25 if mask is not None else 1,
        spacing=spacing,
        title="3D brain",
        clean=True,
        show_legend=False,
    )
    return _style_ui_figure(_tag_tumor_traces(fig))


class MainWindow(QMainWindow):
    def _size_to_screen(self):
        """
        Sizes and centers the window relative to whatever screen it's
        opened on — 85% of available space, capped so it doesn't become
        awkwardly huge on an ultrawide/4K monitor, with a sane minimum so
        it doesn't get squashed on a small laptop screen either.
        """
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1200, 720)
            return

        avail = screen.availableGeometry()
        width = max(min(int(avail.width() * 0.85), 1500), 900)
        height = max(min(int(avail.height() * 0.88), 980), 680)
        self.resize(width, height)

        x = avail.x() + (avail.width() - width) // 2
        y = avail.y() + (avail.height() - height) // 2
        self.move(x, y)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("A.I. Copilot for Cancer Detection and Patient Survival Prediction")
        self.setWindowIcon(_asset_icon(_APP_ICO_PATH if _APP_ICO_PATH.is_file() else _APP_ICON_PATH))
        self._size_to_screen()

        # Current case state — shared between "Enter image" and "Run
        # segmentation" so picking a patient folder once is enough for both.
        self.current_folder: Optional[Path] = None
        self._nii_files: list = []
        self._display_volume: Optional[np.ndarray] = None
        self._display_affine: Optional[np.ndarray] = None
        self._volume_transform: Optional[VolumeTransform] = None
        self._committed_mask: Optional[np.ndarray] = None
        self._committed_probs: Optional[np.ndarray] = None
        self._preview_mask: Optional[np.ndarray] = None
        self._mask_undo_stack: list[np.ndarray] = []
        self._mask_revision = 0
        self._mask_refresh_in_progress = False
        self._pending_mask_refresh = False
        self._sam_refiner = MedSAM2Refiner(MEDSAM2_CHECKPOINT)
        self._model = None
        self._device = None
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[_BackgroundJob] = None
        self._busy = False
        self._survival_stats = load_survival_stats()
        try:
            self._survival_table = load_survival_days()
        except (FileNotFoundError, KeyError):
            self._survival_table = {}
        shell = RootShell()

        # ---------------- Left column: image input + patient record -------
        left = QVBoxLayout()
        left.setSpacing(SPACING)

        self.drop_zone = DropVolumeZone()
        self.drop_zone.folder_chosen.connect(self._load_folder)
        self.drop_zone.setMinimumHeight(88)

        self.run_seg_btn = QPushButton()
        self.run_seg_btn.setFixedSize(88, 88)
        self.run_seg_btn.setIcon(_asset_icon(_RUN_ICON_PATH))
        self.run_seg_btn.setIconSize(QSize(56, 56))
        self.run_seg_btn.setToolTip("Run segmentation")
        self.run_seg_btn.setCursor(Qt.PointingHandCursor)
        self.run_seg_btn.setEnabled(False)
        self.run_seg_btn.setStyleSheet(
            f"QPushButton {{ border: 1.5px solid {ACCENT_AMBER}; border-radius: 16px; "
            f"background-color: #2a1c0c; padding: 8px; }}"
            f"QPushButton:hover {{ background-color: #3a2810; border-color: {ACCENT_AMBER_SOFT}; }}"
            f"QPushButton:pressed {{ background-color: #c46800; }}"
            f"QPushButton:disabled {{ background-color: #2c2c2e; border-color: #4a3a22; }}"
        )
        self.run_seg_btn.clicked.connect(self.on_run_segmentation)

        drop_row = QHBoxLayout()
        drop_row.setContentsMargins(0, 0, 0, 0)
        drop_row.setSpacing(8)
        drop_row.addWidget(self.drop_zone, 1)
        drop_row.addWidget(self.run_seg_btn, 0, Qt.AlignTop)
        left.addLayout(drop_row)

        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)  # indeterminate — pulses while active
        self.busy_bar.setTextVisible(False)
        self.busy_bar.hide()
        left.addWidget(self.busy_bar)

        self.patients_record_panel = PatientRecordPanel()
        left.addWidget(self.patients_record_panel, 1)

        self.survival_panel = SurvivalDaysPanel()
        left.addWidget(self.survival_panel)
        left.addSpacing(22)
        left.addWidget(SquadCredit())

        self.left_container = QWidget()
        self.left_container.setLayout(left)
        self.left_container.setMinimumWidth(200)
        self.left_container.setMaximumWidth(280)

        # ---------------- Center column: 2D views --------------------------
        # min_height is kept modest (not the visual target size) — it's only
        # the floor for the smallest supported window. QSizePolicy.Expanding
        # + the stretch factors below do the actual responsive growing, so
        # these panels fill available space on a normal-sized window without
        # ever needing a scrollbar at the small end either.
        center = QVBoxLayout()
        center.setSpacing(SPACING)
        # axis 0 = L-R (sagittal), axis 1 = A-P (coronal), axis 2 = S-I (axial)
        self.sagittal_panel = SlicePanel("2D sagittal view", axis=0, min_height=140)
        self.axial_panel = SlicePanel("2D axial view", axis=2, min_height=140)
        self.coronal_panel = SlicePanel("2D coronal view", axis=1, min_height=140)
        center.addWidget(self.sagittal_panel, stretch=1)
        center.addWidget(self.axial_panel, stretch=1)
        center.addWidget(self.coronal_panel, stretch=1)

        self.center_container = QWidget()
        self.center_container.setLayout(center)

        # ---------------- Right column: 3D views ---------------------------
        right = QVBoxLayout()
        right.setSpacing(SPACING)
        self.tumor_3d_panel = Panel3D("3D tumor", min_height=150, accent=ACCENT_AMBER)
        self.brain_3d_panel = Panel3D("3D brain view", min_height=200, accent=ACCENT_AMBER)
        # brain view gets more of the extra vertical space than the smaller
        # tumor-only panel above it, matching the original wireframe's
        # proportions (small panel on top, tall panel below)
        right.addWidget(self.tumor_3d_panel, stretch=1)
        right.addWidget(self.brain_3d_panel, stretch=2)

        self.right_container = QWidget()
        self.right_container.setLayout(right)
        self.right_container.setMinimumWidth(240)

        # ---------------- Assemble ------------------------------------------
        # Plain QHBoxLayout with stretch factors — pure responsive design.
        # Every column has QSizePolicy.Expanding, so the layout grows/shrinks
        # proportionally with the window on its own; no draggable splitter
        # and no scroll-area fallback needed, since the minimum heights above
        # were chosen to always fit within _size_to_screen()'s smallest
        # supported window size.
        self.sep_left = _column_separator()
        self.sep_right = _column_separator()

        body = QHBoxLayout()
        body.setSpacing(10)
        body.setContentsMargins(SPACING, 10, SPACING, SPACING)
        body.addWidget(self.left_container, stretch=0)
        body.addWidget(self.sep_left)
        body.addWidget(self.center_container, stretch=3)
        body.addWidget(self.sep_right)
        body.addWidget(self.right_container, stretch=2)

        page = QVBoxLayout(shell)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(0)
        page.addWidget(AppHeader())
        page.addLayout(body, stretch=1)

        self.setCentralWidget(shell)

        # ---------------- Maximize / restore wiring --------------------------
        self.center_panels = [self.sagittal_panel, self.axial_panel, self.coronal_panel]
        self.right_panels = [self.tumor_3d_panel, self.brain_3d_panel]
        self.all_panels = self.center_panels + self.right_panels
        self._maximized_panel = None

        for panel in self.all_panels:
            panel.maximize_btn.clicked.connect(lambda checked=False, p=panel: self.toggle_maximize(p))
        for panel in self.right_panels:
            panel.tumor_selected.connect(self.on_tumor_selected)
        for panel in self.center_panels:
            panel.refinement_requested.connect(self.on_run_sam_refinement)
            panel.accept_requested.connect(self.on_accept_sam_refinement)
            panel.discard_requested.connect(self.on_discard_sam_refinement)
            panel.mask_edited.connect(self.on_mask_edited)
            panel.mask_undo_requested.connect(self.on_mask_undo)

    def on_tumor_selected(self, component_id):
        """Synchronize tumor isolation across both interactive 3D panels."""
        for panel in self.right_panels:
            panel.select_tumor(component_id)

    def toggle_maximize(self, panel):
        """Clicking a panel's maximize button hides every other panel (and
        the now-empty side column) so it fills the available space; clicking
        it again restores the normal 3-column grid. A plain QHBoxLayout
        already collapses a hidden widget's space to its neighbors, so no
        manual width juggling is needed here."""
        if self._maximized_panel is panel:
            # restore
            self.left_container.show()
            self.center_container.show()
            self.right_container.show()
            self.sep_left.show()
            self.sep_right.show()
            for p in self.all_panels:
                p.show()
            panel.set_maximized(False)
            self._maximized_panel = None
            if isinstance(panel, SlicePanel):
                QTimer.singleShot(0, panel.refresh_after_layout_change)
            return

        if self._maximized_panel is not None:
            self._maximized_panel.set_maximized(False)

        self.left_container.hide()
        self.sep_left.hide()
        self.sep_right.hide()
        if panel in self.center_panels:
            self.right_container.hide()
            self.center_container.show()
            for p in self.center_panels:
                p.setVisible(p is panel)
        else:
            self.center_container.hide()
            self.right_container.show()
            for p in self.right_panels:
                p.setVisible(p is panel)

        panel.set_maximized(True)
        self._maximized_panel = panel
        if isinstance(panel, SlicePanel):
            QTimer.singleShot(0, panel.refresh_after_layout_change)

    def _start_background(self, fn, on_ok, on_err) -> bool:
        if self._busy or (self._worker_thread is not None and self._worker_thread.isRunning()):
            return False
        self._busy = True
        worker = _BackgroundJob(fn)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.ok.connect(on_ok)
        worker.err.connect(on_err)
        worker.ok.connect(thread.quit)
        worker.err.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_worker_finished)
        self._worker = worker
        self._worker_thread = thread
        thread.start()
        return True

    def _on_worker_finished(self):
        self._worker = None
        self._worker_thread = None
        if self._pending_mask_refresh and self._committed_mask is not None:
            self._pending_mask_refresh = False
            QTimer.singleShot(0, lambda: self._start_mask_refresh(self._committed_mask))

    def on_run_segmentation(self):
        if self.current_folder is None or self._busy:
            return

        folder_path = self.current_folder
        survival_stats = self._survival_stats

        self.busy_bar.show()
        self.run_seg_btn.setEnabled(False)
        self._set_2d_loading(True)

        def job():
            modality_paths = MainWindow._find_modality_files(folder_path)
            display_volume, model_input, affine, transform = MainWindow._load_and_stack(modality_paths)
            model, device = _ensure_model()
            mask, probs, predicted_days = MainWindow._run_model_static(
                model, device, model_input, display_volume.shape, survival_stats
            )
            volume_3d, mask_3d, spacing = _prepare_3d(display_volume, mask, affine)
            has_tumor = mask_3d is not None and float(np.asarray(mask_3d).sum()) > 0
            tumor_fig = build_tumor_figure(mask_3d, spacing) if has_tumor else None
            brain_fig = build_brain_figure(volume_3d, mask_3d, spacing)
            return {
                "display_volume": display_volume,
                "affine": affine,
                "transform": transform,
                "mask": mask,
                "probs": probs,
                "predicted_days": predicted_days,
                "tumor_fig": tumor_fig,
                "brain_fig": brain_fig,
            }

        self._start_background(job, self._on_segmentation_ready, self._on_segmentation_failed)

    def _on_segmentation_ready(self, result: dict):
        display_volume = result["display_volume"]
        affine = result["affine"]
        mask = result["mask"]
        probs = result["probs"]

        self.sagittal_panel.set_volume(display_volume)
        self.axial_panel.set_volume(display_volume)
        self.coronal_panel.set_volume(display_volume)

        voxel_vol_cm3 = self._voxel_volume_cm3(affine)
        labels, _components, color_map = self._label_tumor_components(mask, probs, voxel_vol_cm3)
        self.sagittal_panel.set_detection(probs, labels, color_map)
        self.axial_panel.set_detection(probs, labels, color_map)
        self.coronal_panel.set_detection(probs, labels, color_map)
        self._set_2d_loading(False)

        self.patients_record_panel.set_volume(float(mask.sum()) * voxel_vol_cm3)
        self.survival_panel.set_days(result["predicted_days"])
        self._display_volume = display_volume
        self._display_affine = affine
        self._volume_transform = result["transform"]
        self._committed_mask = mask.copy()
        self._committed_probs = probs.copy()
        self._preview_mask = None
        self._mask_undo_stack.clear()
        self._set_mask_undo_available()
        self._set_sam_available(True)

        self.busy_bar.hide()
        self.run_seg_btn.setEnabled(True)
        self._busy = False

        # 2D results stay on this window. Load 3D on the next tick into the
        # already-created web views so Qt does not remount the right column
        # (that remount was flashing as if the window closed and reopened).
        self._pending_3d = (result.get("tumor_fig"), result.get("brain_fig"))
        QTimer.singleShot(0, self._apply_pending_3d)

    def _apply_pending_3d(self):
        pending = getattr(self, "_pending_3d", None)
        self._pending_3d = None
        if not pending:
            return
        tumor_fig, brain_fig = pending
        try:
            if tumor_fig is not None:
                self.tumor_3d_panel.set_figure(tumor_fig)
            if brain_fig is not None:
                self.brain_3d_panel.set_figure(brain_fig)
        except Exception as exc:
            QMessageBox.warning(self, "3D render failed", str(exc))

    def _on_segmentation_failed(self, message: str):
        self.busy_bar.hide()
        self._set_2d_loading(False)
        self.run_seg_btn.setEnabled(self.current_folder is not None)
        self._busy = False
        QMessageBox.critical(self, "Inference failed", message)

    def _label_tumor_components(self, mask: np.ndarray, probs: Optional[np.ndarray], voxel_vol_cm3: float):
        """
        Splits the binary mask into separate connected components — a
        brain can have more than one lesion — and assigns each a distinct
        color from COMPONENT_PALETTE, largest volume first, so the same
        color consistently identifies the same tumor across the 2D
        heatmap, the report panel, and the PDF export.

        Returns (labels, components, color_map):
            labels:     int array, same shape as mask (0=background, 1..N=component id)
            components: list of dicts, sorted largest-first, each with
                        label_id, voxels, volume_cm3, confidence, rank,
                        color_name, color_hex
            color_map:  {label_id: hex_color}, for passing straight into
                        SlicePanel.set_detection()
        """
        labels, n_components = ndi_label(mask > 0.5)

        entries = []
        for label_id in range(1, n_components + 1):
            comp_mask = labels == label_id
            voxels = int(comp_mask.sum())
            if voxels == 0:
                continue
            entries.append({
                "label_id": label_id,
                "voxels": voxels,
                "volume_cm3": voxels * voxel_vol_cm3,
                "confidence": float(probs[comp_mask].mean()) if probs is not None else 1.0,
            })

        # Largest first, so "#1" consistently means the biggest lesion
        # rather than an arbitrary scan-order label id.
        entries.sort(key=lambda e: e["volume_cm3"], reverse=True)

        color_map = {}
        for rank, entry in enumerate(entries):
            color_name, color_hex = COMPONENT_PALETTE[rank % len(COMPONENT_PALETTE)]
            entry["rank"] = rank + 1
            entry["color_name"] = color_name
            entry["color_hex"] = color_hex
            color_map[entry["label_id"]] = color_hex

        return labels, entries, color_map

    def _set_2d_loading(self, active: bool):
        """Shows/hides the spinner overlay on all three 2D panels at once."""
        for panel in self.center_panels:
            panel.set_loading(active)

    def _set_sam_available(self, available: bool):
        for panel in self.center_panels:
            panel.enable_sam(available)

    def _set_sam_busy(self, busy: bool):
        self.drop_zone.setEnabled(not busy)
        self.run_seg_btn.setEnabled(not busy and self.current_folder is not None)
        for panel in self.center_panels:
            panel.set_sam_busy(busy)

    def _set_sam_preview_available(self, available: bool):
        for panel in self.center_panels:
            panel.set_sam_preview_available(available)

    def _set_mask_undo_available(self):
        available = bool(self._mask_undo_stack)
        for panel in self.center_panels:
            panel.set_mask_undo_available(available)

    def _show_binary_mask(self, mask: np.ndarray):
        voxel_vol_cm3 = self._voxel_volume_cm3(self._display_affine)
        labels, _components, color_map = self._label_tumor_components(mask, None, voxel_vol_cm3)
        for panel in self.center_panels:
            panel.set_refined_detection(mask, labels, color_map)
        self.patients_record_panel.set_volume(float(mask.sum()) * voxel_vol_cm3)

    def _show_committed_mask(self):
        if self._committed_mask is None:
            for panel in self.center_panels:
                panel.clear_detection()
            self.patients_record_panel.set_volume(None)
            return
        if self._committed_probs is not None:
            voxel_vol_cm3 = self._voxel_volume_cm3(self._display_affine)
            labels, _components, color_map = self._label_tumor_components(
                self._committed_mask, self._committed_probs, voxel_vol_cm3
            )
            for panel in self.center_panels:
                panel.set_detection(self._committed_probs, labels, color_map)
            self.patients_record_panel.set_volume(float(self._committed_mask.sum()) * voxel_vol_cm3)
        else:
            self._show_binary_mask(self._committed_mask)

    def on_mask_edited(self, stroke: dict):
        """Apply one 2D eraser stroke to the shared 3D tumor mask."""
        if self._committed_mask is None or self._busy:
            return

        # Edit the authoritative mask in place while dragging. Copying and
        # relabeling the whole 3D volume for every mouse event makes an eraser
        # feel delayed on clinical-size scans.
        mask = self._committed_mask
        if stroke.get("start", False):
            # Boolean snapshots use one byte per voxel instead of retaining
            # the model's larger floating-point mask representation.
            self._mask_undo_stack.append(mask > 0.5)
            if len(self._mask_undo_stack) > 10:
                del self._mask_undo_stack[0]
            self._set_mask_undo_available()
        plane = stroke["plane"]
        index = int(stroke["slice_index"])
        radius = max(1, int(stroke["radius"]))
        final = bool(stroke.get("final", False))

        if plane == "sagittal":
            view = mask[index, :, :]
        elif plane == "coronal":
            view = mask[:, index, :]
        else:
            view = mask[:, :, index]

        rows, cols = view.shape
        yy, xx = np.ogrid[:rows, :cols]
        points = stroke.get("points", [])
        # Interpolate between sampled mouse positions so fast drags make a
        # continuous stroke rather than a row of disconnected circles.
        dense_points = []
        for point_index, (x, y) in enumerate(points):
            if point_index == 0:
                dense_points.append((x, y))
                continue
            px, py = points[point_index - 1]
            steps = max(abs(x - px), abs(y - py), 1)
            dense_points.extend(
                (int(round(px + (x - px) * step / steps)),
                 int(round(py + (y - py) * step / steps)))
                for step in range(1, steps + 1)
            )
        for x, y in dense_points:
            if 0 <= x < cols and 0 <= y < rows:
                disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
                view[disk] = 0.0

        self._committed_probs = None
        self._preview_mask = None
        self._set_sam_preview_available(False)
        if final:
            # Synchronize 2D immediately and build expensive 3D surfaces on
            # the worker thread, using the same fast path as Undo.
            self._start_mask_refresh(mask)
        else:
            active_panel = next(
                (panel for panel in self.center_panels if panel.plane == plane), None
            )
            if active_panel is not None:
                active_panel.set_mask(mask)

    def on_mask_undo(self):
        """Restore the mask as it was before the most recent eraser stroke."""
        if not self._mask_undo_stack or self._display_volume is None:
            return
        self._committed_mask = self._mask_undo_stack.pop().astype(np.float32)
        self._committed_probs = None
        self._preview_mask = None
        self._set_sam_preview_available(False)
        self._start_mask_refresh(self._committed_mask)

    def _start_mask_refresh(self, corrected_mask: np.ndarray):
        """Show a corrected mask now and refresh component/3D data off-thread."""
        if self._display_volume is None or self._display_affine is None:
            return

        # Give immediate feedback first. Connected-component analysis and 3D
        # surface extraction are deliberately moved to the worker below.
        self._mask_revision += 1
        revision = self._mask_revision
        mask = corrected_mask.copy()
        volume = self._display_volume.copy()
        affine = self._display_affine.copy()
        for panel in self.center_panels:
            panel.set_mask(self._committed_mask)
            panel.enable_mask_editing(False)
        voxel_vol_cm3 = self._voxel_volume_cm3(affine)
        self.patients_record_panel.set_volume(float(mask.sum()) * voxel_vol_cm3)
        self.busy_bar.show()

        # A prior surface build cannot be cancelled safely. Keep the newly
        # restored 2D mask visible now and queue its 3D refresh; the old result
        # is rejected by its revision number below.
        if self._mask_refresh_in_progress:
            self._pending_mask_refresh = True
            self._set_mask_undo_available()
            return

        def job():
            labels, _components, color_map = self._label_tumor_components(
                mask, None, voxel_vol_cm3
            )
            volume_3d, mask_3d, spacing = _prepare_3d(volume, mask, affine)
            has_tumor = bool(np.any(mask_3d))
            tumor_fig = build_tumor_figure(mask_3d, spacing) if has_tumor else None
            brain_fig = build_brain_figure(volume_3d, mask_3d, spacing)
            return {
                "mask": mask,
                "labels": labels,
                "color_map": color_map,
                "tumor_fig": tumor_fig,
                "brain_fig": brain_fig,
                "revision": revision,
            }

        started = self._start_background(
            job, self._on_mask_refresh_ready, self._on_mask_refresh_failed
        )
        self._mask_refresh_in_progress = started
        if not started:
            self.busy_bar.hide()
            for panel in self.center_panels:
                panel.enable_mask_editing(True)
        self._set_mask_undo_available()

    def _on_mask_refresh_ready(self, result: dict):
        """Finish a deferred component and 3D refresh after an edit."""
        if result["revision"] == self._mask_revision:
            for panel in self.center_panels:
                panel.set_refined_detection(
                    result["mask"], result["labels"], result["color_map"]
                )
                panel.enable_mask_editing(True)
            if result["tumor_fig"] is None:
                self.tumor_3d_panel.clear_figure()
            else:
                self.tumor_3d_panel.set_figure(result["tumor_fig"])
            self.brain_3d_panel.set_figure(result["brain_fig"])
        self.busy_bar.hide()
        self._busy = False
        self._mask_refresh_in_progress = False
        self._set_mask_undo_available()

    def _on_mask_refresh_failed(self, message: str):
        # The restored 2D mask remains valid even if optional 3D rendering
        # fails, so keep it and simply re-enable editing.
        self.busy_bar.hide()
        self._busy = False
        self._mask_refresh_in_progress = False
        for panel in self.center_panels:
            panel.enable_mask_editing(True)
        self._set_mask_undo_available()
        QMessageBox.warning(self, "3D mask refresh failed", message)

    def on_run_sam_refinement(self, prompt: dict):
        if self._busy or self._display_volume is None:
            return
        volume = self._display_volume.copy()
        seed = self._committed_mask.copy() if self._committed_mask is not None else None
        self.busy_bar.show()
        self._set_sam_busy(True)
        self._set_2d_loading(True)

        def job():
            mask = self._sam_refiner.refine(
                volume=volume,
                seed_mask=seed,
                plane=prompt["plane"],
                slice_index=prompt["slice_index"],
                box=prompt["box"],
                points=prompt["points"],
                point_labels=prompt["point_labels"],
            )
            volume_3d, mask_3d, spacing = _prepare_3d(volume, mask, self._display_affine)
            tumor_fig = build_tumor_figure(mask_3d, spacing) if np.any(mask_3d) else None
            brain_fig = build_brain_figure(volume_3d, mask_3d, spacing)
            return {"mask": mask, "tumor_fig": tumor_fig, "brain_fig": brain_fig}

        self._start_background(job, self._on_sam_refinement_ready, self._on_sam_refinement_failed)

    def _on_sam_refinement_ready(self, result: dict):
        self._preview_mask = result["mask"]
        self._show_binary_mask(self._preview_mask)
        self._set_2d_loading(False)
        self._set_sam_busy(False)
        self._set_sam_preview_available(True)
        self.busy_bar.hide()
        self._busy = False
        self._pending_3d = (result.get("tumor_fig"), result.get("brain_fig"))
        QTimer.singleShot(0, self._apply_pending_3d)

    def _on_sam_refinement_failed(self, message: str):
        self._set_2d_loading(False)
        self._set_sam_busy(False)
        self.busy_bar.hide()
        self._busy = False
        QMessageBox.critical(self, "MedSAM2 refinement failed", message)

    def on_accept_sam_refinement(self):
        if self._preview_mask is None or self._volume_transform is None or self.current_folder is None:
            return
        destination = REFINED_MASK_DIR / f"{self.current_folder.name}-sam2-refined.nii.gz"
        try:
            self._volume_transform.save_mask(self._preview_mask, destination)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save refined mask", str(exc))
            return
        self._committed_mask = self._preview_mask.copy()
        self._committed_probs = None
        self._preview_mask = None
        self._mask_undo_stack.clear()
        self._set_mask_undo_available()
        self._set_sam_preview_available(False)
        for panel in self.center_panels:
            panel.clear_sam_prompts()
        self._show_committed_mask()
        self._update_3d_views(self._display_volume, self._committed_mask, self._display_affine)
        QMessageBox.information(self, "Refined mask saved", f"Saved to:\n{destination}")

    def on_discard_sam_refinement(self):
        if self._preview_mask is None:
            return
        self._preview_mask = None
        self._set_sam_preview_available(False)
        self._show_committed_mask()
        self._update_3d_views(self._display_volume, self._committed_mask, self._display_affine)

    def _update_3d_views(
        self,
        display_volume: np.ndarray,
        mask: Optional[np.ndarray],
        affine: np.ndarray,
    ):
        """
        Build the two 3D Plotly figures via `show_volume_3d` and load them
        into the right-column panels. A None/empty mask still draws the brain.
        """
        volume_3d, mask_3d, spacing = _prepare_3d(display_volume, mask, affine)

        try:
            has_tumor = mask_3d is not None and float(np.asarray(mask_3d).sum()) > 0
            if has_tumor:
                self.tumor_3d_panel.set_figure(build_tumor_figure(mask_3d, spacing))
            else:
                self.tumor_3d_panel.clear_figure()
            self.brain_3d_panel.set_figure(build_brain_figure(volume_3d, mask_3d, spacing))
        except Exception as exc:
            QMessageBox.warning(self, "3D render failed", str(exc))

    @staticmethod
    def _find_modality_files(folder: Path) -> dict:
        """
        Looks for files whose names contain a modality tag (case-insensitive),
        immediately preceded by a hyphen or underscore and followed by a dot
        or another separator — e.g. "BraTS-PED-00005-000-t1n.nii.gz" (BraTS
        uses hyphens) or "patient_01_t1.nii.gz" (underscore convention).

        Tries both common BraTS naming conventions per modality:
            - classic BraTS:  t1 / t1ce / t2 / flair
            - BraTS-PEDs:     t1n / t1c / t2w / t2f
        (t1n=T1 native, t1c=T1 post-contrast, t2w=T2-weighted, t2f=T2-FLAIR —
        matching the MODALITIES list in your existing config.py)
        """
        tag_options = {
            "t1c": ["t1c", "t1ce"],
            "t1n": ["t1n", "t1"],
            "t2f": ["t2f", "flair"],
            "t2w": ["t2w", "t2"],
        }
        candidates = list(folder.glob("*.nii*"))
        found = {}
        for key in MODALITIES:
            tags = tag_options[key]
            match = None
            for tag in tags:
                pattern = re.compile(rf"[_-]{tag}[_.]", re.IGNORECASE)
                hits = [p for p in candidates if pattern.search(p.name)]
                if hits:
                    match = hits[0]
                    break
            if match is None:
                found_names = "\n  ".join(p.name for p in candidates) or "(no .nii/.nii.gz files found)"
                raise FileNotFoundError(
                    f"No file matching modality '{key}' (tried tags {tags}) in {folder}\n"
                    f"Files present:\n  {found_names}"
                )
            found[key] = match
        return found

    @staticmethod
    def _load_and_stack(paths: dict):
        """
        Loads the 4 modalities, returns:
            display_volume: normalized FLAIR volume for background display (D,H,W)
            model_input:    torch tensor [1, 4, D, H, W], z-score normalized per channel
            affine:         nibabel affine of the FLAIR image, for voxel-size calculation
        Assumes all 4 modalities are already co-registered to the same grid
        (standard for BraTS-style preprocessed data). If yours aren't, resample
        them to a common grid before this step.
        """
        arrays = {}
        affine = None
        flair_image = None
        for key in MODALITIES:
            img = nib.load(str(paths[key]))
            data = img.get_fdata().astype(np.float32)
            if key == "t2f":
                affine = img.affine
                flair_image = img
            arrays[key] = data

        shape = arrays["t2f"].shape
        for key, arr in arrays.items():
            if arr.shape != shape:
                raise ValueError(
                    f"Modality '{key}' has shape {arr.shape}, expected {shape} "
                    f"(all 4 modalities must be co-registered to the same grid)"
                )

        # Keep preprocessing identical to training: normalize each native
        # modality and let _run_model_static resize once to TARGET_SHAPE.
        flair = arrays["t2f"]
        d_min, d_max = float(flair.min()), float(flair.max())
        display_volume = (flair - d_min) / (d_max - d_min) if d_max > d_min else flair
        display_volume, affine = _standardize_volume(display_volume, affine)

        channels = [_normalize(arrays[key]) for key in MODALITIES]
        stacked = np.stack(channels, axis=0)  # [4, D, H, W]
        tensor = torch.from_numpy(stacked).unsqueeze(0).float()  # [1, 4, D, H, W]
        transform = VolumeTransform.from_image(flair_image, STANDARD_DISPLAY_SHAPE)
        return display_volume, tensor, affine, transform

    def _get_model(self):
        model, device = _ensure_model()
        self._model = model
        self._device = device
        if self._survival_stats is None:
            self._survival_stats = load_survival_stats()
        return model

    @staticmethod
    def _run_model_static(model, device, model_input: torch.Tensor, target_shape: tuple, survival_stats):
        x = model_input.to(device)
        resized = torch.nn.functional.interpolate(
            x, size=INFERENCE_SIZE, mode="trilinear", align_corners=False
        )
        with torch.no_grad():
            outputs = run_model(model, resized)
            logits, _ = split_model_outputs(outputs)
            probs = torch.sigmoid(logits)
        probs_native = torch.nn.functional.interpolate(
            probs, size=target_shape, mode="trilinear", align_corners=False
        )
        probs_np = probs_native.squeeze(0).squeeze(0).cpu().numpy()
        mask = (probs_np > 0.5).astype(np.float32)
        predicted_days = None
        if survival_stats is not None:
            try:
                predicted_days = predict_survival_days(
                    model, resized.squeeze(0), device, survival_stats
                )
            except AttributeError:
                predicted_days = None
        return mask, probs_np, predicted_days

    def _run_model(self, model, model_input: torch.Tensor, target_shape: tuple):
        return self._run_model_static(
            model, self._device, model_input, target_shape, self._survival_stats
        )

    @staticmethod
    def _voxel_volume_cm3(affine: np.ndarray) -> float:
        """Voxel volume in cm^3, from the NIfTI affine's voxel dimensions (mm)."""
        voxel_dims_mm = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        voxel_volume_mm3 = float(np.prod(voxel_dims_mm))
        return voxel_volume_mm3 / 1000.0

    def _load_folder(self, folder_path: Path) -> bool:
        """
        Loads a patient folder and previews FLAIR (or the first NIfTI).
        Heavy I/O runs on a worker thread so the 2D loading animation can paint.
        """
        if self._busy:
            return False
        folder_path = Path(folder_path)
        nii_files = sorted(folder_path.glob("*.nii*"))
        if not nii_files:
            QMessageBox.warning(
                self, "No images found",
                f"No .nii or .nii.gz files found in:\n{folder_path}"
            )
            return False

        self.current_folder = folder_path
        self._nii_files = nii_files
        case_id = folder_path.name
        self.drop_zone.set_case(case_id)
        self.patients_record_panel.set_case(
            case_id, self._survival_table.get(case_id)
        )
        self.survival_panel.set_days(None)
        self._committed_mask = None
        self._committed_probs = None
        self._preview_mask = None
        self._mask_undo_stack.clear()
        self._set_mask_undo_available()
        self._volume_transform = None
        self._set_sam_available(False)
        self._set_sam_preview_available(False)
        self.run_seg_btn.setEnabled(False)

        default_index = 0
        for i, f in enumerate(nii_files):
            if re.search(r"[_-](t2f|flair)[_.]", f.name, re.IGNORECASE):
                default_index = i
                break
        preview_path = nii_files[default_index]

        self._set_2d_loading(True)

        def job():
            return MainWindow._load_volume(str(preview_path))

        started = self._start_background(job, self._on_preview_ready, self._on_preview_failed)
        if not started:
            self._set_2d_loading(False)
            self.run_seg_btn.setEnabled(True)
            return False
        return True

    def _on_preview_ready(self, result):
        volume, affine, transform = result
        self.sagittal_panel.set_volume(volume)
        self.axial_panel.set_volume(volume)
        self.coronal_panel.set_volume(volume)
        self._display_volume = volume
        self._display_affine = affine
        self._volume_transform = transform
        self._set_2d_loading(False)
        self.run_seg_btn.setEnabled(True)
        self._set_sam_available(True)
        self._busy = False

    def _on_preview_failed(self, message: str):
        self._set_2d_loading(False)
        self.run_seg_btn.setEnabled(self.current_folder is not None)
        self._busy = False
        QMessageBox.critical(self, "Failed to load volume", message)

    @staticmethod
    def _load_volume(path: str):
        """
        Loads a NIfTI file and returns a normalized 3D numpy array ready for
        display (values roughly 0-1, resampled to STANDARD_DISPLAY_SHAPE)
        plus a matching affine and native-space transform. The transform lets
        MedSAM2 export a mask even when the U-Net has not been run.
        """
        img = nib.load(path)
        data = img.get_fdata()

        if data.ndim == 4:
            # Some NIfTI files carry a trailing singleton or modality dim;
            # take the first volume/channel so we always end up 3D.
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D volume, got shape {data.shape}")

        data = data.astype(np.float32)
        d_min, d_max = float(data.min()), float(data.max())
        if d_max > d_min:
            data = (data - d_min) / (d_max - d_min)
        volume, affine = _standardize_volume(data, img.affine)
        transform = VolumeTransform.from_image(img, STANDARD_DISPLAY_SHAPE)
        return volume, affine, transform


def _pin_windows_app_id():
    """Stop Windows from pinning the generic python.exe icon on the taskbar."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "UniKL.MIIT.AICopilot.TumorViewer"
        )
    except Exception:
        pass


def main():
    _pin_windows_app_id()
    app = QApplication(sys.argv)
    icon = _asset_icon(_APP_ICO_PATH if _APP_ICO_PATH.is_file() else _APP_ICON_PATH)
    app.setWindowIcon(icon)
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
