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
import time
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

from PySide6.QtCore import Qt, QUrl, QTimer, QPoint, QPointF, QRectF, QObject, QThread, Signal, QSize
from PySide6.QtGui import (
    QColor, QFont, QIcon, QImage, QLinearGradient, QPainter, QPainterPath,
    QPen, QPixmap, QPolygonF, QRadialGradient, QRegion,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QFileDialog, QSizePolicy, QSlider, QMessageBox,
    QProgressBar, QStackedLayout, QScrollArea, QGridLayout, QButtonGroup,
)
from PySide6.QtWebEngineWidgets import QWebEngineView

import numpy as np
import nibabel as nib
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom, label as ndi_label, center_of_mass as ndi_center_of_mass, gaussian_filter
from skimage.measure import marching_cubes
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
# Blue iridescent theme: indigo glass, sky-blue accent, violet / magenta / teal highlights.
# ---------------------------------------------------------------------------
# Clean minimal theme: white cards on a soft off-white page, one blue accent,
# and a dark "Dynamic Island" pill in the header holding the main actions.
BG_PAGE = "#F5F8FD"
BG_PANEL = "#FFFFFF"                       # cards, dialogs, Qt defaults
GLASS_BG = "#FFFFFF"                       # card fill (name kept from the old glass theme)
GLASS_EDGE = "#E6ECF5"                     # hairline card border
BORDER_DIM = "#DCE4F0"
ACCENT = "#1F6FEB"                         # primary blue
ACCENT_SOFT = "#2F7CF6"
ACCENT_DEEP = "#1558C9"
ACCENT_VIOLET = "#3B82F6"
ACCENT_MAGENTA = "#5AA2FF"
ACCENT_TEAL = "#3B82F6"                    # loading effects over the dark scan viewport
ACCENT_TEAL_SOFT = "#1F6FEB"
ACCENT_CORAL = ACCENT
ACCENT_CORAL_SOFT = ACCENT_SOFT
TEXT_LIGHT = "#0E1B33"                     # primary text (name kept; now dark on white)
TEXT_MUTED = "#6B7A93"
TEXT_DIM = "#A3AFC2"
ISLAND_BG = "#0A1224"                      # the header "Dynamic Island" pill
ISLAND_TEXT = "#C9D6EE"
SCAN_BG = "#070B16"                        # dark viewport behind MRI slices
SCAN_TEXT = "#9CC7FF"
# Primary buttons, slider handles and bars (Qt stylesheet syntax).
GRADIENT_QSS = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1F6FEB, stop:1 #3B82F6)"
GRADIENT_QSS_HOVER = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2F7CF6, stop:1 #5AA2FF)"
# Ambient motion is off in the minimal theme (static, calm background).
AMBIENT_MOTION = False    # False = still background (lighter on older laptops)
AMBIENT_FPS = 15          # repaint rate of the background animation
HEADER_SHOW_BRAIN_ART = False   # the old amber header artwork; off in the blue theme
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_HEADER_BRAIN_PATH = _ASSETS_DIR / "header_brain.png"
_UNIKL_LOGO_PATH = _ASSETS_DIR / "unikl_logo.png"
_APP_ICON_PATH = _ASSETS_DIR / "app_icon.png"
_APP_ICO_PATH = _ASSETS_DIR / "app_icon.ico"
_RUN_ICON_PATH = _ASSETS_DIR / "icon_run.png"
SURVIVAL_EDGE_GAP = 8
SURVIVAL_DIGITS_HEIGHT = 64
_ICON_CACHE: dict = {}
_HIVE_RGB_CACHE: dict = {}

# Distinct colors assigned to separate tumor components (a brain can have
# more than one lesion) — largest lesion gets the first color, and the
# same color is used consistently across the 2D heatmap, the report
# panel, and the PDF export so "the blue one" means the same tumor
# everywhere. Cycles if there are more components than colors.
COMPONENT_PALETTE = [
    ("Red", "#E5484D"),
    ("Amber", "#F5A524"),
    ("Violet", "#8E6CEF"),
    ("Green", "#30A46C"),
    ("Pink", "#E54D9A"),
    ("Teal", "#12A594"),
    ("Orange", "#F76B15"),
    ("Brown", "#AD7F58"),
]


def _hex_to_rgb01(hex_color: str) -> tuple:
    """'#ff3b30' -> (1.0, 0.23, 0.19), for blending into a matplotlib RGB array."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)

# Single spacing unit used everywhere: gap between columns, gap between
# panels within a column, gap between buttons, and the window's outer
# margin. One number instead of several near-matching ones keeps the
# rhythm consistent across the whole interface.
SPACING = 12        # gap between stacked panels and around the window edge
COLUMN_GAP = 14     # gap between the report, 2D and 3D columns
SAM_CARD_GAP = 10   # gap between each 2D view and its MedSAM2 card
DROP_ROW_HEIGHT = 96                              # drop zone height = Run + gap + Clear
ACTION_BTN_GAP = 8
ACTION_BTN_W = 88
ACTION_BTN_H = (DROP_ROW_HEIGHT - ACTION_BTN_GAP) // 2   # 44 each
LEFT_COL_MAX = 300    # 2D slice column
RIGHT_COL_MAX = 340   # 3D tumour / record / report column
QT_MAX = 16777215

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
QToolTip {{
    color: #ffffff;
    background-color: {ISLAND_BG};
    border: none;
    border-radius: 6px;
    padding: 4px 8px;
}}
QSlider {{
    background: transparent;
    min-height: 18px;
    max-height: 18px;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: #E3ECFA;
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: #BFD6FB;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: #ffffff;
    border: 2px solid {ACCENT};
    width: 10px;
    height: 10px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:disabled {{
    background: #ffffff;
    border: 2px solid #D5DDEA;
}}
QProgressBar {{
    border: none;
    border-radius: 3px;
    background-color: #E8EFFA;
    height: 6px;
}}
QProgressBar::chunk {{
    background: {GRADIENT_QSS};
    border-radius: 3px;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 2px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: rgba(31, 111, 235, 0.25);
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
    border = f"1px solid {border_color}" if border_color else f"1px solid {GLASS_EDGE}"
    return f"QFrame {{ border: {border}; border-radius: 18px; background-color: {GLASS_BG}; }}"


MAXIMIZE_BTN_STYLE = f"""
QPushButton {{
    border: none;
    border-radius: 6px;
    background-color: transparent;
    color: {ACCENT};
    font-size: 13px;
    padding: 1px 5px;
}}
QPushButton:hover {{
    color: {ACCENT_SOFT};
    background-color: #EAF2FF;
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
    border: 1.5px dashed {ACCENT};
    border-radius: 16px;
    background-color: #F3F8FF;
}}
"""

DROP_ZONE_LOADED_STYLE = f"""
QFrame {{
    border: 1.5px solid {ACCENT};
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
    line.setStyleSheet(f"QFrame {{ background-color: {GLASS_EDGE}; border: none; }}")
    return line


def _hex_points(cx: float, cy: float, radius: float) -> QPolygonF:
    pts = QPolygonF()
    for i in range(6):
        angle = math.radians(60 * i - 30)
        pts.append(QPointF(cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return pts


def _paint_hive_chrome(painter: QPainter, rect, color: QColor) -> None:
    """Card chrome hook — intentionally empty in the minimal theme."""
    return
    painter.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(QRectF(rect), 18, 18)
    sheen = QLinearGradient(
        QPointF(rect.left(), rect.top()),
        QPointF(rect.left() + rect.width() * 0.5, rect.top() + rect.height() * 0.5),
    )
    sheen.setColorAt(0.0, QColor(255, 255, 255, 14))
    sheen.setColorAt(1.0, QColor(255, 255, 255, 0))
    painter.fillPath(path, sheen)

def _paint_sparse_hives(painter: QPainter, rect, seeds) -> None:
    """A few irregular hex marks — not a filled honeycomb grid."""
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(Qt.NoBrush)
    for cx, cy, radius, alpha, sides in seeds:
        x = rect.left() + cx * rect.width()
        y = rect.top() + cy * rect.height()
        color = QColor(ACCENT)
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
    color = QColor(ACCENT)
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


def _clear_icon() -> QIcon:
    """Trash-can glyph for the Clear button, drawn so it needs no asset file."""
    size = 64
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor("#f5c6ee"), 4.5)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawLine(QPointF(12, 18), QPointF(52, 18))                 # lid
    p.drawRoundedRect(QRectF(25, 9, 14, 9), 3, 3)                # handle
    body = QPolygonF([QPointF(17, 18), QPointF(47, 18), QPointF(44, 55), QPointF(20, 55)])
    p.drawPolygon(body)
    for x in (26, 32, 38):
        p.drawLine(QPointF(x, 26), QPointF(x, 47))
    p.end()
    return QIcon(pix)


class HiveField(QWidget):
    """Full-rectangle honeycomb behind a 2D viewer."""

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(BG_PANEL))
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
        _paint_hive_chrome(painter, self.rect().adjusted(1, 1, -1, -1), QColor(ACCENT))
        painter.end()


class HivePanel(QFrame):
    """Card with amber honeycomb corners layered above inner content."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hive_color = QColor(ACCENT)
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


def _card_head(title: str, meta: str = ""):
    """Mockup-style card header: title on the left, muted meta on the right."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    t = QLabel(title)
    t.setStyleSheet(
        f"border: none; background: transparent; color: {TEXT_LIGHT}; font-size: 12px; font-weight: 600;"
    )
    row.addWidget(t)
    row.addStretch(1)
    m = QLabel(meta)
    m.setStyleSheet(f"border: none; background: transparent; color: {TEXT_MUTED}; font-size: 11px;")
    row.addWidget(m)
    return row, t, m


def _card_qss(name: str) -> str:
    return (
        f"QFrame#{name} {{ background-color: {BG_PANEL}; border: 1px solid {GLASS_EDGE}; border-radius: 18px; }}"
        f"QFrame#{name} QLabel {{ border: none; background: transparent; }}"
    )


class _RoundedHost(QWidget):
    """Clips its children to a rounded rectangle (dark scan viewport corners)."""

    def __init__(self, radius: float = 12.0, parent=None):
        super().__init__(parent)
        self._radius = radius

    def resizeEvent(self, event):
        super().resizeEvent(event)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), self._radius, self._radius)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))


class _SegmentedToggle(QFrame):
    """White pill with exclusive options, e.g. All / Tumour / Brain."""

    changed = Signal(str)

    def __init__(self, options, parent=None):
        super().__init__(parent)
        self.setObjectName("segToggle")
        self.setStyleSheet(
            f"QFrame#segToggle {{ background: #ffffff; border: 1px solid {GLASS_EDGE}; border-radius: 16px; }}"
            f"QFrame#segToggle QPushButton {{ border: none; background: transparent; color: {TEXT_MUTED}; "
            "padding: 0 12px; border-radius: 13px; font-size: 12px; }"
            f"QFrame#segToggle QPushButton:hover {{ color: {TEXT_LIGHT}; }}"
            f"QFrame#segToggle QPushButton:checked {{ background: #EAF2FF; color: {ACCENT}; font-weight: 600; }}"
        )
        self.setFixedHeight(32)
        row = QHBoxLayout(self)
        row.setContentsMargins(3, 3, 3, 3)
        row.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for i, text in enumerate(options):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setFixedHeight(26)
            b.setCursor(Qt.PointingHandCursor)
            b.setFocusPolicy(Qt.NoFocus)
            b.setChecked(i == 0)
            b.clicked.connect(lambda _=False, t=text: self.changed.emit(t))
            self._group.addButton(b)
            row.addWidget(b)
            self._buttons = getattr(self, "_buttons", {})
            self._buttons[text] = b

    def select(self, text: str):
        """Check an option from code (e.g. a hand gesture) and emit `changed`."""
        b = getattr(self, "_buttons", {}).get(text)
        if b is not None:
            b.setChecked(True)
            self.changed.emit(text)


def _dot_icon(hex_color: str, alpha: float = 1.0) -> QIcon:
    """Small round colour swatch used as a legend-chip icon."""
    pix = QPixmap(18, 18)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    c = QColor(hex_color)
    c.setAlphaF(max(0.0, min(1.0, alpha)))
    p.setPen(Qt.NoPen)
    p.setBrush(c)
    p.drawEllipse(QRectF(1, 1, 16, 16))
    p.end()
    return QIcon(pix)


class _LegendPill(QFrame):
    """Rounded legend under the 3D brain: one chip per detected tumour.
    Clicking a chip focuses that tumour in 3D (the others fade back);
    clicking it again shows all of them."""

    focus_changed = Signal(int)   # tumour rank (1 = largest), 0 = show all

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("legendPill")
        self.setStyleSheet(
            f"QFrame#legendPill {{ background: #ffffff; border: 1px solid {GLASS_EDGE}; border-radius: 16px; }}"
            "QFrame#legendPill QLabel { border: none; background: transparent; }"
        )
        self.setFixedHeight(34)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(4, 3, 4, 3)
        self._row.setSpacing(4)
        self._chips = []          # [(button, rank, hex)]
        self._focus = 0
        self._hint = None
        self.set_items([])

    def set_items(self, items):
        while self._row.count():
            it = self._row.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        self._chips = []
        for rank, (name, hex_color) in enumerate(items, start=1):
            b = QPushButton(name)
            b.setIconSize(QSize(9, 9))
            b.setCursor(Qt.PointingHandCursor)
            b.setFocusPolicy(Qt.NoFocus)
            b.setFixedHeight(26)
            b.setToolTip(f"Focus {name} in 3D (click again to show all)")
            b.clicked.connect(lambda _=False, r=rank: self._on_chip(r))
            self._row.addWidget(b)
            self._chips.append((b, rank, hex_color))
        self._hint = QLabel("")
        self._hint.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; padding: 0 8px 0 2px;")
        self._row.addWidget(self._hint)
        self.setVisible(bool(items))
        self._set_focus(0, emit=True)

    def _on_chip(self, rank: int):
        self._set_focus(0 if rank == self._focus else rank, emit=True)

    def focus_next(self):
        """Tumour 1 -> 2 -> ... -> all. Returns (new rank, number of tumours)."""
        n = len(self._chips)
        if n == 0:
            return 0, 0
        nxt = self._focus + 1 if self._focus < n else 0
        self._set_focus(nxt, emit=True)
        return nxt, n

    def clear_focus(self):
        self._set_focus(0, emit=True)

    def _set_focus(self, rank: int, emit: bool):
        self._focus = rank
        for b, r, hex_color in self._chips:
            c = QColor(hex_color)
            dimmed = bool(rank) and r != rank
            b.setIcon(_dot_icon(hex_color, 0.35 if dimmed else 1.0))
            if r == rank:
                style = (f"QPushButton {{ color: {c.darker(140).name()}; font-weight: 600; "
                         f"background: rgba({c.red()},{c.green()},{c.blue()},0.12); "
                         f"border: 1px solid rgba({c.red()},{c.green()},{c.blue()},0.45); "
                         "border-radius: 13px; padding: 0 10px; font-size: 12px; }")
            else:
                fg = "#A3AFC2" if dimmed else TEXT_LIGHT
                style = (f"QPushButton {{ color: {fg}; background: transparent; border: 1px solid transparent; "
                         "border-radius: 13px; padding: 0 10px; font-size: 12px; }"
                         "QPushButton:hover { background: #F3F7FD; }")
            b.setStyleSheet(style)
        if self._hint is not None:
            self._hint.setText("Click again to show all" if rank else "")
            self._hint.setVisible(bool(rank))
        if emit:
            self.focus_changed.emit(rank)


class _PresetChips(QFrame):
    """Front · Side · Top buttons that fly the 3D camera to a standard view."""

    chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("presetChips")
        self.setStyleSheet(
            f"QFrame#presetChips {{ background: #ffffff; border: 1px solid {GLASS_EDGE}; border-radius: 16px; }}"
            f"QFrame#presetChips QPushButton {{ border: none; background: transparent; color: {TEXT_MUTED}; "
            "padding: 0 11px; border-radius: 13px; font-size: 12px; }"
            f"QFrame#presetChips QPushButton:hover {{ color: {ACCENT}; background: #EAF2FF; }}"
            f"QFrame#presetChips QPushButton:pressed {{ color: #ffffff; background: {ACCENT}; }}"
        )
        self.setFixedHeight(32)
        row = QHBoxLayout(self)
        row.setContentsMargins(3, 3, 3, 3)
        row.setSpacing(0)
        for label, key, tip in (("Front", "front", "View from the front (anterior)"),
                                ("Side", "side", "View from the patient's left side"),
                                ("Top", "top", "View from above (superior)")):
            b = QPushButton(label)
            b.setFixedHeight(26)
            b.setCursor(Qt.PointingHandCursor)
            b.setFocusPolicy(Qt.NoFocus)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self.chosen.emit(k))
            row.addWidget(b)


class RootShell(QWidget):
    """Window background: soft off-white with two very faint blue washes.
    Object name keeps the global stylesheet off children."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rootShell")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor(BG_PAGE))
        painter.setPen(Qt.NoPen)
        side = float(max(w, h))
        for fx, fy, rad, alpha in ((0.08, 0.0, 0.45, 26), (0.95, 1.0, 0.40, 20)):
            glow = QRadialGradient(QPointF(fx * w, fy * h), rad * side)
            c = QColor(ACCENT)
            c.setAlpha(alpha)
            glow.setColorAt(0.0, c)
            c2 = QColor(ACCENT)
            c2.setAlpha(0)
            glow.setColorAt(1.0, c2)
            painter.setBrush(glow)
            painter.drawEllipse(QPointF(fx * w, fy * h), rad * side, rad * side)
        painter.end()


def _island_icon(kind: str, color: str = "#C9D6EE") -> QIcon:
    """Thin line icons for the island buttons, drawn so they need no assets."""
    size = 48
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color), 3.6)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    if kind == "open":
        path = QPainterPath()
        path.moveTo(7, 14); path.lineTo(19, 14); path.lineTo(23, 18); path.lineTo(41, 18)
        path.lineTo(41, 37); path.lineTo(7, 37); path.closeSubpath()
        p.drawPath(path)
    elif kind == "run":
        p.setBrush(QColor(color))
        p.drawPolygon(QPolygonF([QPointF(16, 11), QPointF(37, 24), QPointF(16, 37)]))
    elif kind == "clear":
        p.drawLine(QPointF(10, 14), QPointF(38, 14))
        p.drawLine(QPointF(20, 9), QPointF(28, 9))
        p.drawPolygon(QPolygonF([QPointF(14, 14), QPointF(34, 14), QPointF(32, 39), QPointF(16, 39)]))
    elif kind == "refine":
        p.drawLine(QPointF(12, 36), QPointF(30, 18))
        p.drawLine(QPointF(34, 8), QPointF(34, 14))
        p.drawLine(QPointF(31, 11), QPointF(37, 11))
        p.drawLine(QPointF(40, 22), QPointF(40, 26))
        p.drawLine(QPointF(38, 24), QPointF(42, 24))
    elif kind == "check":
        pen.setWidthF(5.0)
        p.setPen(pen)
        p.drawPolyline(QPolygonF([QPointF(13, 25), QPointF(21, 33), QPointF(36, 16)]))
    elif kind == "gesture":
        p.drawRoundedRect(QRectF(13, 24, 22, 16), 7, 7)          # palm
        for x in (16, 21, 26, 31):
            p.drawLine(QPointF(x, 24), QPointF(x, 11))            # fingers
        p.drawLine(QPointF(13, 31), QPointF(8, 24))               # thumb
    p.end()
    return QIcon(pix)


class DynamicIsland(QFrame):
    """iPhone-style pill at the top of the window.

    idle  -> case name · Open · Segment · Clear
    busy  -> spinner · caption · progress bar (pill stretches wider)
    done  -> green check · message, then springs back to idle
    The pill animates its width with a slight overshoot, like the real thing."""

    open_clicked = Signal()
    refine_clicked = Signal()
    sam_plane_chosen = Signal(str)
    sam_action = Signal(str)
    gesture_toggled = Signal(bool)

    BTN_STYLE = (
        "QPushButton {{ color: {fg}; background: {bg}; border: none; border-radius: 17px; "
        "padding: 0 14px; font-size: 12px; font-weight: 600; }}"
        "QPushButton:hover {{ background: {hover}; color: #ffffff; }}"
        "QPushButton:pressed {{ background: {press}; }}"
        "QPushButton:disabled {{ color: #4A5877; background: {dis}; }}"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("island")
        self.setFixedHeight(46)
        self.setStyleSheet(
            f"QFrame#island {{ background-color: {ISLAND_BG}; border-radius: 23px; }}"
            "QFrame#island QLabel { background: transparent; border: none; }"
        )
        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)

        # ---- idle page ----
        idle = QWidget()
        idle.setStyleSheet("background: transparent;")
        row = QHBoxLayout(idle)
        row.setContentsMargins(16, 6, 6, 6)
        row.setSpacing(4)
        self._dot = QLabel()
        self._dot.setFixedSize(7, 7)
        row.addWidget(self._dot, 0, Qt.AlignVCenter)
        row.addSpacing(4)
        self.case_label = QLabel("No case loaded")
        self.case_label.setStyleSheet(f"color: {ISLAND_TEXT}; font-size: 12px; font-weight: 500;")
        row.addWidget(self.case_label)
        row.addSpacing(10)
        sep = QFrame()
        sep.setFixedSize(1, 18)
        sep.setStyleSheet("background: rgba(255,255,255,0.14); border: none;")
        row.addWidget(sep, 0, Qt.AlignVCenter)
        row.addSpacing(4)

        ghost = self.BTN_STYLE.format(fg=ISLAND_TEXT, bg="transparent", hover="rgba(255,255,255,0.10)",
                                      press="rgba(255,255,255,0.16)", dis="transparent")
        primary = self.BTN_STYLE.format(fg="#ffffff", bg=ACCENT, hover="#2F7CF6",
                                        press=ACCENT_DEEP, dis="rgba(255,255,255,0.06)")
        self.open_btn = self._button("Open", "open", ghost, "Open a patient folder")
        self.run_btn = self._button("Segment", "run", primary, "Run segmentation", icon_color="#ffffff")
        self.refine_btn = self._button("Refine", "refine", ghost, "Refine the mask with MedSAM2")
        self.clear_btn = self._button("Clear", "clear", ghost, "Clear the 2D and 3D views")
        # Gesture toggle: OFF at startup. The webcam only starts when clicked.
        self.gesture_btn = self._button(
            "Gesture", "gesture",
            ghost + f"QPushButton:checked {{ background: {ACCENT}; color: #ffffff; }}",
            "Control the 3D brain with your hand (webcam)",
        )
        self.gesture_btn.setCheckable(True)
        self.gesture_btn.setChecked(False)
        self.gesture_btn.toggled.connect(self.gesture_toggled.emit)
        for btn in (self.open_btn, self.run_btn, self.refine_btn, self.clear_btn, self.gesture_btn):
            row.addWidget(btn)
        self.open_btn.clicked.connect(self.open_clicked.emit)
        self.refine_btn.clicked.connect(self.refine_clicked.emit)
        self.refine_btn.setEnabled(False)
        self._stack.addWidget(idle)
        self._ghost_style = ghost
        self._primary_style = primary
        self.mode = "idle"

        # ---- busy page ----
        busy = QWidget()
        busy.setStyleSheet("background: transparent;")
        brow = QHBoxLayout(busy)
        brow.setContentsMargins(16, 0, 20, 0)
        brow.setSpacing(12)
        self.spinner = ArcSpinner(diameter=18, color="#5AA2FF")
        brow.addWidget(self.spinner, 0, Qt.AlignVCenter)
        self.busy_label = QLabel("Working")
        self.busy_label.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: 500;")
        brow.addWidget(self.busy_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.setStyleSheet(
            "QProgressBar { border: none; border-radius: 2px; background: rgba(255,255,255,0.12); "
            "min-height: 4px; max-height: 4px; }"
            "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #1F6FEB, stop:1 #5AA2FF); border-radius: 2px; }"
        )
        brow.addWidget(self.progress, 1)
        self._stack.addWidget(busy)

        # ---- done page ----
        done = QWidget()
        done.setStyleSheet("background: transparent;")
        drow = QHBoxLayout(done)
        drow.setContentsMargins(14, 0, 20, 0)
        drow.setSpacing(10)
        check = QLabel()
        check.setFixedSize(20, 20)
        check.setAlignment(Qt.AlignCenter)
        check.setStyleSheet("background: #30A46C; border-radius: 10px;")
        check.setPixmap(_island_icon("check", "#ffffff").pixmap(12, 12))
        drow.addWidget(check, 0, Qt.AlignVCenter)
        self.done_label = QLabel("Done")
        self.done_label.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: 500;")
        drow.addWidget(self.done_label, 1)
        self._stack.addWidget(done)

        # ---- MedSAM2 page (index 3) ----
        sam = QWidget()
        sam.setStyleSheet("background: transparent;")
        srow = QHBoxLayout(sam)
        srow.setContentsMargins(16, 6, 6, 6)
        srow.setSpacing(4)
        sam_title = QLabel("MedSAM2")
        sam_title.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: 600;")
        srow.addWidget(sam_title)
        srow.addSpacing(8)
        chip = (
            f"QPushButton {{ color: {ISLAND_TEXT}; background: rgba(255,255,255,0.06); border: none; "
            "border-radius: 13px; padding: 0 10px; font-size: 11px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.12); color: #ffffff; }"
            f"QPushButton:checked {{ background: {ACCENT}; color: #ffffff; font-weight: 600; }}"
            "QPushButton:disabled { color: #4A5877; }"
        )
        self._plane_group = QButtonGroup(self)
        self._plane_group.setExclusive(True)
        self._plane_btns = {}
        for name in ("Sagittal", "Axial", "Coronal"):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setFixedHeight(26)
            b.setCursor(Qt.PointingHandCursor)
            b.setFocusPolicy(Qt.NoFocus)
            b.setStyleSheet(chip)
            b.setToolTip(f"Place prompts on the {name.lower()} slice")
            b.clicked.connect(lambda _=False, n=name: self.sam_plane_chosen.emit(n))
            self._plane_group.addButton(b)
            self._plane_btns[name] = b
            srow.addWidget(b)
        srow.addSpacing(10)
        self.sam_hint = QLabel("")
        self.sam_hint.setStyleSheet("color: #8FA0BF; font-size: 11px;")
        srow.addWidget(self.sam_hint)
        srow.addSpacing(8)
        self.sam_btns = {}
        for key, text, style, tip in (
            ("undo", "Undo", ghost, "Remove the last prompt"),
            ("clear", "Clear", ghost, "Remove all prompts"),
            ("preview", "Preview", primary, "Run MedSAM2 with these prompts"),
            ("accept", "Accept", primary, "Keep the refined mask and save it"),
            ("discard", "Discard", ghost, "Throw the preview away"),
            ("done", "Done", ghost, "Leave MedSAM2 editing"),
        ):
            b = QPushButton(text)
            b.setFixedHeight(34)
            b.setCursor(Qt.PointingHandCursor)
            b.setFocusPolicy(Qt.NoFocus)
            b.setToolTip(tip)
            b.setStyleSheet(style)
            b.clicked.connect(lambda _=False, k=key: self.sam_action.emit(k))
            self.sam_btns[key] = b
            srow.addWidget(b)
        self.sam_btns["accept"].hide()
        self.sam_btns["discard"].hide()
        self._stack.addWidget(sam)

        from PySide6.QtCore import QVariantAnimation, QEasingCurve
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(520)
        self._anim.setEasingCurve(QEasingCurve.OutBack)
        self._anim.valueChanged.connect(self._apply_width)
        self._done_timer = QTimer(self)
        self._done_timer.setSingleShot(True)
        self._done_timer.timeout.connect(self.show_idle)
        self.set_case(None)
        self.setFixedWidth(self._idle_width())

    def _button(self, text, icon_kind, style, tip, icon_color=ISLAND_TEXT):
        btn = QPushButton(" " + text)
        btn.setIcon(_island_icon(icon_kind, icon_color))
        btn.setIconSize(QSize(15, 15))
        btn.setFixedHeight(34)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFocusPolicy(Qt.NoFocus)
        btn.setToolTip(tip)
        btn.setStyleSheet(style)
        return btn

    def _page_width(self, index: int) -> int:
        page = self._stack.widget(index)
        page.layout().invalidate()
        page.layout().activate()
        return page.sizeHint().width()

    def _idle_width(self) -> int:
        return max(300, self._page_width(0))

    def _home_index(self) -> int:
        return 3 if self.mode == "sam" else 0

    # ---- MedSAM2 mode ----
    def enter_sam(self, plane: str):
        self.mode = "sam"
        self.set_sam_plane(plane)
        self._done_timer.stop()
        self._stack.setCurrentIndex(3)
        self._animate_to(self._page_width(3))

    def exit_sam(self):
        was = self.mode
        self.mode = "idle"
        if was == "sam" and self._stack.currentIndex() == 3:
            self.show_idle()

    def set_sam_plane(self, plane: str):
        b = self._plane_btns.get(plane)
        if b is not None:
            b.setChecked(True)

    def update_sam(self, hint: str, enabled: dict, preview: bool):
        changed = self.sam_hint.text() != hint
        self.sam_hint.setText(hint)
        for key, b in self.sam_btns.items():
            want = (key in ("accept", "discard")) if preview else (key not in ("accept", "discard"))
            if b.isHidden() == want:
                b.setVisible(want)
                changed = True
            b.setEnabled(bool(enabled.get(key, True)))
        for b in self._plane_btns.values():
            b.setEnabled(not preview)
        if changed and self._stack.currentIndex() == 3:
            self._animate_to(self._page_width(3))

    def _apply_width(self, value):
        self.setFixedWidth(int(value))
        parent = self.parentWidget()
        if parent is not None and hasattr(parent, "place_island"):
            parent.place_island()

    def _animate_to(self, width: int):
        self._anim.stop()
        self._anim.setStartValue(self.width())
        self._anim.setEndValue(int(width))
        self._anim.start()

    def set_case(self, case_id: Optional[str]):
        loaded = bool(case_id)
        text = case_id if loaded else "No case loaded"
        if len(text) > 26:
            text = text[:24] + "…"
        self.case_label.setText(text)
        self.case_label.setToolTip(case_id or "")
        self._dot.setStyleSheet(
            f"background: {'#30A46C' if loaded else '#4A5877'}; border-radius: 3px;"
        )
        if self._stack.currentIndex() == 0:
            self._animate_to(self._idle_width())

    def show_idle(self):
        self._done_timer.stop()
        self.spinner.stop()
        home = self._home_index()
        self._stack.setCurrentIndex(home)
        self._animate_to(self._idle_width() if home == 0 else self._page_width(home))

    def show_busy(self, text: str = "Working"):
        self._done_timer.stop()
        self.busy_label.setText(text)
        self._stack.setCurrentIndex(1)
        self.spinner.start()
        self._animate_to(max(460, self._idle_width() + 20))

    def show_done(self, text: str, hold_ms: int = 2400):
        self.spinner.stop()
        self.done_label.setText(text)
        self._stack.setCurrentIndex(2)
        fm = self.done_label.fontMetrics()
        self._animate_to(max(260, fm.horizontalAdvance(text) + 90))
        self._done_timer.start(hold_ms)


class _IslandBusyProxy:
    """Stands in for the old left-column busy bar: show()/hide() drive the island."""

    def __init__(self, island: DynamicIsland):
        self._island = island

    def show(self, text: str = "Working"):
        self._island.show_busy(text)

    def hide(self):
        self._island.show_idle()


class CameraBubble(QWidget):
    """Small mirrored webcam preview that pops out beside the Dynamic Island
    while gesture control is on, so visitors can see themselves and their hand.
    Springs open from the island's edge (same feel as the island) and tucks
    back in when Gesture is switched off. Purely visual: clicks pass through."""

    W, H = 128, 86          # visible bubble size
    RADIUS = 16
    MARGIN = 8              # room for the spring overshoot

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(self.W + 2 * self.MARGIN, self.H + 2 * self.MARGIN)
        self._pix: Optional[QPixmap] = None
        self._state = "Starting camera…"
        self._progress = 0.0        # hold-to-confirm ring, 0-1
        self._progress_label = ""
        self._done_until = 0.0      # show the green tick until this time
        self._t = 0.0       # 0 = tucked away, 1 = fully open
        from PySide6.QtCore import QVariantAnimation, QEasingCurve
        self._ease_in = QEasingCurve.OutBack
        self._ease_out = QEasingCurve.InCubic
        self._anim = QVariantAnimation(self)
        self._anim.valueChanged.connect(self._on_anim)
        self._anim.finished.connect(self._on_anim_done)
        self.hide()

    # ---- public API ----
    def pop_in(self):
        self._pix = None
        self._state = "Starting camera…"
        self.show()
        self.raise_()
        self._animate(1.0, 460, self._ease_in)

    def pop_out(self):
        if not self.isVisible():
            return
        self._animate(0.0, 220, self._ease_out)

    def set_frame(self, img: QImage):
        if self.isVisible():
            self._pix = QPixmap.fromImage(img)
            self.update()

    def set_state(self, text: str):
        self._state = text
        self.update()

    def set_progress(self, value: float, label: str = ""):
        """Hold-to-confirm ring in the top-right corner (0 hides it)."""
        self._progress = max(0.0, min(1.0, float(value)))
        self._progress_label = label if self._progress > 0 else ""
        self.update()

    def flash_done(self, seconds: float = 1.0):
        """Green tick + "Cooling down" after a gesture command fires."""
        self._progress, self._progress_label = 0.0, ""
        self._done_until = time.monotonic() + seconds
        self.update()
        QTimer.singleShot(int(seconds * 1000) + 30, self.update)

    # ---- animation ----
    def _animate(self, target: float, ms: int, curve):
        self._anim.stop()
        self._anim.setDuration(ms)
        self._anim.setEasingCurve(curve)
        self._anim.setStartValue(float(self._t))
        self._anim.setEndValue(float(target))
        self._anim.start()

    def _on_anim(self, value):
        self._t = float(value)
        self.update()

    def _on_anim_done(self):
        if self._t <= 0.001:
            self._t = 0.0
            self._pix = None
            self.hide()

    # ---- painting ----
    def paintEvent(self, event):
        t = self._t
        if t <= 0.001:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        m = self.MARGIN
        p.setOpacity(max(0.0, min(1.0, t)))
        # Grow out of the island's side: scale around the bubble's left-middle.
        p.translate(m, m + self.H / 2)
        p.scale(t, t)
        p.translate(-m, -(m + self.H / 2))

        outer = QRectF(m, m, self.W, self.H)
        shell = QPainterPath()
        shell.addRoundedRect(outer, self.RADIUS, self.RADIUS)
        p.fillPath(shell, QColor(ISLAND_BG))

        inner = outer.adjusted(3, 3, -3, -3)
        clip = QPainterPath()
        clip.addRoundedRect(inner, self.RADIUS - 3, self.RADIUS - 3)
        p.save()
        p.setClipPath(clip)
        if self._pix is not None and not self._pix.isNull():
            scaled = self._pix.scaled(inner.size().toSize(), Qt.KeepAspectRatioByExpanding,
                                      Qt.SmoothTransformation)
            p.drawPixmap(
                int(inner.center().x() - scaled.width() / 2),
                int(inner.center().y() - scaled.height() / 2),
                scaled,
            )
        else:
            p.fillRect(inner, QColor("#111A30"))
        # Soft dark fade at the bottom so the status text stays readable.
        fade = QLinearGradient(0, inner.bottom() - 30, 0, inner.bottom())
        fade.setColorAt(0.0, QColor(10, 18, 36, 0))
        fade.setColorAt(1.0, QColor(10, 18, 36, 190))
        p.fillRect(QRectF(inner.left(), inner.bottom() - 30, inner.width(), 30), fade)
        p.restore()

        # "Live" dot, top-left.
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#FF453A"))
        p.drawEllipse(QPointF(inner.left() + 10, inner.top() + 10), 3.5, 3.5)

        done = time.monotonic() < self._done_until
        corner = QPointF(inner.right() - 13, inner.top() + 13)
        if self._progress > 0:
            # Hold-to-confirm ring: grey track, blue arc filling clockwise.
            ring = QRectF(corner.x() - 8, corner.y() - 8, 16, 16)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(10, 18, 36, 150))
            p.drawEllipse(corner, 10.5, 10.5)
            p.setPen(QPen(QColor(255, 255, 255, 60), 2.6))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(ring)
            arc = QPen(QColor("#5AA2FF"), 2.6)
            arc.setCapStyle(Qt.RoundCap)
            p.setPen(arc)
            p.drawArc(ring, 90 * 16, int(-self._progress * 360 * 16))
        elif done:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#30A46C"))
            p.drawEllipse(corner, 8, 8)
            tick = QPen(QColor("#FFFFFF"), 1.8)
            tick.setCapStyle(Qt.RoundCap)
            tick.setJoinStyle(Qt.RoundJoin)
            p.setPen(tick)
            p.drawPolyline(QPolygonF([QPointF(corner.x() - 4, corner.y() + 0.2),
                                      QPointF(corner.x() - 1.4, corner.y() + 2.6),
                                      QPointF(corner.x() + 3.8, corner.y() - 2.6)]))

        # Current gesture (or the pending command), bottom-left.
        if self._progress > 0 and self._progress_label:
            text = self._progress_label
        elif done:
            text = "Cooling down\u2026"
        else:
            text = self._state
        p.setPen(QColor("#FFFFFF"))
        font = QFont("Segoe UI", 8)
        font.setWeight(QFont.DemiBold)
        p.setFont(font)
        p.drawText(inner.adjusted(8, 0, -6, -5), int(Qt.AlignLeft | Qt.AlignBottom),
                   p.fontMetrics().elidedText(text, Qt.ElideRight, int(inner.width() - 14)))
        p.end()


# Gesture guide card: icon strokes in a 40x40 box (same drawings as the mockup).
_GESTURE_ICONS = {
    "palm": [("rect", 13, 20, 19, 13, 5), ("lines", (15, 20, 15, 8), (20, 20, 20, 6), (25, 20, 25, 7),
                                           (30, 20, 30, 9), (13, 28, 7, 21))],
    "ok": [("rect", 13, 20, 19, 13, 5), ("circle", 13, 16, 3.6), ("lines", (20, 20, 20, 6), (25, 20, 25, 7),
                                                                    (30, 20, 30, 9))],
    "fist": [("rect", 13, 18, 19, 15, 6), ("lines", (15, 18, 15, 15), (20, 18, 20, 14), (25, 18, 25, 14),
                                           (30, 18, 30, 15), (13, 27, 17, 24))],
    "point": [("rect", 13, 20, 19, 13, 5), ("lines", (15, 20, 15, 5), (20, 20, 20, 16), (25, 20, 25, 16),
                                            (30, 20, 30, 16), (13, 27, 17, 24))],
    "peace": [("rect", 13, 20, 19, 13, 5), ("lines", (15, 20, 12, 6), (21, 20, 23, 6), (25, 20, 25, 16),
                                            (30, 20, 30, 16), (13, 27, 17, 24))],
    "thumbs": [("rect", 16, 19, 16, 15, 6), ("lines", (17, 19, 17, 6), (22, 22, 29, 22), (22, 27, 29, 27))],
    "three": [("rect", 13, 20, 19, 13, 5), ("lines", (15, 20, 15, 8), (20, 20, 20, 6), (25, 20, 25, 7),
                                            (30, 20, 30, 16), (13, 27, 17, 24))],
    "reset": [("dashcircle", 22, 20, 17), ("rect", 14, 21, 16, 11, 4),
              ("lines", (16, 21, 16, 11), (20, 21, 20, 9), (24, 21, 24, 10), (28, 21, 28, 12))],
}
_GESTURE_GUIDE_ITEMS = (            # icon, title, how, new?
    ("palm", "Rotate", "Open palm, move", False),
    ("ok", "Zoom", "OK sign, up / down", False),
    ("fist", "Hold", "Fist", False),
    ("point", "Slice", "Point; hold = next plane", True),
    ("peace", "View mode", "Peace sign, hold", True),
    ("thumbs", "Next view", "Thumbs up, hold", True),
    ("three", "Next tumour", "Three fingers, hold", True),
    ("reset", "Reset", "Palm still, 1.5 s", True),
)


class GestureGuideCard(QWidget):
    """Dark card listing every hand gesture. Slides out under the camera
    bubble the first time Gesture is switched on and hides itself after
    SHOW_SECONDS. Purely visual: clicks pass through.

    It is a small frameless floating window rather than a child widget: the
    3D views are web views, which Windows draws as native windows that an
    ordinary child widget cannot sit on top of (it would appear behind them)."""

    W, H = 372, 300
    MARGIN = 8
    SHOW_SECONDS = 30

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus
                         | Qt.WindowTransparentForInput | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFixedSize(self.W + 2 * self.MARGIN, self.H + 2 * self.MARGIN)
        self._t = 0.0
        self._left = self.SHOW_SECONDS
        from PySide6.QtCore import QVariantAnimation, QEasingCurve
        self._anim = QVariantAnimation(self)
        self._anim.valueChanged.connect(self._on_anim)
        self._anim.finished.connect(self._on_anim_done)
        self._ease_in, self._ease_out = QEasingCurve.OutCubic, QEasingCurve.InCubic
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._countdown)
        self.hide()

    def pop_in(self):
        self._left = self.SHOW_SECONDS
        self.show()
        self.raise_()
        self._animate(1.0, 320, self._ease_in)
        self._tick.start()

    def pop_out(self):
        self._tick.stop()
        if self.isVisible():
            self._animate(0.0, 200, self._ease_out)

    def _countdown(self):
        self._left -= 1
        if self._left <= 0:
            self.pop_out()
        self.update()

    def _animate(self, target, ms, curve):
        self._anim.stop()
        self._anim.setDuration(ms)
        self._anim.setEasingCurve(curve)
        self._anim.setStartValue(float(self._t))
        self._anim.setEndValue(float(target))
        self._anim.start()

    def _on_anim(self, value):
        self._t = float(value)
        self.update()

    def _on_anim_done(self):
        if self._t <= 0.001:
            self._t = 0.0
            self.hide()

    @staticmethod
    def _draw_icon(p: QPainter, key: str, box: QRectF, color: QColor):
        s = box.width() / 40.0

        def pt(x, y):
            return QPointF(box.left() + x * s, box.top() + y * s)

        pen = QPen(color, 2.4 * s)
        pen.setCapStyle(Qt.RoundCap)
        p.setBrush(Qt.NoBrush)
        for shape in _GESTURE_ICONS[key]:
            kind = shape[0]
            if kind == "rect":
                _, x, y, w, h, r = shape
                p.setPen(pen)
                p.drawRoundedRect(QRectF(pt(x, y), pt(x + w, y + h)), r * s, r * s)
            elif kind == "circle":
                _, cx, cy, r = shape
                p.setPen(pen)
                p.drawEllipse(pt(cx, cy), r * s, r * s)
            elif kind == "dashcircle":
                _, cx, cy, r = shape
                dash = QPen(color, 1.6 * s)
                dash.setStyle(Qt.DashLine)
                p.setPen(dash)
                p.drawEllipse(pt(cx, cy), r * s, r * s)
            else:
                p.setPen(pen)
                for x1, y1, x2, y2 in shape[1:]:
                    p.drawLine(pt(x1, y1), pt(x2, y2))

    def paintEvent(self, event):
        t = self._t
        if t <= 0.001:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setOpacity(max(0.0, min(1.0, t)))
        p.translate(0, (1.0 - t) * -10)            # slides down into place
        m = self.MARGIN
        card = QRectF(m, m, self.W, self.H)
        path = QPainterPath()
        path.addRoundedRect(card, 18, 18)
        p.fillPath(path, QColor(ISLAND_BG))

        p.setPen(QColor("#FFFFFF"))
        title = QFont("Segoe UI", 10)
        title.setWeight(QFont.DemiBold)
        p.setFont(title)
        head = QRectF(m + 14, m + 12, self.W - 28, 20)
        p.drawText(head, int(Qt.AlignLeft | Qt.AlignVCenter), "Hand gestures")
        small = QFont("Segoe UI", 8)
        p.setFont(small)
        p.setPen(QColor("#8FA0BF"))
        p.drawText(head, int(Qt.AlignRight | Qt.AlignVCenter), f"Hides in {max(0, self._left)} s")

        col_w = (self.W - 28 - 8) / 2.0
        row_h, gap = 46.0, 8.0
        name_f = QFont("Segoe UI", 9)
        name_f.setWeight(QFont.DemiBold)
        how_f = QFont("Segoe UI", 8)
        for n, (icon, name, how, new) in enumerate(_GESTURE_GUIDE_ITEMS):
            col, row = n % 2, n // 2
            x = m + 14 + col * (col_w + 8)
            y = m + 42 + row * (row_h + gap)
            cell = QRectF(x, y, col_w, row_h)
            cp = QPainterPath()
            cp.addRoundedRect(cell, 11, 11)
            p.fillPath(cp, QColor(31, 111, 235, 56) if new else QColor(255, 255, 255, 13))
            self._draw_icon(p, icon, QRectF(x + 8, y + 8, 30, 30),
                            QColor("#FFFFFF") if new else QColor("#9CC7FF"))
            p.setPen(QColor("#FFFFFF"))
            p.setFont(name_f)
            p.drawText(QRectF(x + 46, y + 6, col_w - 50, 18), int(Qt.AlignLeft | Qt.AlignVCenter), name)
            p.setPen(QColor("#C9D6EE") if new else QColor("#8FA0BF"))
            p.setFont(how_f)
            p.drawText(QRectF(x + 46, y + 23, col_w - 50, 16), int(Qt.AlignLeft | Qt.AlignVCenter), how)
        p.setPen(QColor("#8FA0BF"))
        p.setFont(small)
        p.drawText(QRectF(m + 14, card.bottom() - 26, self.W - 28, 16), int(Qt.AlignLeft | Qt.AlignVCenter),
                   "Highlighted = new \u00b7 hold command poses until the ring fills")
        p.end()


class AppHeader(QWidget):
    """Minimal header: blue app mark + title on the left, the Dynamic Island
    centred, and model / device status on the right."""

    TITLE = "NeuroVista"
    SUBTITLE = "A.I. copilot for cancer detection and patient survival prediction"

    def __init__(self, island: DynamicIsland):
        super().__init__()
        self.setFixedHeight(72)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.island = island
        island.setParent(self)
        island.show()
        try:
            self._device = "CUDA" if torch.cuda.is_available() else "CPU"
        except Exception:
            self._device = "CPU"
        try:
            self._model_ok = Path(MODEL_PATH).is_file()
        except Exception:
            self._model_ok = False

    def place_island(self):
        self.island.move((self.width() - self.island.width()) // 2, (self.height() - self.island.height()) // 2)
        self.update()
        follow = getattr(self, "on_island_moved", None)   # keeps the camera bubble beside the island
        if follow is not None:
            follow()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.place_island()

    def showEvent(self, event):
        super().showEvent(event)
        self.place_island()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        rect = self.rect()
        h = rect.height()

        # App mark: small rounded blue tile with a "neuron" glyph.
        mark = QRectF(SPACING + 2, (h - 26) / 2, 26, 26)
        painter.setPen(Qt.NoPen)
        tile = QLinearGradient(mark.topLeft(), mark.bottomRight())
        tile.setColorAt(0.0, QColor(ACCENT))
        tile.setColorAt(1.0, QColor("#5AA2FF"))
        painter.setBrush(tile)
        painter.drawRoundedRect(mark, 8, 8)
        c = mark.center()
        painter.setPen(QPen(QColor(255, 255, 255, 240), 1.6))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(c, 3.6, 3.6)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 225))
        for ang in (45, 135, 225, 315):
            r = math.radians(ang)
            painter.drawEllipse(QPointF(c.x() + 7.4 * math.cos(r), c.y() + 7.4 * math.sin(r)), 1.4, 1.4)

        # Left text block must stop before the island.
        left_limit = max(60, self.island.x() - 16)
        x = int(mark.right() + 12)
        title_font = QFont("Segoe UI", 13, QFont.DemiBold)
        painter.setFont(title_font)
        tm = painter.fontMetrics()
        ty = int(h / 2 - 2)
        painter.setPen(QColor(TEXT_LIGHT))
        painter.drawText(x, ty, tm.elidedText(self.TITLE, Qt.ElideRight, left_limit - x))
        sub_font = QFont("Segoe UI", 8)
        painter.setFont(sub_font)
        sm = painter.fontMetrics()
        painter.setPen(QColor(TEXT_MUTED))
        painter.drawText(x, ty + sm.ascent() + 4, sm.elidedText(self.SUBTITLE, Qt.ElideRight, max(0, left_limit - x)))

        # Right: model + device status.
        status_font = QFont("Segoe UI", 9)
        painter.setFont(status_font)
        fm = painter.fontMetrics()
        model_txt = "Model ready" if self._model_ok else "No checkpoint"
        right = rect.width() - SPACING - 4
        dev_w = fm.horizontalAdvance(self._device)
        painter.setPen(QColor(TEXT_MUTED))
        baseline = int(h / 2 + fm.ascent() / 2 - 1)
        painter.drawText(right - dev_w, baseline, self._device)
        mx = right - dev_w - 16 - fm.horizontalAdvance(model_txt)
        if mx > self.island.geometry().right() + 12:
            painter.drawText(mx, baseline, model_txt)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#30A46C" if self._model_ok else "#F5A524"))
            painter.drawEllipse(QPointF(mx - 9, h / 2), 3.5, 3.5)
        painter.end()

class TechRule(QWidget):
    """Horizontal circuit separator used above the UniKL credit."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(10)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        y = self.height() / 2
        w = self.width()
        line = QLinearGradient(0, 0, w, 0)
        line.setColorAt(0.0, QColor(31, 111, 235, 0))
        line.setColorAt(0.5, QColor(31, 111, 235, 90))
        line.setColorAt(1.0, QColor(31, 111, 235, 0))
        painter.setPen(QPen(line, 1))
        painter.drawLine(QPointF(8, y), QPointF(w - 8, y))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(ACCENT_SOFT))
        painter.drawEllipse(QPointF(w / 2, y), 2.2, 2.2)
        painter.end()

class SquadCredit(QWidget):
    """Developed-by strip: circuit rule, UniKL logo, A.I. Squad remark."""

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 8, 2, 4)
        layout.setSpacing(6)
        layout.addWidget(TechRule())

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        logo = QLabel()
        logo.setStyleSheet(f"background: {ISLAND_BG}; border: none; border-radius: 8px; padding: 4px 6px;")
        pix = _tight_pixmap(_UNIKL_LOGO_PATH)
        if not pix.isNull():
            logo.setPixmap(pix.scaled(72, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        row.addWidget(logo, 0, Qt.AlignVCenter)

        copy = QLabel("Developed by\nUniKL MIIT A.I. Squad")
        copy.setWordWrap(True)
        copy.setStyleSheet(
            f"border: none; background: transparent; color: {TEXT_MUTED}; "
            f"font-size: 8px; font-weight: 600; letter-spacing: 0.4px;"
        )
        row.addWidget(copy, 1)
        layout.addLayout(row)


class TechLabel(QLabel):
    """Yellowish-orange HUD label for the 3D panels."""

    def __init__(self, text: str):
        super().__init__(text.upper())
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            f"border: none; color: {TEXT_MUTED}; font-size: 10px; "
            f"font-weight: 700; letter-spacing: 1.6px; background: transparent;"
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
            f"border: none; color: {ACCENT}; font-size: 16px; "
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
        color = TEXT_LIGHT if filled else TEXT_DIM
        self.value.setStyleSheet(
            f"border: none; color: {color}; font-size: 13px; font-weight: 600; "
            f"background: transparent; padding: 0;"
        )

    def set_value(self, text: str, filled: bool) -> None:
        self.value.setText(text)
        self.value.setToolTip(text if filled else "")
        self._apply_value_style(filled)


REPORT_MAX_TUMOR_ROWS = 5   # rows shown in the Report card; the rest are summarised


class _TumorRow(QWidget):
    """One detected tumor on a single line: ● Red   4.21 cm³   94.3%,
    with a thin confidence bar in the tumor's colour underneath."""

    def __init__(self, color_name: str, color_hex: str,
                 confidence: Optional[float], volume_cm3: float):
        super().__init__()
        self.setStyleSheet("background: transparent; border: none;")
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(6)
        swatch = QLabel()
        swatch.setFixedSize(8, 8)
        swatch.setStyleSheet(f"background-color: {color_hex}; border: none; border-radius: 4px;")
        line.addWidget(swatch, 0, Qt.AlignVCenter)

        name = QLabel(color_name)
        name.setStyleSheet(
            f"border: none; background: transparent; color: {TEXT_LIGHT}; "
            "font-size: 11px; font-weight: 600;"
        )
        line.addWidget(name)
        line.addStretch(1)

        vol = QLabel(f"{volume_cm3:.2f} cm\u00b3")
        vol.setStyleSheet(
            f"border: none; background: transparent; color: {TEXT_MUTED}; font-size: 10px;"
        )
        line.addWidget(vol)

        if confidence is None:
            conf_text, conf_color = "Refined", TEXT_MUTED
            tip = "Mask edited with MedSAM2 — no model confidence for this region"
        else:
            conf_text, conf_color = f"{confidence * 100:.1f}%", TEXT_LIGHT
            tip = "Mean model probability across this tumor's voxels"
        conf = QLabel(conf_text)
        conf.setToolTip(tip)
        conf.setMinimumWidth(40)
        conf.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        conf.setStyleSheet(
            f"border: none; background: transparent; color: {conf_color}; "
            "font-size: 11px; font-weight: 700;"
        )
        line.addWidget(conf)
        col.addLayout(line)

        if confidence is not None:
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(int(round(max(0.0, min(1.0, confidence)) * 1000)))
            bar.setTextVisible(False)
            bar.setFixedHeight(3)
            bar.setStyleSheet(
                "QProgressBar { border: none; border-radius: 1px; background-color: #EEF2F8; "
                "min-height: 3px; max-height: 3px; }"
                f"QProgressBar::chunk {{ background-color: {color_hex}; border-radius: 1px; }}"
            )
            col.addWidget(bar)


class PatientRecordPanel(HivePanel):
    """Report card: total tumor volume + one row per detected tumor, listed
    in COMPONENT_PALETTE order (largest first), each with its own model
    confidence and volume."""

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(
            f"QFrame {{ border: 1px solid {GLASS_EDGE}; border-radius: 18px; background-color: {BG_PANEL}; }}"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setStyleSheet(
            "QFrame { background: transparent; border: none; }"
        )
        self.setObjectName("reportCard")
        self.setStyleSheet(
            f"QFrame#reportCard {{ border: 1px solid {GLASS_EDGE}; border-radius: 18px; background-color: {BG_PANEL}; }}"
        )
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 2)
        head_row, _, self.report_meta = _card_head("Report Summary", "")
        header_layout.addLayout(head_row)
        layout.addWidget(header)

        # Pinned summary tiles — stay put while the tumor list scrolls.
        summary_wrap = QWidget()
        summary_wrap.setStyleSheet("background: transparent;")
        summary_layout = QVBoxLayout(summary_wrap)
        summary_layout.setContentsMargins(12, 8, 12, 6)
        summary_layout.setSpacing(0)
        self.summary = ReportSummaryPanel()
        summary_layout.addWidget(self.summary)
        layout.addWidget(summary_wrap)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"QFrame {{ background-color: {BORDER_DIM}; border: none; }}")
        divider_wrap = QHBoxLayout()
        divider_wrap.setContentsMargins(12, 0, 12, 0)
        divider_wrap.addWidget(divider)
        layout.addLayout(divider_wrap)

        body = QWidget()
        body.setStyleSheet("background: transparent;")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(12, 6, 12, 8)
        body_layout.setSpacing(6)

        self.tumor_caption = QLabel("Detected tumors")
        self.tumor_caption.setStyleSheet(
            f"border: none; color: {TEXT_MUTED}; font-size: 10px; "
            f"font-weight: 500; letter-spacing: 0.3px; background: transparent;"
        )
        body_layout.addWidget(self.tumor_caption)

        self._tumor_list = QVBoxLayout()
        self._tumor_list.setContentsMargins(0, 0, 0, 0)
        self._tumor_list.setSpacing(5)
        body_layout.addLayout(self._tumor_list)
        body_layout.addStretch(1)

        # No scroll area: the list is capped at REPORT_MAX_TUMOR_ROWS instead.
        layout.addWidget(body, 1)
        self.reset()

    def _clear_tumor_rows(self) -> None:
        while self._tumor_list.count():
            item = self._tumor_list.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()

    def _set_placeholder(self, text: str) -> None:
        placeholder = QLabel(text)
        placeholder.setWordWrap(True)
        placeholder.setStyleSheet(
            "border: none; background: transparent; color: " + TEXT_DIM + "; "
            "font-size: 11px; font-weight: 600;"
        )
        self._tumor_list.addWidget(placeholder)

    def reset(self) -> None:
        """Empty state — a new case is loaded but not yet segmented."""
        self.tumor_caption.setText("Detected tumors")
        self._clear_tumor_rows()
        self._set_placeholder("—")

    def set_tumors(self, components: list, total_cm3: Optional[float]) -> None:
        """components: entries from MainWindow._label_tumor_components, already
        sorted largest-first, so the rows follow the palette colour order."""
        if total_cm3 is None:
            self.reset()
            return
        self._clear_tumor_rows()
        self.tumor_caption.setText("Detected tumors")
        if not components:
            self._set_placeholder("No tumor detected")
            return
        for comp in components[:REPORT_MAX_TUMOR_ROWS]:
            self._tumor_list.addWidget(_TumorRow(
                f"Tumour {comp['rank']}", comp["color_hex"],
                comp.get("confidence"), comp["volume_cm3"],
            ))
        hidden = len(components) - REPORT_MAX_TUMOR_ROWS
        if hidden > 0:
            more = QLabel(f"+{hidden} smaller tumor{'s' if hidden != 1 else ''}")
            more.setStyleSheet(
                f"border: none; background: transparent; color: {TEXT_MUTED}; font-size: 10px;"
            )
            self._tumor_list.addWidget(more)


class PatientInfoPanel(QFrame):
    """Patient Record card: case ID, voxel size, dimensions and which of the
    four MRI modalities were found in the folder (missing ones show greyed)."""

    MODALITY_TAGS = (
        ("T1n", r"[_-](t1n|t1)[_.]"),
        ("T1c", r"[_-](t1c|t1ce|t1gd)[_.]"),
        ("T2w", r"[_-](t2w|t2)[_.]"),
        ("T2f", r"[_-](t2f|flair)[_.]"),
    )

    def __init__(self):
        super().__init__()
        self.setObjectName("patientCard")
        self.setStyleSheet(_card_qss("patientCard"))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        head, _, self.meta = _card_head("Patient Record", "BraTS-PEDs")
        layout.addLayout(head)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(5)
        self._values = {}
        for r, key in enumerate(("Patient ID", "Voxel size", "Dimensions")):
            k = QLabel(key)
            k.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
            v = QLabel("\u2014")
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            v.setStyleSheet(f"color: {TEXT_LIGHT}; font-size: 12px; font-weight: 500;")
            grid.addWidget(k, r, 0)
            grid.addWidget(v, r, 1)
            self._values[key] = v
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        chips = QHBoxLayout()
        chips.setContentsMargins(0, 2, 0, 0)
        chips.setSpacing(6)
        self._chips = {}
        for tag, _ in self.MODALITY_TAGS:
            c = QLabel(tag)
            chips.addWidget(c)
            self._chips[tag] = c
        chips.addStretch(1)
        layout.addLayout(chips)
        self.reset()

    def _chip_style(self, label: QLabel, state: str):
        if state == "on":
            style = f"color: {ACCENT}; background: #EAF2FF;"
            tip = "Found in folder"
        elif state == "missing":
            style = "color: #B5C0D3; background: #F3F5F9; text-decoration: line-through;"
            tip = "Missing from folder"
        else:
            style = "color: #B5C0D3; background: #F3F5F9;"
            tip = ""
        label.setStyleSheet(
            style + " border: none; border-radius: 10px; padding: 3px 10px; font-size: 11px; font-weight: 600;"
        )
        label.setToolTip(tip)

    def reset(self):
        for v in self._values.values():
            v.setText("\u2014")
            v.setToolTip("")
        for c in self._chips.values():
            self._chip_style(c, "idle")

    def set_case(self, case_id: str, files, shape=None, zooms=None):
        self._values["Patient ID"].setText(case_id if len(case_id) <= 24 else case_id[:22] + "\u2026")
        self._values["Patient ID"].setToolTip(case_id)
        if zooms:
            self._values["Voxel size"].setText(" \u00d7 ".join(f"{z:.1f}" for z in zooms) + " mm")
        if shape:
            self._values["Dimensions"].setText(" \u00d7 ".join(str(int(v)) for v in shape))
        names = [Path(f).name.lower() for f in files]
        for tag, pattern in self.MODALITY_TAGS:
            found = any(re.search(pattern, n) for n in names)
            self._chip_style(self._chips[tag], "on" if found else "missing")


# ---------------------------------------------------------------------------
# Report summary — four headline metrics for the whole case.
# ---------------------------------------------------------------------------
# BraTS volumes are skull-stripped, so anything above ~0 in the normalised
# FLAIR is brain tissue. Raise this if a non-stripped scan counts background.
BRAIN_TISSUE_THRESHOLD = 0.01


def _brain_tissue_mask(volume: np.ndarray, tumor_mask: Optional[np.ndarray] = None) -> np.ndarray:
    brain = volume > BRAIN_TISSUE_THRESHOLD
    if tumor_mask is not None:
        brain |= tumor_mask > 0.5   # a tumor voxel is always inside the brain
    return brain


def _rough_location(component: np.ndarray, brain: np.ndarray, axcodes) -> Optional[tuple]:
    """
    Where a tumor sits inside the brain's bounding box, as
    (side, front/back, height), e.g. ("Right", "Posterior", "Inferior").

    Uses the scan's native orientation codes (nibabel.aff2axcodes), which the
    direct resize to the display grid preserves. Purely geometric: it says
    which part of the head the tumor centroid is in, not which lobe.
    Side is the patient's left/right.
    """
    if axcodes is None or not component.any():
        return None
    hits = np.argwhere(brain)
    if hits.size == 0:
        lo, hi = np.zeros(3), np.array(brain.shape, dtype=float) - 1
    else:
        lo, hi = hits.min(axis=0), hits.max(axis=0)
    centroid = ndi_center_of_mass(component)
    toward = {"R": "R", "L": "R", "A": "A", "P": "A", "S": "S", "I": "S"}
    pos = {}
    for axis, code in enumerate(axcodes):
        if code not in toward:
            return None
        t = (centroid[axis] - lo[axis]) / max(1.0, float(hi[axis] - lo[axis]))
        pos[toward[code]] = t if code == toward[code] else 1.0 - t
    if len(pos) != 3:
        return None
    r, a, s = pos["R"], pos["A"], pos["S"]
    side = "Right" if r > 0.55 else "Left" if r < 0.45 else "Midline"
    ap = "Anterior" if a > 0.62 else "Posterior" if a < 0.38 else "Central"
    si = "Superior" if s > 0.62 else "Inferior" if s < 0.38 else "Mid-level"
    return side, ap, si


class _SummaryTile(QFrame):
    """One metric: small caption, large value, muted sub-line."""

    def __init__(self, caption: str):
        super().__init__()
        self.setObjectName("summaryTile")
        self.setStyleSheet(
            f"QFrame#summaryTile {{ background-color: #F7FAFF; border: 1px solid {GLASS_EDGE}; "
            "border-radius: 10px; }"
            "QFrame#summaryTile QLabel { border: none; background: transparent; }"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(1)

        cap = QLabel(caption.upper())
        cap.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 8px; font-weight: 700; letter-spacing: 1.2px;"
        )
        layout.addWidget(cap)

        self.value = QLabel("—")
        self.value.setWordWrap(True)
        layout.addWidget(self.value)

        self.sub = QLabel("")
        self.sub.setWordWrap(True)
        self.sub.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 8px;")
        layout.addWidget(self.sub)
        self.set("—", "", filled=False)

    def set(self, value: str, sub: str = "", filled: bool = True, small: bool = False) -> None:
        color = ACCENT if filled else TEXT_DIM
        size = 11 if small else 14
        self.value.setStyleSheet(f"color: {color}; font-size: {size}px; font-weight: 700;")
        self.value.setText(value)
        self.sub.setText(sub)
        self.sub.setVisible(bool(sub))


class ReportSummaryPanel(QWidget):
    """Total volume · component count · rough location · % of brain volume.
    Pinned at the top of the Report card, above the scrolling tumor list."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background: transparent;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        self.volume_tile = _SummaryTile("Total volume")
        self.count_tile = _SummaryTile("Components")
        self.percent_tile = _SummaryTile("% of brain volume")
        self.location_tile = _SummaryTile("Rough location")
        grid.addWidget(self.volume_tile, 0, 0)
        grid.addWidget(self.count_tile, 0, 1)
        grid.addWidget(self.percent_tile, 1, 0)
        grid.addWidget(self.location_tile, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        self.reset()

    def reset(self) -> None:
        for tile in (self.volume_tile, self.count_tile, self.percent_tile, self.location_tile):
            tile.set("—", "", filled=False)

    def set_summary(self, total_cm3: float, count: int, percent_brain: Optional[float],
                    location: Optional[tuple], largest_color: Optional[str],
                    bilateral: bool) -> None:
        self.volume_tile.set(f"{total_cm3:.2f} cm\u00b3")
        self.volume_tile.setToolTip("All tumors combined")
        self.count_tile.set(str(count))
        self.count_tile.setToolTip("Separate tumor regions in the mask")
        if percent_brain is None:
            self.percent_tile.set("—", "", filled=False)
        else:
            self.percent_tile.set(f"{percent_brain:.2f}%")
            self.percent_tile.setToolTip("Tumor voxels as a share of brain tissue")

        if count == 0 or location is None:
            self.location_tile.set("—", "", filled=False)
            return
        side, ap, si = location
        headline = "Bilateral" if bilateral else (
            "Midline" if side == "Midline" else f"{side} hemisphere"
        )
        detail = f"{ap} \u00b7 {si}"
        self.location_tile.set(headline, detail, small=True)
        self.location_tile.setToolTip(
            f"Position of the largest ({largest_color}) tumor" if count > 1 and largest_color
            else "Position of the tumor inside the brain"
        )


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

    def reset(self) -> None:
        self._loaded = False
        self._title.setText("Drop volume")
        self._hint.setText("Drop a patient folder here\nor click to browse")
        self._set_style(hover=False)

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

_NEON_OFF = QColor(220, 228, 240)


def _neon_shades(hex_color: str) -> tuple:
    """(outer glow, inner glow, dark body, main, hot core) for one segment colour."""
    base = QColor(hex_color)
    outer = QColor(base); outer.setAlpha(50)
    inner = QColor(base); inner.setAlpha(95)
    body = base.darker(160)
    core = base.lighter(165)
    return (outer, inner, body, QColor(base), core)


# Segments pick up the iridescent palette: top blue, middle violet, bottom magenta.
_NEON_SEG_SHADES = {
    "a": _neon_shades("#56c2ff"),
    "b": _neon_shades("#6f8dff"),
    "f": _neon_shades("#6f8dff"),
    "g": _neon_shades("#8b6cff"),
    "c": _neon_shades("#b066ea"),
    "e": _neon_shades("#b066ea"),
    "d": _neon_shades("#e85ad6"),
}


class NeonDigits(QWidget):
    """Seven-segment readout with layered yellowish-orange neon glow."""

    def __init__(self, digits: str = "\u2014"):
        super().__init__()
        self._digits = digits
        self.setFixedHeight(SURVIVAL_DIGITS_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent; border: none;")

    def set_value(self, days: Optional[int]) -> None:
        if days is None:
            self._digits = "\u2014"   # em dash: no prediction yet
        else:
            n = max(0, int(days))
            self._digits = f"{n:03d}" if n < 1000 else str(n)
        self.update()

    def paintEvent(self, event):
        """Thin, widely spaced digits filled with a sky -> violet -> pink
        gradient and a soft blue glow — same look as the design preview."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        text = self._digits
        area = QRectF(self.rect()).adjusted(8, 4, -8, -4)

        font = QFont("Segoe UI")
        font.setWeight(QFont.Light)
        size = max(10, int(area.height() * 0.95))
        while True:
            font.setPixelSize(size)
            font.setLetterSpacing(QFont.AbsoluteSpacing, size * 0.15)
            path = QPainterPath()
            path.addText(0, 0, font, text)
            box = path.boundingRect()
            if (box.width() <= area.width() and box.height() <= area.height()) or size <= 10:
                break
            size -= 2
        path.translate(area.center().x() - box.center().x(), area.center().y() - box.center().y())
        box = path.boundingRect()

        # Soft glow: a few wide, faint strokes under the fill.
        for width, alpha in ((12, 16), (7, 28), (3.5, 44)):
            pen = QPen(QColor(31, 111, 235, alpha // 3), width)
            pen.setJoinStyle(Qt.RoundJoin)
            pen.setCapStyle(Qt.RoundCap)
            painter.strokePath(path, pen)

        fill = QLinearGradient(box.left(), 0, box.right(), 0)
        fill.setColorAt(0.0, QColor("#1F6FEB"))
        fill.setColorAt(1.0, QColor("#5AA2FF"))
        painter.fillPath(path, fill)
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
            f"QFrame {{ border: 1px solid {GLASS_EDGE}; border-radius: 18px; background-color: {GLASS_BG}; }}"
            "QLabel { border: none; }"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, SURVIVAL_EDGE_GAP + 4, 14, SURVIVAL_EDGE_GAP)
        layout.setSpacing(6)

        title = QWidget()
        title.setFixedHeight(16)
        title.setStyleSheet("background: transparent;")
        head, _, _ = _card_head("Predicted Survival", "days")
        title.setLayout(head)
        layout.addWidget(title)

        self.digits = NeonDigits("000")
        layout.addWidget(self.digits)
        self.setFixedHeight(SURVIVAL_EDGE_GAP + 4 + 16 + 6 + SURVIVAL_DIGITS_HEIGHT + SURVIVAL_EDGE_GAP)

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


# MedSAM2 card on the right of each 2D view.
SAM_SIDEBAR_WIDTH = 104
_SAM_PROMPT_HINT = "L/R CLICK: FG/BG\nDRAG: BOX"
_SAM_PROMPT_TIP = "Left click: foreground point\nRight click: background point\nDrag: box prompt"


# ---------------------------------------------------------------------------
# 2D hive backdrop — the same design as the 3D panels' SVG backdrop, redrawn
# with QPainter so it can sit behind the matplotlib slice image. The dark
# surround of each MRI slice is made transparent so this shows around the head.
# ---------------------------------------------------------------------------
# Voxel intensity ramp (normalised FLAIR) over which the surround fades from
# see-through to solid; brain tissue sits well above SLICE_BG_HI.
SLICE_BG_LO = 0.005
SLICE_BG_HI = 0.04

_HIVE_2D_LAYOUTS = {
    "sagittal": {
        "hexes": ((90, 22, 10, 0.50, 6), (95, 40, 6, 0.34, 4), (84, 14, 5, 0.28, 3),
                  (7, 76, 9, 0.44, 5), (13, 88, 5, 0.30, 6)),
        "traces": (((3, 10), (26, 10)), ((3, 10), (3, 46)), ((74, 90), (97, 90)), ((97, 90), (97, 58))),
        "nodes": ((3, 10), (26, 10), (3, 46), (74, 90), (97, 58)),
    },
    "axial": {
        "hexes": ((8, 22, 10, 0.50, 6), (4, 40, 6, 0.32, 3), (14, 12, 5, 0.28, 4),
                  (92, 78, 9, 0.44, 5), (86, 90, 5, 0.30, 6)),
        "traces": (((97, 10), (72, 10)), ((97, 10), (97, 44)), ((3, 90), (28, 90)), ((3, 90), (3, 60))),
        "nodes": ((97, 10), (72, 10), (97, 44), (28, 90), (3, 60)),
    },
    "coronal": {
        "hexes": ((88, 80, 10, 0.50, 6), (94, 64, 6, 0.34, 4), (10, 18, 9, 0.44, 5),
                  (16, 30, 5, 0.30, 3), (4, 12, 5, 0.26, 6)),
        "traces": (((3, 90), (24, 90)), ((24, 90), (29, 82)), ((97, 10), (70, 10)), ((97, 10), (97, 40))),
        "nodes": ((3, 90), (29, 82), (70, 10), (97, 40)),
    },
}
_HIVE_2D_CACHE: dict = {}


SLICE_BG_BASE = SCAN_BG
# Per-plane glow positions so the three 2D rows don't look identical.
_SLICE_GLOWS = {
    "sagittal": ((0.22, 0.28, "#4fa3ff"), (0.82, 0.76, "#8b6cff")),
    "axial": ((0.80, 0.24, "#8b6cff"), (0.18, 0.78, "#2fd4c4")),
    "coronal": ((0.20, 0.74, "#e85ad6"), (0.80, 0.26, "#4fa3ff")),
}


def _hive_backdrop_2d(width: int, height: int, variant: str) -> np.ndarray:
    """(height, width, 3) float RGB: still version of the blue ambient backdrop
    for the 2D slices (animating it would force constant matplotlib redraws)."""
    key = (width, height, variant)
    cached = _HIVE_2D_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_HIVE_2D_CACHE) > 12:
        _HIVE_2D_CACHE.clear()
    w, h = max(1, width), max(1, height)
    image = QImage(w, h, QImage.Format_RGBA8888)
    image.fill(QColor(SLICE_BG_BASE))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)

    # Minimal theme: flat dark viewport with a barely-there lift in the centre.
    side = float(max(w, h))
    painter.setPen(Qt.NoPen)
    centre = QRadialGradient(QPointF(w / 2, h / 2), side * 0.5)
    centre.setColorAt(0.0, QColor(40, 70, 140, 34))
    centre.setColorAt(1.0, QColor(40, 70, 140, 0))
    painter.setBrush(centre)
    painter.drawEllipse(QPointF(w / 2, h / 2), side * 0.5, side * 0.5)
    painter.end()

    buf = bytes(image.constBits())
    rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, image.bytesPerLine())
    rgb = rgba[:, : w * 4].reshape(h, w, 4)[:, :, :3].astype(np.float32) / 255.0
    _HIVE_2D_CACHE[key] = rgb
    return rgb

class SamCard(HivePanel):
    """Separate bordered card holding one 2D view's MedSAM2 controls."""

    def __init__(self, controls: QWidget):
        super().__init__()
        self.setObjectName("samCard")
        self.setFixedWidth(SAM_SIDEBAR_WIDTH)
        # Ignored vertical policy: the card takes whatever height its 2D view
        # row has and never pushes the row taller when REFINE reveals tools.
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Ignored)
        self.setMinimumHeight(0)
        # Same frame as the 2D views, scoped by object name so the border
        # doesn't cascade onto the labels inside.
        self.setStyleSheet(
            f"QFrame#samCard {{ border: 1px solid {GLASS_EDGE}; border-radius: 18px; "
            f"background-color: {GLASS_BG}; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 10, 9, 12)
        layout.setSpacing(0)
        layout.addWidget(controls)
        self._sync_hive_overlay()

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

        self.setFrameShape(QFrame.Box)
        # Pure black background (not the charcoal BG_PANEL used elsewhere)
        # — MRI slices have their own black background outside the brain,
        # so matching that exactly makes the panel disappear into the
        # image instead of showing as a separate charcoal-colored frame
        # around it.
        self.setStyleSheet(
            f"QFrame {{ border: 1px solid {GLASS_EDGE}; border-radius: 18px; "
            f"background-color: {GLASS_BG}; }}"
        )
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 10)
        layout.setSpacing(8)

        header, self.title_label, self.index_label = _card_head(title, "")
        self._base_title = title
        self.maximize_btn = _fullscreen_button("Maximize")
        header.addWidget(self.maximize_btn)
        layout.addLayout(header)

        self.figure = Figure(figsize=(3, 3))
        self.figure.patch.set_facecolor(SLICE_BG_BASE)
        self._bg_artist = None
        self._bg_key = None
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setStyleSheet(f"background-color: {SLICE_BG_BASE};")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor((0, 0, 0, 0))
        self.ax.axis("off")
        self.canvas.mpl_connect("resize_event", self._on_canvas_resize)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)

        # Canvas + loading overlay share the same space via a stacked
        # layout, so the spinner appears directly on top of the slice
        # image instead of needing a separate panel or popup.
        self.canvas_stack = QStackedLayout()
        self.canvas_stack.setStackingMode(QStackedLayout.StackAll)
        canvas_container = _RoundedHost(12)
        canvas_container.setLayout(self.canvas_stack)
        self.canvas_stack.addWidget(self.canvas)

        self.loading_overlay = QWidget()
        # Lighter than before — the pixelated image itself now carries most
        # of the visual weight, so a heavy flat black backdrop would just
        # compete with it instead of framing it.
        self.loading_overlay.setStyleSheet("background-color: rgba(5, 8, 30, 120);")
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
        self.loading_animation = ArcSpinner(diameter=26, color=SCAN_TEXT)
        spinner_row.addWidget(self.loading_animation)
        corner_layout.addLayout(spinner_row)
        corner_layout.addStretch(1)

        self.loading_caption = QLabel("SCANNING")
        self.loading_caption.setAlignment(Qt.AlignCenter)
        self.loading_caption.setStyleSheet(
            f"color: {SCAN_TEXT}; font-size: 10px; font-weight: 600; "
            f"letter-spacing: 2px; background: transparent; border: none;"
        )
        corner_layout.addWidget(self.loading_caption)

        overlay_stack.addWidget(corner_layer)

        self.canvas_stack.addWidget(self.loading_overlay)
        self.loading_overlay.hide()

        # Image + slider on the left, MedSAM2 sidebar on the right, so the
        # slice image gets the panel's full height.
        body_row = QHBoxLayout()
        body_row.setContentsMargins(0, 0, 0, 0)
        body_row.setSpacing(8)
        image_col = QVBoxLayout()
        image_col.setContentsMargins(0, 0, 0, 0)
        image_col.setSpacing(8)
        image_col.addWidget(canvas_container, stretch=1)

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
        image_col.addWidget(slider_wrap)
        body_row.addLayout(image_col, stretch=1)

        # MedSAM2 sidebar. Progressive disclosure as before: title, status and
        # REFINE are always shown; prompt tools appear while editing, and
        # ACCEPT / DISCARD only when a preview is ready.
        self.sam_controls = QWidget()
        self.sam_controls.setObjectName("samControls")
        self.sam_controls.setStyleSheet(
            "QWidget#samControls { background: transparent; border: none; }"
            "QWidget#samControls QLabel { border: none; background: transparent; }"
        )
        sam_layout = QVBoxLayout(self.sam_controls)
        sam_layout.setContentsMargins(0, 0, 0, 0)
        sam_layout.setSpacing(5)

        sam_title = QLabel("MEDSAM2")
        sam_title.setAlignment(Qt.AlignCenter)
        sam_title.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1.6px;"
        )
        sam_layout.addWidget(sam_title)

        self.sam_status = QLabel("LOCKED")
        self.sam_status.setAlignment(Qt.AlignCenter)
        self.sam_status.setStyleSheet(
            f"color: {TEXT_MUTED}; background: #EEF3FB; border: none; border-radius: 7px; "
            "padding: 2px 4px; font-size: 8px; font-weight: 700;"
        )
        sam_layout.addWidget(self.sam_status)

        secondary_style = (
            f"QPushButton {{ color: {TEXT_MUTED}; background: transparent; border: 1px solid {BORDER_DIM}; "
            "border-radius: 6px; padding: 4px 4px; font-size: 9px; font-weight: 600; }}"
            f"QPushButton:hover {{ color: {TEXT_LIGHT}; border-color: {ACCENT_DEEP}; "
            "background: #EAF2FF; }}"
            "QPushButton:disabled { color: #B5C0D3; border-color: #EEF2F8; background: transparent; }"
        )
        primary_style = (
            f"QPushButton {{ color: #ffffff; background: {GRADIENT_QSS}; border: none; "
            "border-radius: 6px; padding: 5px 4px; font-size: 9px; font-weight: 800; }}"
            f"QPushButton:hover {{ background: {GRADIENT_QSS_HOVER}; }}"
            "QPushButton:disabled { color: #A3AFC2; background: #E3E9F3; }"
        )
        self.sam_refine_btn = QPushButton("REFINE")
        self.sam_refine_btn.setCheckable(True)
        self.sam_refine_btn.setEnabled(False)
        self.sam_refine_btn.setCursor(Qt.PointingHandCursor)
        self.sam_refine_btn.setStyleSheet(
            secondary_style
            + f"QPushButton:checked {{ color: {ACCENT_SOFT}; border-color: {ACCENT}; "
            "background: #EAF2FF; }"
        )
        sam_layout.addWidget(self.sam_refine_btn)

        self.sam_editor = QWidget()
        self.sam_editor.setStyleSheet("background: transparent; border: none;")
        editor_layout = QVBoxLayout(self.sam_editor)
        editor_layout.setContentsMargins(0, 2, 0, 0)
        editor_layout.setSpacing(4)
        self.sam_prompt_summary = QLabel(_SAM_PROMPT_HINT)
        self.sam_prompt_summary.setToolTip(_SAM_PROMPT_TIP)
        self.sam_prompt_summary.setAlignment(Qt.AlignCenter)
        self.sam_prompt_summary.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 8px; letter-spacing: .3px;"
        )
        self.sam_prompt_summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.sam_prompt_summary.setWordWrap(True)
        editor_layout.addWidget(self.sam_prompt_summary)

        self.sam_undo_btn = QPushButton("Undo")
        self.sam_clear_btn = QPushButton("Clear")
        self.sam_run_btn = QPushButton("PREVIEW")
        undo_clear_row = QHBoxLayout()
        undo_clear_row.setContentsMargins(0, 0, 0, 0)
        undo_clear_row.setSpacing(4)
        for button in (self.sam_undo_btn, self.sam_clear_btn):
            button.setStyleSheet(secondary_style)
            button.setCursor(Qt.PointingHandCursor)
            undo_clear_row.addWidget(button)
        editor_layout.addLayout(undo_clear_row)
        self.sam_run_btn.setStyleSheet(primary_style)
        self.sam_run_btn.setCursor(Qt.PointingHandCursor)
        editor_layout.addWidget(self.sam_run_btn)
        sam_layout.addWidget(self.sam_editor)

        self.sam_review = QWidget()
        self.sam_review.setStyleSheet("background: transparent; border: none;")
        review_col = QVBoxLayout(self.sam_review)
        review_col.setContentsMargins(0, 2, 0, 0)
        review_col.setSpacing(5)
        review_label = QLabel("PREVIEW\nREADY")
        review_label.setAlignment(Qt.AlignCenter)
        review_label.setStyleSheet(
            f"color: {ACCENT_TEAL_SOFT}; font-size: 8px; font-weight: 700; letter-spacing: 1px;"
        )
        self.sam_discard_btn = QPushButton("DISCARD")
        self.sam_accept_btn = QPushButton("ACCEPT")
        self.sam_discard_btn.setStyleSheet(secondary_style)
        self.sam_accept_btn.setStyleSheet(primary_style)
        for button in (self.sam_accept_btn, self.sam_discard_btn):
            button.setCursor(Qt.PointingHandCursor)
        review_col.addWidget(self.sam_accept_btn)
        review_col.addWidget(self.sam_discard_btn)
        sam_layout.addWidget(self.sam_review)
        sam_layout.addStretch(1)

        self.sam_refine_btn.toggled.connect(self._set_sam_editing)
        self.sam_undo_btn.clicked.connect(self._undo_sam_prompt)
        self.sam_clear_btn.clicked.connect(self.clear_sam_prompts)
        self.sam_run_btn.clicked.connect(self._request_sam_refinement)
        self.sam_accept_btn.clicked.connect(self.accept_requested.emit)
        self.sam_discard_btn.clicked.connect(self.discard_requested.emit)
        self.sam_editor.hide()
        self.sam_review.hide()
        self.set_sam_preview_available(False)
        # The MedSAM2 controls live in their own card, placed beside this
        # panel by MainWindow (see self.sam_card).
        self.sam_card = SamCard(self.sam_controls)
        layout.addLayout(body_row, stretch=1)
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
        self._draw_slice(self.slider.value())

    def clear_detection(self):
        """Remove segmentation overlays while keeping the loaded MRI volume."""
        self.mask = None
        self.probs = None
        self.labels = None
        self.color_map = None
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
        # Only one tool group at a time keeps the card compact.
        self.sam_editor.setVisible(self._sam_editing and not self._sam_preview_available)

        if self._sam_busy:
            status, color = "PROCESSING", ACCENT_TEAL_SOFT
        elif self._sam_preview_available:
            status, color = "REVIEW", ACCENT_TEAL_SOFT
        elif self._sam_enabled:
            status, color = "READY", ACCENT_SOFT
        else:
            status, color = "LOAD MRI", TEXT_MUTED
        self.sam_status.setText(status)
        self.sam_status.setStyleSheet(
            f"color: {color}; background: #EEF3FB; border: none; border-radius: 7px; "
            "padding: 2px 6px; font-size: 8px; font-weight: 700;"
        )

        count = len(self._sam_points) + int(self._sam_box is not None)
        self.sam_prompt_summary.setText(
            f"{count} PROMPT{'S' if count != 1 else ''}\nDRAG: BOX"
            if count else _SAM_PROMPT_HINT
        )

    def _set_sam_editing(self, editing: bool):
        self._sam_editing = editing and self._sam_enabled
        self.canvas.setCursor(Qt.CrossCursor if self._sam_editing else Qt.ArrowCursor)
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

    def _on_canvas_press(self, event):
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
        self._ensure_backdrop()
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

        # Fade the near-black surround to transparent so the hive backdrop
        # shows around the head; tissue and any tumor overlay stay opaque.
        alpha = np.clip(
            (img_slice - SLICE_BG_LO) / (SLICE_BG_HI - SLICE_BG_LO), 0.0, 1.0
        ).astype(np.float32)
        if self.probs is not None:
            alpha = np.maximum(alpha, (self._slice_along_axis(self.probs, index) > 0.2).astype(np.float32))
        elif self.mask is not None:
            alpha = np.maximum(alpha, (self._slice_along_axis(self.mask, index) > 0.5).astype(np.float32))
        rgba = np.dstack([np.clip(rgb, 0.0, 1.0), alpha])
        shown = np.rot90(rgba)
        self.ax.clear()
        self.ax.imshow(shown, aspect="equal", interpolation="bilinear")
        self._draw_sam_prompts(rgb.shape[1])
        self.ax.set_aspect("equal", adjustable="datalim")
        self._apply_brain_zoom(shown.shape[0], shown.shape[1], rgb.shape[1])
        self.ax.axis("off")
        self.canvas.draw_idle()

        self.index_label.setText(f"{index} / {self.slider.maximum()}")
        self._sync_hive_overlay()

    def _draw_sam_prompts(self, pre_rot_cols: int):
        for x, y, label in self._sam_points:
            shown_x, shown_y = prompt_to_rotated_xy(x, y, pre_rot_cols)
            color = ACCENT_TEAL if label == 1 else "#ff5c8a"
            marker = "+" if label == 1 else "x"
            self.ax.plot(shown_x, shown_y, marker=marker, color=color, markersize=9, markeredgewidth=2)
        if self._sam_box is not None:
            x0, y0, x1, y1 = self._sam_box
            shown_x0, shown_x1 = y0, y1
            shown_y0, shown_y1 = pre_rot_cols - 1 - x1, pre_rot_cols - 1 - x0
            self.ax.add_patch(Rectangle(
                (shown_x0, shown_y0), shown_x1 - shown_x0, shown_y1 - shown_y0,
                fill=False, edgecolor=ACCENT_SOFT, linewidth=1.8,
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

    def _ensure_backdrop(self):
        """(Re)place the hive backdrop behind the axes at the canvas's exact
        pixel size; only regenerated when the size changes."""
        width = int(round(self.figure.bbox.width))
        height = int(round(self.figure.bbox.height))
        if width < 2 or height < 2:
            return
        key = (width, height)
        if key == self._bg_key and self._bg_artist is not None:
            return
        if self._bg_artist is not None:
            try:
                self._bg_artist.remove()
            except (ValueError, NotImplementedError):
                pass
        backdrop = _hive_backdrop_2d(width, height, self.plane)
        self._bg_artist = self.figure.figimage(backdrop, xo=0, yo=0, origin="upper", zorder=-10)
        self._bg_key = key

    def _on_canvas_resize(self, _event=None):
        """The canvas changed size: re-fit the backdrop so it covers it fully."""
        self._ensure_backdrop()
        self.canvas.draw_idle()

    def clear_volume(self):
        """Back to the empty state: no MRI, no overlays, no prompts."""
        self.volume = None
        self.mask = None
        self.probs = None
        self.labels = None
        self.color_map = None
        self._bbox = None
        self.clear_sam_prompts()
        self.enable_sam(False)
        self.set_sam_preview_available(False)
        self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.blockSignals(False)
        self.index_label.setText("")
        self._show_hive_empty()

    def _show_hive_empty(self):
        self._ensure_backdrop()
        self.ax.clear()
        self.ax.set_facecolor((0, 0, 0, 0))
        self.ax.axis("off")
        self.ax.text(
            0.5, 0.5, "LOAD AN MRI FOLDER TO BEGIN", transform=self.ax.transAxes,
            ha="center", va="center", color=SCAN_TEXT, fontsize=8, fontweight="bold", alpha=0.75,
        )
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


# ---------------------------------------------------------------------------
# 3D hive backdrop — the header's amber honeycomb / circuit language, drawn
# as an SVG layer *behind* the Plotly scene. Plotly's own backgrounds are
# made transparent (see _style_ui_figure) so this shows through the WebGL
# canvas while rotate / zoom / middle-click pan keep working on top.
# Tweak these three numbers to make the pattern louder or quieter.
# ---------------------------------------------------------------------------
HIVE_3D_GRID_RADIUS = 14      # px, honeycomb cell radius
HIVE_3D_GRID_ALPHA = 0.16     # grid stroke opacity at the panel edges
HIVE_3D_CLEAR_ZONE = 0.34     # 0-1, how wide the pattern-free centre (where the model sits) is

# Per-panel layouts so the two 3D cards don't look copy-pasted.
# hexes:  (x%, y%, radius px, stroke opacity, sides drawn)  — same idea as the
#         header's _paint_sparse_hives: a few irregular, partly-open hexes.
# traces: circuit polylines in % coordinates; nodes: small filled hex pads.
_HIVE_3D_LAYOUTS = {
    "tumor": {
        "hexes": (
            (86, 20, 11, 0.55, 6), (92, 34, 7, 0.40, 4), (80, 30, 6, 0.30, 3),
            (10, 78, 9, 0.45, 5), (18, 88, 6, 0.32, 6), (6, 64, 5, 0.28, 3),
        ),
        "traces": (
            ((4, 10), (34, 10)), ((4, 10), (4, 42)),
            ((66, 92), (96, 92)), ((96, 92), (96, 60)),
        ),
        "nodes": ((4, 10), (34, 10), (4, 42), (66, 92), (96, 60)),
    },
    "brain": {
        "hexes": (
            (88, 16, 12, 0.55, 6), (82, 26, 7, 0.36, 4), (94, 30, 8, 0.42, 5),
            (74, 12, 5, 0.26, 3), (8, 84, 10, 0.48, 6), (16, 74, 6, 0.32, 4),
            (12, 94, 5, 0.26, 3), (90, 86, 7, 0.34, 5),
        ),
        "traces": (
            ((3, 7), (40, 7)), ((3, 7), (3, 36)), ((40, 7), (46, 13)),
            ((58, 95), (97, 95)), ((97, 95), (97, 62)), ((58, 95), (52, 89)),
        ),
        "nodes": ((3, 7), (46, 13), (3, 36), (52, 89), (97, 62)),
    },
}


def _svg_hex_pts(cx: float, cy: float, r: float) -> list:
    """Pointy-top hex vertices, same orientation as _hex_points()."""
    return [
        (cx + r * math.cos(math.radians(60 * i - 30)), cy + r * math.sin(math.radians(60 * i - 30)))
        for i in range(6)
    ]


def _fmt_pts(pts) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)


# Dotted background of the 3D Brain view. Tweak these to taste.
DOT_GRID_SPACING = 18     # px between dots (smaller = denser)
DOT_GRID_SIZE = 1.2       # dot radius in px
DOT_GRID_ALPHA = 0.22     # dot opacity (0 = invisible, 1 = solid blue)


def _hive_backdrop_svg(variant: str = "brain") -> str:
    """Quiet layer behind a 3D scene: for the brain view, a faint platform
    ring the model sits above. (Name kept from the old hive design.)"""
    if variant != "brain":
        return ""
    return (
        '<div class="nv-dots"></div>'
        '<svg class="nv-floor" viewBox="0 0 400 80" preserveAspectRatio="xMidYMid meet">'
        '<ellipse cx="200" cy="40" rx="118" ry="17" fill="none" stroke="#1F6FEB" stroke-opacity=".22"/>'
        '<ellipse cx="200" cy="40" rx="160" ry="24" fill="none" stroke="#1F6FEB" stroke-opacity=".10"/>'
        '<ellipse cx="200" cy="40" rx="78" ry="10" fill="#1F6FEB" fill-opacity=".05"/>'
        "</svg>"
    )


def _hive_page_css() -> str:
    """Page styling shared by the empty state and the rendered Plotly page."""
    return (
        "<style>"
        "html,body{margin:0;padding:0;height:100%;background:transparent;overflow:hidden;}"
        "#hive-bg{position:fixed;inset:0;z-index:0;pointer-events:none;border-radius:12px;overflow:hidden;"
        "background:radial-gradient(ellipse at 50% 45%,#FFFFFF 0%,#F4F8FE 55%,#E9F1FD 100%);}"
        # Dot grid behind the 3D brain: fades out toward the card edges.
        f".nv-dots{{position:absolute;inset:0;"
        f"background-image:radial-gradient(circle,rgba(31,111,235,{DOT_GRID_ALPHA}) {DOT_GRID_SIZE}px,"
        f"transparent {DOT_GRID_SIZE + 0.4}px);"
        f"background-size:{DOT_GRID_SPACING}px {DOT_GRID_SPACING}px;"
        f"background-position:center center;"
        "-webkit-mask-image:radial-gradient(ellipse 75% 70% at 50% 50%,#000 35%,transparent 100%);"
        "mask-image:radial-gradient(ellipse 75% 70% at 50% 50%,#000 35%,transparent 100%);}"
        ".nv-floor{position:absolute;left:50%;bottom:4%;width:78%;transform:translateX(-50%);}"
        ".plotly-graph-div{position:relative;z-index:1;background:transparent!important;}"
        "#hive-msg{position:relative;z-index:1;height:100%;display:flex;align-items:center;"
        "justify-content:center;font:600 11px 'Segoe UI',sans-serif;letter-spacing:1.6px;"
        f"color:{TEXT_MUTED};text-transform:uppercase;}}"
        "</style>"
    )


def _hive_backdrop_div(variant: str) -> str:
    return f'<div id="hive-bg">{_hive_backdrop_svg(variant)}</div>'


def _hive_empty_page(variant: str, message: str) -> str:
    """Empty-state page for a 3D panel: same backdrop, centred hint text."""
    return (
        f"<html><head>{_hive_page_css()}</head><body>"
        f"{_hive_backdrop_div(variant)}"
        f'<div id="hive-msg">{message}</div>'
        "</body></html>"
    )


def _inject_hive_backdrop(html_path: str, variant: str):
    """Add the hive page CSS + backdrop layer to a Plotly write_html file."""
    try:
        path = Path(html_path)
        content = path.read_text(encoding="utf-8")
    except OSError:
        return
    if "<head>" in content:
        content = content.replace("<head>", f"<head>{_hive_page_css()}", 1)
    content = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + _hive_backdrop_div(variant), content, count=1)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError:
        pass


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


# ---------------------------------------------------------------------------
# Auto-rotation for the 3D brain: the camera orbits slowly around the scene's
# up axis. Any mouse interaction (rotate, zoom, middle-click pan) pauses it,
# and it resumes from wherever the user left the view after a short idle.
# ---------------------------------------------------------------------------
AUTO_ROTATE_SECONDS_PER_TURN = 24   # lower = faster spin
AUTO_ROTATE_RESUME_MS = 3000        # idle time after interaction before it spins again
AUTO_ROTATE_FPS = 30                # camera updates per second (keeps the GPU load light)


def _inject_auto_rotate(html_path: str):
    try:
        path = Path(html_path)
        content = path.read_text(encoding="utf-8")
    except OSError:
        return
    speed = 2 * math.pi / max(1.0, float(AUTO_ROTATE_SECONDS_PER_TURN))
    script_block = """
<script>
(function() {
    var SPEED = %(speed).6f;          // radians per second
    var RESUME_MS = %(resume)d;
    var FRAME_MS = %(frame).3f;

    function setup() {
        var gd = document.querySelector('.plotly-graph-div');
        if (!gd || !window.Plotly || !gd._fullLayout || !gd._fullLayout.scene) {
            setTimeout(setup, 100);
            return;
        }
        var paused = false, busy = false, last = null, resumeTimer = null;

        function pause() { paused = true; clearTimeout(resumeTimer); }
        function resumeLater() {
            clearTimeout(resumeTimer);
            resumeTimer = setTimeout(function() { paused = false; last = null; }, RESUME_MS);
        }
        gd.addEventListener('mousedown', pause, true);
        window.addEventListener('mouseup', function() { if (paused) resumeLater(); }, true);
        gd.addEventListener('wheel', function() { pause(); resumeLater(); },
                            {capture: true, passive: true});
        gd.addEventListener('touchstart', pause, {capture: true, passive: true});
        gd.addEventListener('touchend', resumeLater, {capture: true, passive: true});
        // Hold still while a tumour's hover label is showing.
        if (gd.on) { gd.on('plotly_hover', pause); gd.on('plotly_unhover', resumeLater); }

        function tick(now) {
            requestAnimationFrame(tick);
            if (paused || busy || document.hidden || window.nvGestureActive) { last = now; return; }
            if (last === null) { last = now; return; }
            if (now - last < FRAME_MS) return;
            var dt = Math.min(0.1, (now - last) / 1000);
            last = now;

            var cam = gd._fullLayout.scene.camera;
            if (!cam || !cam.eye) return;
            var e = cam.eye, c = cam.center || {x: 0, y: 0, z: 0};
            var md = (gd.layout && gd.layout.meta && gd.layout.meta.nv_dirs) || null;
            var u = md ? {x: md.superior[0], y: md.superior[1], z: md.superior[2]}
                       : (cam.up || {x: 0, y: 0, z: 1});
            var ul = Math.sqrt(u.x * u.x + u.y * u.y + u.z * u.z) || 1;
            var kx = u.x / ul, ky = u.y / ul, kz = u.z / ul;

            // Rodrigues rotation of (eye - center) around the up axis, so it
            // still orbits correctly after the user has panned the view.
            var vx = e.x - c.x, vy = e.y - c.y, vz = e.z - c.z;
            var a = SPEED * dt, cs = Math.cos(a), sn = Math.sin(a);
            var dot = kx * vx + ky * vy + kz * vz;
            var cx = ky * vz - kz * vy, cy = kz * vx - kx * vz, cz = kx * vy - ky * vx;
            var nx = vx * cs + cx * sn + kx * dot * (1 - cs);
            var ny = vy * cs + cy * sn + ky * dot * (1 - cs);
            var nz = vz * cs + cz * sn + kz * dot * (1 - cs);

            busy = true;
            Plotly.relayout(gd, {
                'scene.camera.eye': {x: c.x + nx, y: c.y + ny, z: c.z + nz}
            }).then(function() { busy = false; }, function() { busy = false; });
        }
        requestAnimationFrame(tick);
    }
    setup();
})();
</script>
""" % {"speed": speed, "resume": int(AUTO_ROTATE_RESUME_MS), "frame": 1000.0 / max(1, AUTO_ROTATE_FPS)}
    if "</body>" in content:
        content = content.replace("</body>", f"{script_block}</body>", 1)
    else:
        content += script_block
    try:
        path.write_text(content, encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Webcam gesture control: Python sends small camera deltas into the page and
# this hook orbits / tilts / zooms the Plotly camera. It does nothing unless
# the Gesture button in the island is switched on.
# ---------------------------------------------------------------------------
_GESTURE_HOOK_JS = """
<script>
(function() {
    function rot(v, k, a) {
        var cs = Math.cos(a), sn = Math.sin(a);
        var dot = k.x*v.x + k.y*v.y + k.z*v.z;
        var cx = k.y*v.z - k.z*v.y, cy = k.z*v.x - k.x*v.z, cz = k.x*v.y - k.y*v.x;
        return {x: v.x*cs + cx*sn + k.x*dot*(1-cs),
                y: v.y*cs + cy*sn + k.y*dot*(1-cs),
                z: v.z*cs + cz*sn + k.z*dot*(1-cs)};
    }
    function unit(v) { var l = Math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z) || 1;
                       return {x: v.x/l, y: v.y/l, z: v.z/l}; }
    var busy = false, holdTimer = null;
    window.nvGestureActive = false;
    window.nvGesture = function(dAz, dEl, zoom) {
        var gd = document.querySelector('.plotly-graph-div');
        if (!gd || !window.Plotly || !gd._fullLayout || !gd._fullLayout.scene || busy) return;
        window.nvGestureActive = true;               // pauses auto-rotate
        clearTimeout(holdTimer);
        holdTimer = setTimeout(function() { window.nvGestureActive = false; }, %(resume)d);
        var cam = gd._fullLayout.scene.camera;
        if (!cam || !cam.eye) return;
        var c = cam.center || {x: 0, y: 0, z: 0};
        var md = gd.layout && gd.layout.meta && gd.layout.meta.nv_dirs;
        var k = unit(md ? {x: md.superior[0], y: md.superior[1], z: md.superior[2]}
                        : (cam.up || {x: 0, y: 0, z: 1}));
        var v = {x: cam.eye.x - c.x, y: cam.eye.y - c.y, z: cam.eye.z - c.z};
        v = rot(v, k, dAz);
        var vl = Math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z) || 1;
        var rr = {x: k.y*v.z - k.z*v.y, y: k.z*v.x - k.x*v.z, z: k.x*v.y - k.y*v.x};
        if (Math.sqrt(rr.x*rr.x + rr.y*rr.y + rr.z*rr.z) < 1e-6 * vl) {
            var cu = cam.up || {x: 0, y: 1, z: 0};   // looking straight down: tilt about the screen's side axis
            rr = {x: k.y*cu.z - k.z*cu.y, y: k.z*cu.x - k.x*cu.z, z: k.x*cu.y - k.y*cu.x};
        }
        var r = unit(rr);
        var t = rot(v, r, dEl);
        var tl = Math.sqrt(t.x*t.x + t.y*t.y + t.z*t.z) || 1;
        var cv = Math.abs((k.x*v.x + k.y*v.y + k.z*v.z) / vl);
        var ct = Math.abs((k.x*t.x + k.y*t.y + k.z*t.z) / tl);
        if (ct < 0.97 || ct < cv) v = t;
        var len = Math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z) || 1;
        var s = Math.min(5.0, Math.max(0.6, len * zoom)) / len;
        busy = true;
        Plotly.relayout(gd, {'scene.camera.eye': {x: c.x + v.x*s, y: c.y + v.y*s, z: c.z + v.z*s}})
            .then(function() { busy = false; }, function() { busy = false; });
    };
})();
</script>
""" % {"resume": int(AUTO_ROTATE_RESUME_MS)}


def _inject_gesture_hook(html_path: str):
    try:
        path = Path(html_path)
        content = path.read_text(encoding="utf-8")
    except OSError:
        return
    if "</body>" in content:
        content = content.replace("</body>", f"{_GESTURE_HOOK_JS}</body>", 1)
    else:
        content += _GESTURE_HOOK_JS
    try:
        path.write_text(content, encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 3D page tools, injected into every 3D page and called from Python:
#   nvSetSlices([sag, cor, ax])  move the three slice planes (display-grid indices)
#   nvFocus(rank)                fade every tumour except `rank` (0 = show all)
#   nvFlyTo('front'|'side'|'top') animate the camera to a standard view
# Everything they need (trace indices, bounds, scan orientation) is written
# into layout.meta by build_3d_figures().
# ---------------------------------------------------------------------------
_BRAIN_TOOLS_JS = """
<script>
(function() {
    function graph() {
        var gd = document.querySelector('.plotly-graph-div');
        if (!gd || !window.Plotly || !gd._fullLayout || !gd._fullLayout.scene || !gd.layout) return null;
        return gd;
    }
    function whenReady(fn, tries) {
        var gd = graph();
        if (gd) { fn(gd, gd.layout.meta || {}); return; }
        if ((tries || 0) < 60) setTimeout(function() { whenReady(fn, (tries || 0) + 1); }, 100);
    }
    var AX = ['x', 'y', 'z'];
    function quad(a, p, B) {
        var b = (a + 1) % 3, c = (a + 2) % 3;
        var pts = [[B[b][0], B[c][0]], [B[b][1], B[c][0]], [B[b][1], B[c][1]], [B[b][0], B[c][1]]];
        var out = {x: [], y: [], z: []};
        pts.forEach(function(q) {
            var v = [0, 0, 0]; v[a] = p; v[b] = q[0]; v[c] = q[1];
            out.x.push(v[0]); out.y.push(v[1]); out.z.push(v[2]);
        });
        return out;
    }
    window.nvSetSlices = function(idx) {
        whenReady(function(gd, m) {
            var P = m.nv_planes, B = m.nv_bounds, S = m.nv_scale;
            if (!P || !B || !S) return;
            var upd = {x: [], y: [], z: []}, ids = [];
            for (var a = 0; a < 3; a++) {
                var pair = P[String(a)];
                if (!pair || idx[a] === null || idx[a] === undefined) continue;
                var q = quad(a, idx[a] * S[a], B);
                upd.x.push(q.x); upd.y.push(q.y); upd.z.push(q.z); ids.push(pair[0]);
                upd.x.push(q.x.concat([q.x[0]])); upd.y.push(q.y.concat([q.y[0]]));
                upd.z.push(q.z.concat([q.z[0]])); ids.push(pair[1]);
            }
            if (ids.length) Plotly.restyle(gd, upd, ids);
        });
    };
    window.nvFocus = function(rank) {
        whenReady(function(gd, m) {
            var T = m.nv_tumours || [], ops = [], ids = [];
            T.forEach(function(t) { ids.push(t.i); ops.push(!rank || t.rank === rank ? 1.0 : 0.13); });
            (m.nv_brain || []).forEach(function(i) {
                ids.push(i); ops.push(rank ? (m.nv_shell_focus || 0.07) : (m.nv_shell || 0.16));
            });
            if (ids.length) Plotly.restyle(gd, {opacity: ops}, ids);
        });
    };
    window.nvFlyTo = function(name, wantDist) {
        whenReady(function(gd, m) {
            var D = m.nv_dirs; if (!D) return;
            var cam = gd._fullLayout.scene.camera, c = cam.center || {x: 0, y: 0, z: 0};
            var e0 = {x: cam.eye.x - c.x, y: cam.eye.y - c.y, z: cam.eye.z - c.z};
            var dist = wantDist || Math.sqrt(e0.x * e0.x + e0.y * e0.y + e0.z * e0.z) || 2;
            var dir = name === 'front' ? D.anterior : (name === 'side' ? D.left : D.superior);
            var up = name === 'top' ? D.anterior : D.superior;
            var e1 = {x: dir[0] * dist, y: dir[1] * dist, z: dir[2] * dist};
            var u0 = cam.up || {x: 0, y: 0, z: 1}, u1 = {x: up[0], y: up[1], z: up[2]};
            window.nvGestureActive = true;
            clearTimeout(window._nvFlyHold);
            var t0 = null, T = 700;
            function mix(a, b, s) { return {x: a.x + (b.x - a.x) * s, y: a.y + (b.y - a.y) * s, z: a.z + (b.z - a.z) * s}; }
            function step(now) {
                if (t0 === null) t0 = now;
                var k = Math.min(1, (now - t0) / T);
                var s = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
                var v = mix(e0, e1, s), l = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
                if (l < 1e-3) { v = mix(e0, e1, Math.min(1, s + 0.05)); l = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z) || 1; }
                var eye = {x: c.x + v.x / l * dist, y: c.y + v.y / l * dist, z: c.z + v.z / l * dist};
                Plotly.relayout(gd, {'scene.camera.eye': eye, 'scene.camera.up': mix(u0, u1, s)}).then(function() {
                    if (k < 1) requestAnimationFrame(step);
                    else window._nvFlyHold = setTimeout(function() { window.nvGestureActive = false; }, __RESUME__);
                });
            }
            requestAnimationFrame(step);
        });
    };
    window.nvActivePlane = function(axis) {
        whenReady(function(gd, m) {
            var P = m.nv_planes; if (!P) return;
            var base = m.nv_plane_opacity || 0.14, ops = [], widths = [], meshes = [], edges = [];
            for (var a = 0; a < 3; a++) {
                var pair = P[String(a)]; if (!pair) continue;
                var on = axis === null || axis === undefined || axis === a;
                meshes.push(pair[0]); ops.push(axis === null || axis === undefined ? base : (on ? 0.26 : 0.05));
                edges.push(pair[1]); widths.push(on ? (axis === null || axis === undefined ? 3 : 5) : 1.5);
            }
            if (meshes.length) Plotly.restyle(gd, {opacity: ops}, meshes);
            if (edges.length) Plotly.restyle(gd, {'line.width': widths}, edges);
        });
    };
})();
</script>
""".replace("__RESUME__", str(int(AUTO_ROTATE_RESUME_MS)))


def _inject_brain_tools(html_path: str):
    try:
        path = Path(html_path)
        content = path.read_text(encoding="utf-8")
    except OSError:
        return
    if "</body>" in content:
        content = content.replace("</body>", f"{_BRAIN_TOOLS_JS}</body>", 1)
    else:
        content += _BRAIN_TOOLS_JS
    try:
        path.write_text(content, encoding="utf-8")
    except OSError:
        pass


class Panel3D(HivePanel):
    """
    A bordered panel that shows an interactive 3D Plotly figure (rotate,
    zoom, pan) via an embedded web view. Qt has no native Plotly renderer,
    so the figure is written to a temporary standalone HTML file and loaded
    into a QWebEngineView — the same approach as save_figure_html() in the
    existing viewer3d.py, just pointed at a temp file instead of a
    user-chosen path.
    """

    MSG_NO_CASE = "Load an MRI folder to begin"
    MSG_READY = "Run segmentation to render"

    def __init__(self, title: str, min_height: int = 200, accent: str = ACCENT_TEAL,
                 backdrop: str = "brain", auto_rotate: bool = False,
                 meta: str = "", subtitle: str = "", view_toggle: bool = False):
        super().__init__()
        self._view_mode = "All"
        self._slices = [None, None, None]   # display-grid index per axis (sagittal, coronal, axial)
        self._focus = 0
        self._active_plane = None           # slice plane highlighted while gesture-slicing
        self._slice_timer = QTimer(self)
        self._slice_timer.setSingleShot(True)
        self._slice_timer.setInterval(33)    # coalesce slider drags to ~30 updates/s
        self._slice_timer.timeout.connect(self._push_slices)
        self.presets = None
        self.accent = accent
        self.backdrop = backdrop
        self.auto_rotate = auto_rotate
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

        self.view_toggle = None
        if view_toggle:
            layout.setContentsMargins(18, 16, 18, 14)
            header = QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 6)
            header.setSpacing(8)
            text_col = QVBoxLayout()
            text_col.setContentsMargins(0, 0, 0, 0)
            text_col.setSpacing(1)
            self.title_label = QLabel(title)
            self.title_label.setStyleSheet(
                f"border: none; background: transparent; color: {TEXT_LIGHT}; font-size: 15px; font-weight: 600;"
            )
            text_col.addWidget(self.title_label)
            sub = QLabel(subtitle)
            sub.setStyleSheet(f"border: none; background: transparent; color: {TEXT_MUTED}; font-size: 12px;")
            text_col.addWidget(sub)
            header.addLayout(text_col)
            header.addStretch(1)
            self.view_toggle = _SegmentedToggle(("All", "Tumour", "Brain"))
            self.view_toggle.changed.connect(self.set_view_mode)
            header.addWidget(self.view_toggle, 0, Qt.AlignVCenter)
            self.presets = _PresetChips()
            self.presets.chosen.connect(self.fly_to)
            header.addWidget(self.presets, 0, Qt.AlignVCenter)
            self.meta_label = None
        else:
            header, self.title_label, self.meta_label = _card_head(title, meta)
        self.maximize_btn = _fullscreen_button("Maximize")
        header.addWidget(self.maximize_btn, 0, Qt.AlignVCenter)
        layout.addLayout(header)

        view_host = QWidget()
        view_stack = QStackedLayout(view_host)
        view_stack.setStackingMode(QStackedLayout.StackAll)

        self.web_view = QWebEngineView()
        self.web_view.setStyleSheet(f"background-color: {BG_PANEL}; border: none;")
        self.web_view.page().setBackgroundColor(Qt.transparent)
        # Drops fall through to the window (open a patient folder anywhere).
        self.web_view.setAcceptDrops(False)
        self.web_view.loadFinished.connect(lambda _ok: self._on_page_loaded())
        # Empty state is drawn inside the page itself, so the hive backdrop
        # is visible before anything has been segmented.
        self._empty_message = self.MSG_NO_CASE
        self._has_figure = False
        self.web_view.setHtml(_hive_empty_page(self.backdrop, self._empty_message))
        view_stack.addWidget(self.web_view)
        self._view_stack = view_stack
        self._build_camera_preview()
        layout.addWidget(view_host, stretch=1)
        self.legend = None
        if view_toggle:
            self.legend = _LegendPill()
            self.legend.focus_changed.connect(self.set_focus)
            layout.addSpacing(6)
            layout.addWidget(self.legend, 0, Qt.AlignHCenter)

        self._layout = layout
        self._temp_files = []  # keep references so temp files aren't GC'd/deleted early

    # ---- webcam gesture preview (hidden until Gesture is switched on) ----
    def _build_camera_preview(self):
        """Small dark webcam card in the bottom-right corner of the 3D view."""
        self.cam_layer = QWidget()
        self.cam_layer.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.cam_layer.setStyleSheet("background: transparent;")
        col = QVBoxLayout(self.cam_layer)
        col.setContentsMargins(10, 10, 10, 10)
        col.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        card = QFrame()
        card.setObjectName("camCard")
        card.setStyleSheet(
            f"QFrame#camCard {{ background: {ISLAND_BG}; border-radius: 12px; }}"
            "QFrame#camCard QLabel { background: transparent; border: none; }"
        )
        inner = QVBoxLayout(card)
        inner.setContentsMargins(6, 6, 6, 6)
        inner.setSpacing(4)
        self.cam_preview = QLabel()
        self.cam_preview.setFixedSize(176, 132)
        self.cam_preview.setAlignment(Qt.AlignCenter)
        self.cam_state = QLabel("Show an open palm")
        self.cam_state.setAlignment(Qt.AlignCenter)
        self.cam_state.setStyleSheet(f"color: {ISLAND_TEXT}; font-size: 10px; font-weight: 600;")
        inner.addWidget(self.cam_preview)
        inner.addWidget(self.cam_state)
        row.addWidget(card)
        col.addLayout(row)
        self.cam_layer.hide()
        self._view_stack.addWidget(self.cam_layer)

    def show_camera_preview(self, on: bool):
        self.cam_layer.setVisible(on)
        if on:
            self.cam_layer.raise_()
        else:
            self.cam_preview.clear()

    def set_camera_frame(self, img: QImage):
        if self.cam_layer.isVisible():
            self.cam_preview.setPixmap(QPixmap.fromImage(img).scaled(
                self.cam_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_camera_state(self, text: str):
        self.cam_state.setText(text)

    def send_gesture(self, d_az: float, d_el: float, zoom: float):
        if self._has_figure:
            self.web_view.page().runJavaScript(
                f"window.nvGesture&&window.nvGesture({d_az:.5f},{d_el:.5f},{zoom:.5f});"
            )

    def set_empty_message(self, message: str):
        """Hint shown while the panel is empty; refreshed immediately if empty."""
        self._empty_message = message
        if not self._has_figure:
            self.web_view.setHtml(_hive_empty_page(self.backdrop, message))

    def set_figure(self, fig: go.Figure):
        self._has_figure = True
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
        _inject_hive_backdrop(tmp.name, self.backdrop)
        _inject_middle_click_pan(tmp.name)
        _inject_gesture_hook(tmp.name)
        _inject_brain_tools(tmp.name)
        if self.auto_rotate:
            _inject_auto_rotate(tmp.name)
        self._temp_files.append(tmp.name)
        self.web_view.load(QUrl.fromLocalFile(tmp.name))
        QTimer.singleShot(0, self._sync_hive_overlay)

    def clear_figure(self):
        """Restore the empty state when there is no tumor mask to render."""
        self._has_figure = False
        self.web_view.setHtml(_hive_empty_page(self.backdrop, self._empty_message))
        self._sync_hive_overlay()

    def set_maximized(self, is_max: bool):
        self.maximize_btn.setText("🗗" if is_max else "⛶")
        self.maximize_btn.setToolTip("Restore" if is_max else "Maximize")
        self._sync_hive_overlay()

    def set_legend(self, items):
        if self.legend is not None:
            self.legend.set_items(items)

    def _on_page_loaded(self):
        """A new 3D page finished loading: re-apply view mode, planes and focus."""
        self._apply_view_mode()
        if self._has_figure:
            self._push_slices()
            if self._focus:
                self._run_js(f"window.nvFocus&&window.nvFocus({int(self._focus)});")
            if self._active_plane is not None:
                self.set_active_plane(self._active_plane)

    def _run_js(self, js: str):
        if self._has_figure:
            self.web_view.page().runJavaScript(js)

    def set_slice(self, axis: int, index: int):
        """Move one slice plane (called as a 2D slider moves)."""
        self._slices[axis] = int(index)
        if not self._slice_timer.isActive():
            self._slice_timer.start()

    def _push_slices(self):
        vals = ",".join("null" if v is None else str(int(v)) for v in self._slices)
        self._run_js(f"window.nvSetSlices&&window.nvSetSlices([{vals}]);")

    def set_active_plane(self, axis: Optional[int]):
        """Draw one slice plane stronger and fade the other two (None = all equal)."""
        self._active_plane = axis
        self._run_js(f"window.nvActivePlane&&window.nvActivePlane({'null' if axis is None else int(axis)});")

    def set_focus(self, rank: int):
        """Highlight one tumour (rank 1 = largest) and fade the rest; 0 = all."""
        self._focus = int(rank)
        self._run_js(f"window.nvFocus&&window.nvFocus({int(rank)});")

    def fly_to(self, view: str, distance: Optional[float] = None):
        """Animate the camera to 'front', 'side' or 'top' (optionally at a set zoom)."""
        if view in ("front", "side", "top"):
            d = "" if distance is None else f",{float(distance):.3f}"
            self._run_js(f"window.nvFlyTo&&window.nvFlyTo('{view}'{d});")

    def set_view_mode(self, mode: str):
        """All / Tumour / Brain: toggles trace visibility inside the Plotly page.
        The brain trace indices are tagged in layout.meta by build_brain_figure."""
        self._view_mode = mode
        self._apply_view_mode()

    def _apply_view_mode(self):
        if not self._has_figure or self.view_toggle is None:
            return
        js = (
            "(function(){var gd=document.querySelector('.plotly-graph-div');"
            "if(!gd||!window.Plotly||!gd.data)return;"
            "var md=(gd.layout&&gd.layout.meta)||{};var b=md.nv_brain||[];"
            "var pl=[];var P=md.nv_planes||{};Object.keys(P).forEach(function(k){pl=pl.concat(P[k]);});"
            f"var mode='{self._view_mode}';"
            "var vis=gd.data.map(function(t,i){if(pl.indexOf(i)>=0)return true;var isB=b.indexOf(i)>=0;"
            "return mode==='All'?true:(mode==='Brain'?isB:!isB);});"
            "Plotly.restyle(gd,{visible:vis});})();"
        )
        self.web_view.page().runJavaScript(js)


def _style_ui_figure(fig: go.Figure) -> go.Figure:
    """Strip notebook chrome so the mesh fills the dark Qt panel."""
    fig.update_layout(
        title=None,
        width=None,
        height=None,
        margin=dict(l=0, r=0, t=0, b=0),
        # Transparent so the hive backdrop behind the page shows through.
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene_bgcolor="rgba(0,0,0,0)",
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


# ---------------------------------------------------------------------------
# 3D figures for the app: glass brain shell, smooth tumour meshes coloured
# exactly like the 2D overlays and the report, and three slice planes.
# Tweak the look with these numbers.
# ---------------------------------------------------------------------------
TUMOUR_SMOOTH_SIGMA = 1.2      # blur (in display voxels) before meshing; 0 = blocky voxel look
BRAIN_SHELL_OPACITY = 0.16     # glass shell opacity (0-1)
BRAIN_SHELL_FOCUS_OPACITY = 0.07   # shell opacity while one tumour is focused
SHELL_LIGHTING = dict(ambient=0.45, diffuse=0.5, specular=0.9, roughness=0.25, fresnel=2.6)
TUMOUR_LIGHTING = dict(ambient=0.55, diffuse=0.75, specular=0.55, roughness=0.35, fresnel=0.5)
SLICE_PLANE_COLORS = {0: "#12A594", 1: "#8E6CEF", 2: "#1F6FEB"}   # sagittal, coronal, axial
SLICE_PLANE_OPACITY = 0.14
_HOVER_LABEL = dict(bgcolor="#0A1224", bordercolor="#0A1224",
                    font=dict(family="Segoe UI, sans-serif", size=12, color="#FFFFFF"))


def _component_table(mask: np.ndarray, probs: Optional[np.ndarray], voxel_vol_cm3: float):
    """Connected tumour components ranked largest-first, each with its
    COMPONENT_PALETTE colour. Shared by the 2D views, the report and 3D, so the
    same tumour has the same colour everywhere. Returns (labels, entries, color_map)."""
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
            "confidence": float(probs[comp_mask].mean()) if probs is not None else None,
        })
    entries.sort(key=lambda e: e["volume_cm3"], reverse=True)
    color_map = {}
    for rank, entry in enumerate(entries):
        color_name, color_hex = COMPONENT_PALETTE[rank % len(COMPONENT_PALETTE)]
        entry["rank"] = rank + 1
        entry["color_name"] = color_name
        entry["color_hex"] = color_hex
        color_map[entry["label_id"]] = color_hex
    return labels, entries, color_map


def _anatomy_dirs(axcodes) -> dict:
    """Unit vectors (in scene x/y/z) pointing to the patient's left, front and
    top, from the scan's orientation codes (nibabel.aff2axcodes)."""
    codes = tuple(axcodes) if axcodes else ("L", "P", "S")   # BraTS default
    out = {}
    for key, (pos, neg) in (("left", ("L", "R")), ("anterior", ("A", "P")), ("superior", ("S", "I"))):
        v = [0.0, 0.0, 0.0]
        for a, code in enumerate(codes[:3]):
            if code == pos:
                v[a] = 1.0
            elif code == neg:
                v[a] = -1.0
        if not any(v):
            v[{"left": 0, "anterior": 1, "superior": 2}[key]] = 1.0
        out[key] = v
    return out


def _prepare_3d_labels(volume: np.ndarray, labels: Optional[np.ndarray], affine: np.ndarray):
    """Downsample volume + component labels for meshing. Also returns, per
    axis, the factor that turns a 2D slider index into a 3D scene coordinate."""
    display_shape = volume.shape
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).astype(np.float64)
    longest = float(max(volume.shape))
    factor = min(1.0, RENDER_MAX_DIM / longest) if longest else 1.0
    if factor < 0.99:
        new_shape = tuple(max(8, int(round(size * factor))) for size in volume.shape)
        zooms = [new / old for new, old in zip(new_shape, volume.shape)]
        volume = zoom(volume, zooms, order=1)
        if labels is not None:
            labels = zoom(labels.astype(np.int32), zooms, order=0)
        spacing = spacing / factor
    scale = [float(spacing[a] * (volume.shape[a] - 1) / max(1, display_shape[a] - 1)) for a in range(3)]
    return volume, labels, tuple(float(s) for s in spacing), scale


def _tumour_meshes(labels: np.ndarray, comps: list, scale: list) -> list:
    """One smooth surface per tumour, as plain dicts (reused by both figures).

    Meshed from the full display-grid labels, cropped to each tumour, so even
    small tumours survive (downsampling for the brain shell would erase them)
    and the surface is finer. `scale` maps a display-grid index to the 3D
    scene coordinate used by the brain shell and the slice planes."""
    out = []
    pad = 3
    sc = np.array(scale, dtype=np.float64)
    for c in comps:
        comp = labels == c["label_id"]
        hits = np.argwhere(comp)
        if hits.size == 0:
            continue
        lo = hits.min(axis=0)
        hi = hits.max(axis=0) + 1
        crop = comp[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
        vol = np.pad(crop, pad)   # so the surface closes around the tumour
        if TUMOUR_SMOOTH_SIGMA > 0:
            vol = gaussian_filter(vol, TUMOUR_SMOOTH_SIGMA)
        peak = float(vol.max())
        if peak <= 0:
            continue
        level = 0.5 if peak > 0.55 else peak * 0.5   # tiny tumours fade under the blur
        try:
            verts, faces, _, _ = marching_cubes(vol, level=level, spacing=tuple(sc))
        except (ValueError, RuntimeError):
            continue
        if len(faces) == 0:
            continue
        verts = verts + (lo - pad) * sc
        conf = c.get("confidence")
        conf_txt = "Refined with MedSAM2" if conf is None else f"Confidence  {conf * 100:.1f}%"
        loc = c.get("location") or ""
        out.append(dict(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=c["color_hex"], opacity=1.0, flatshading=False,
            lighting=TUMOUR_LIGHTING, lightposition=dict(x=1000, y=1500, z=2500),
            name=f"Tumour {c['rank']}", showlegend=False,
            hovertemplate=(f"<b>Tumour {c['rank']}</b><br>Volume  {c['volume_cm3']:.2f} cm\u00b3<br>"
                           f"{conf_txt}" + (f"<br>{loc}" if loc else "") + "<extra></extra>"),
            hoverlabel=_HOVER_LABEL,
            rank=c["rank"],
        ))
    return out


def _add_tumours(fig: go.Figure, meshes: list) -> list:
    """Append the tumour meshes; returns [{'i': trace index, 'rank': n}]."""
    tagged = []
    for m in meshes:
        kw = {k: v for k, v in m.items() if k != "rank"}
        fig.add_trace(go.Mesh3d(**kw))
        tagged.append({"i": len(fig.data) - 1, "rank": int(m["rank"])})
    return tagged


def _add_slice_planes(fig: go.Figure, volume_3d: np.ndarray, spacing: tuple, level: float) -> tuple:
    """Three translucent planes (sagittal, coronal, axial) with outlines.
    Returns (bounds, {axis: [mesh index, outline index]})."""
    hits = np.argwhere(volume_3d > level)
    if hits.size:
        lo, hi = hits.min(axis=0), hits.max(axis=0)
    else:
        lo, hi = np.zeros(3), np.array(volume_3d.shape) - 1
    bounds = []
    for a in range(3):
        a0, a1 = float(lo[a] * spacing[a]), float(hi[a] * spacing[a])
        m = (a1 - a0) * 0.04
        bounds.append([a0 - m, a1 + m])
    planes = {}
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        mid = (bounds[a][0] + bounds[a][1]) / 2
        pts = []
        for qb, qc in ((0, 0), (1, 0), (1, 1), (0, 1)):
            v = [0.0, 0.0, 0.0]
            v[a], v[b], v[c] = mid, bounds[b][qb], bounds[c][qc]
            pts.append(v)
        xs, ys, zs = zip(*pts)
        color = SLICE_PLANE_COLORS[a]
        fig.add_trace(go.Mesh3d(
            x=list(xs), y=list(ys), z=list(zs), i=[0, 0], j=[1, 2], k=[2, 3],
            color=color, opacity=SLICE_PLANE_OPACITY, flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0), hoverinfo="skip",
            showlegend=False, name=f"plane-{a}",
        ))
        mesh_i = len(fig.data) - 1
        fig.add_trace(go.Scatter3d(
            x=list(xs) + [xs[0]], y=list(ys) + [ys[0]], z=list(zs) + [zs[0]],
            mode="lines", line=dict(color=color, width=3), hoverinfo="skip",
            showlegend=False, name=f"plane-{a}-edge",
        ))
        planes[str(a)] = [mesh_i, len(fig.data) - 1]
    return bounds, planes


def build_3d_figures(display_volume: np.ndarray, mask: Optional[np.ndarray],
                     probs: Optional[np.ndarray], affine: np.ndarray, axcodes=None):
    """Both 3D figures for one case: (tumour figure or None, brain figure)."""
    voxel_vol_cm3 = float(np.prod(np.sqrt((affine[:3, :3] ** 2).sum(axis=0)))) / 1000.0
    labels, comps = None, []
    if mask is not None and np.any(mask > 0.5):
        labels, comps, _ = _component_table(mask, probs, voxel_vol_cm3)
        brain_mask = _brain_tissue_mask(display_volume, mask)
        for c in comps:
            loc = _rough_location(labels == c["label_id"], brain_mask, axcodes)
            if loc:
                side, ap, _si = loc
                c["location"] = ("Midline" if side == "Midline" else f"{side} hemisphere") + f" · {ap}"
    volume_3d, labels_3d, spacing, scale = _prepare_3d_labels(display_volume, labels, affine)
    meshes = _tumour_meshes(labels, comps, scale) if comps else []
    dirs = _anatomy_dirs(axcodes)

    # Brain: shell from the shared viewer, restyled as glass.
    brain_fig = show_volume_3d(volume_3d, None, show_brain=True, spacing=spacing,
                               title="3D brain", clean=True, show_legend=False)
    brain_idx = []
    if len(brain_fig.data):
        brain_fig.data[0].update(color=BRAIN_SHELL_COLOR, opacity=BRAIN_SHELL_OPACITY,
                                 lighting=SHELL_LIGHTING, lightposition=dict(x=1000, y=1500, z=2500),
                                 hoverinfo="skip")
        brain_idx = [0]
    level = float((brain_fig.layout.meta or {}).get("surface_level", brain_surface_level(volume_3d)))
    tumours = _add_tumours(brain_fig, meshes)
    bounds, planes = _add_slice_planes(brain_fig, volume_3d, spacing, level)
    brain_fig.update_layout(meta={
        "nv_brain": brain_idx, "nv_tumours": tumours, "nv_planes": planes,
        "nv_bounds": bounds, "nv_scale": scale, "nv_dirs": dirs,
        "nv_shell": BRAIN_SHELL_OPACITY, "nv_shell_focus": BRAIN_SHELL_FOCUS_OPACITY,
        "nv_plane_opacity": SLICE_PLANE_OPACITY,
    })
    _style_ui_figure(brain_fig)

    tumor_fig = None
    if meshes:
        tumor_fig = show_volume_3d(labels_3d.astype(np.float32), None, show_brain=False, level=0.5,
                                   spacing=spacing, title="3D tumor", clean=True, show_legend=False)
        t_tags = _add_tumours(tumor_fig, meshes)
        tumor_fig.update_layout(meta={"nv_brain": [], "nv_tumours": t_tags, "nv_dirs": dirs,
                                      "nv_shell": BRAIN_SHELL_OPACITY,
                                      "nv_shell_focus": BRAIN_SHELL_FOCUS_OPACITY})
        _style_ui_figure(tumor_fig)
    return tumor_fig, brain_fig


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
    return _style_ui_figure(fig)


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
    _tint_brain_shell(fig, has_mask=mask is not None)
    return _style_ui_figure(fig)


BRAIN_SHELL_COLOR = "#8FB4F0"   # soft blue brain surface on the white 3D card


def _trace_size(trace) -> int:
    try:
        x = getattr(trace, "x", None)
        return len(x) if x is not None else 0
    except Exception:
        return 0


def _tint_brain_shell(fig: go.Figure, has_mask: bool = True) -> None:
    """Find the brain surface (the largest trace; tumour meshes are far
    smaller), tint it soft blue, and record its index in layout.meta so the
    All / Tumour / Brain toggle knows which traces are which."""
    traces = list(fig.data)
    if not traces:
        return
    if has_mask and len(traces) > 1:
        brain_idx = [max(range(len(traces)), key=lambda i: _trace_size(traces[i]))]
    else:
        brain_idx = list(range(len(traces)))
    for i in brain_idx:
        t = traces[i]
        try:
            if t.type == "mesh3d":
                t.update(color=BRAIN_SHELL_COLOR)
                if getattr(t, "intensity", None) is not None:
                    t.update(colorscale=[[0, BRAIN_SHELL_COLOR], [1, BRAIN_SHELL_COLOR]], showscale=False)
            elif t.type in ("isosurface", "volume", "surface"):
                t.update(colorscale=[[0, BRAIN_SHELL_COLOR], [1, BRAIN_SHELL_COLOR]], showscale=False)
        except Exception:
            pass
    fig.update_layout(meta={"nv_brain": brain_idx})


def _slider_qss(color: str) -> str:
    """Slice slider whose handle and filled track use a slice-plane colour."""
    c = QColor(color)
    tint = f"rgba({c.red()}, {c.green()}, {c.blue()}, 0.45)"
    return (
        "QSlider { background: transparent; min-height: 18px; max-height: 18px; }"
        "QSlider::groove:horizontal { height: 4px; background: #E3ECFA; border-radius: 2px; }"
        f"QSlider::sub-page:horizontal {{ background: {tint}; border-radius: 2px; }}"
        f"QSlider::handle:horizontal {{ background: #ffffff; border: 2px solid {color}; "
        "width: 10px; height: 10px; margin: -5px 0; border-radius: 7px; }"
        "QSlider::handle:horizontal:disabled { background: #ffffff; border: 2px solid #D5DDEA; }"
    )


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
        self._axcodes: Optional[tuple] = None   # native orientation, for rough location
        self._committed_mask: Optional[np.ndarray] = None
        self._committed_probs: Optional[np.ndarray] = None
        self._preview_mask: Optional[np.ndarray] = None
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

        # ---------------- Dynamic Island (header actions) -------------------
        # Old names (run_seg_btn, clear_btn, busy_bar) are kept so the rest of
        # the window logic is unchanged.
        self.island = DynamicIsland()
        self.run_seg_btn = self.island.run_btn
        self.clear_btn = self.island.clear_btn
        self.run_seg_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.run_seg_btn.clicked.connect(self.on_run_segmentation)
        self.clear_btn.clicked.connect(self.on_clear_views)
        self.island.open_clicked.connect(self._browse_folder)
        self.island.refine_clicked.connect(self._enter_sam_mode)
        self.island.sam_plane_chosen.connect(self._choose_sam_plane)
        self.island.sam_action.connect(self._on_island_sam_action)
        # Webcam gesture control (off until the island's Gesture button is clicked).
        self.island.gesture_toggled.connect(self._toggle_gesture)
        self._gesture = None
        self._gesture_acc = [0.0, 0.0, 1.0]
        self._gesture_timer = QTimer(self)
        self._gesture_timer.setInterval(33)          # ~30 camera updates per second
        self._gesture_timer.timeout.connect(self._flush_gesture)
        self._sam_panel = None
        self.busy_bar = _IslandBusyProxy(self.island)
        self.setAcceptDrops(True)   # drop a patient folder anywhere on the window

        # ---------------- Left column: 2D slices ------------------------------
        # axis 0 = L-R (sagittal), axis 1 = A-P (coronal), axis 2 = S-I (axial)
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(SPACING + 4)
        self.sagittal_panel = SlicePanel("Sagittal", axis=0, min_height=150)
        self.axial_panel = SlicePanel("Axial", axis=2, min_height=150)
        self.coronal_panel = SlicePanel("Coronal", axis=1, min_height=150)
        for slice_panel in (self.sagittal_panel, self.axial_panel, self.coronal_panel):
            left.addWidget(slice_panel, stretch=1)
        self.left_container = QWidget()
        self.left_container.setLayout(left)
        self.left_container.setMinimumWidth(230)
        self.left_container.setMaximumWidth(LEFT_COL_MAX)

        # ---------------- Centre: large 3D brain ------------------------------
        center = QVBoxLayout()
        center.setContentsMargins(0, 0, 0, 0)
        self.brain_3d_panel = Panel3D(
            "3D Brain", min_height=300, accent=ACCENT, backdrop="brain", auto_rotate=True,
            subtitle="Drag to rotate", view_toggle=True,
        )
        center.addWidget(self.brain_3d_panel, 1)
        # 2D sliders drive the 3D slice planes; each slider handle takes its plane's colour.
        for slice_panel in (self.sagittal_panel, self.axial_panel, self.coronal_panel):
            slice_panel.slider.setStyleSheet(_slider_qss(SLICE_PLANE_COLORS[slice_panel.axis]))
            slice_panel.slider.valueChanged.connect(
                lambda value, a=slice_panel.axis: self.brain_3d_panel.set_slice(a, value))
        self.center_container = QWidget()
        self.center_container.setLayout(center)

        # ---------------- Right column: tumour, record, report ----------------
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(SPACING + 4)
        self.tumor_3d_panel = Panel3D("3D Tumour", min_height=150, accent=ACCENT, backdrop="tumor",
                                      meta="isolated")
        self.brain_3d_panel.legend.focus_changed.connect(self.tumor_3d_panel.set_focus)
        self.patient_info_panel = PatientInfoPanel()
        self.patients_record_panel = PatientRecordPanel()
        self.summary_panel = self.patients_record_panel.summary
        self.survival_panel = SurvivalDaysPanel()
        self.credit = SquadCredit()
        right.addWidget(self.tumor_3d_panel, stretch=3)
        right.addWidget(self.patient_info_panel)
        right.addWidget(self.patients_record_panel, stretch=4)
        right.addWidget(self.survival_panel)
        right.addWidget(self.credit)
        self.right_container = QWidget()
        self.right_container.setLayout(right)
        self.right_container.setMinimumWidth(280)
        self.right_container.setMaximumWidth(RIGHT_COL_MAX)

        # ---------------- Assemble ------------------------------------------
        body = QHBoxLayout()
        body.setSpacing(16)
        body.setContentsMargins(20, 0, 20, 20)
        body.addWidget(self.left_container, stretch=0)
        body.addWidget(self.center_container, stretch=1)
        body.addWidget(self.right_container, stretch=0)

        page = QVBoxLayout(shell)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(0)
        self.header = AppHeader(self.island)
        page.addWidget(self.header)
        page.addLayout(body, stretch=1)
        self.setCentralWidget(shell)

        # Webcam bubble beside the island (hidden until Gesture is switched on).
        self.camera_bubble = CameraBubble(shell)
        self.gesture_guide = GestureGuideCard(self)   # floating window above the web views
        self.header.on_island_moved = self._place_camera_bubble

        self._clear_sync_timer = QTimer(self)
        self._clear_sync_timer.timeout.connect(self._sync_island_buttons)
        self._clear_sync_timer.start(250)

        # ---------------- Maximize / restore wiring --------------------------
        self.center_panels = [self.sagittal_panel, self.axial_panel, self.coronal_panel]  # the 2D panels
        self.right_panels = [self.tumor_3d_panel, self.brain_3d_panel]                     # the 3D panels
        self.all_panels = self.center_panels + self.right_panels
        self._column_members = {
            self.left_container: list(self.center_panels),
            self.center_container: [self.brain_3d_panel],
            self.right_container: [self.tumor_3d_panel, self.patient_info_panel,
                                   self.patients_record_panel, self.survival_panel, self.credit],
        }
        self._maximized_panel = None

        for panel in self.all_panels:
            panel.maximize_btn.clicked.connect(lambda checked=False, p=panel: self.toggle_maximize(p))
        for panel in self.center_panels:
            panel.refinement_requested.connect(self.on_run_sam_refinement)
            panel.accept_requested.connect(self.on_accept_sam_refinement)
            panel.discard_requested.connect(self.on_discard_sam_refinement)

    # ---------------- Webcam gesture control ----------------------------------
    _GESTURE_LABELS = {
        "rotate": "\u270b Rotating", "zoom": "\U0001F90F Zooming",
        "hold": "\u270a Holding", "none": "Show an open palm",
        "slice": "\u261d Point up / down \u00b7 hold still = next plane", "peace": "\u270c View mode",
        "thumbs": "\U0001F44D Next view", "three": "Next tumour",
    }
    _VIEW_ORDER = ("All", "Tumour", "Brain")
    _VIEW_MESSAGES = {"All": "View: Brain + tumours", "Tumour": "View: Tumour only", "Brain": "View: Brain only"}
    _PRESET_ORDER = ("front", "side", "top")
    _SLICE_ORDER = (2, 1, 0)                     # axial -> coronal -> sagittal
    _PLANE_NAMES = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
    RESET_CAMERA_DISTANCE = 2.4      # zoom level the palm-still reset returns to

    def _toggle_gesture(self, on: bool):
        if not on:
            self._stop_gesture()
            return
        # Always load the gesture_camera.py that sits next to this file, never
        # an older copy elsewhere on sys.path (the project root comes first).
        import importlib.util
        cam_file = Path(__file__).resolve().parent / "gesture_camera.py"
        try:
            spec = importlib.util.spec_from_file_location("nv_gesture_camera", cam_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            GestureCamera = module.GestureCamera
            if not hasattr(GestureCamera, "command") or not hasattr(module, "MODEL_URL"):
                raise ImportError("old version")
        except (FileNotFoundError, ImportError, AttributeError) as exc:
            self.island.gesture_btn.setChecked(False)
            QMessageBox.warning(
                self, "Gesture control",
                "Could not load the latest gesture_camera.py.\n\n"
                f"Expected it here:\n{cam_file}\n\n"
                "Put the newest gesture_camera.py in that exact folder."
                + ("" if isinstance(exc, FileNotFoundError) else "\n(The file there is an old version.)"),
            )
            return
        self._gesture = GestureCamera()
        self._gesture.gesture_delta.connect(self._on_gesture_delta)
        self._gesture.frame_ready.connect(self.camera_bubble.set_frame)
        self._gesture.state_changed.connect(
            lambda s: self.camera_bubble.set_state(self._GESTURE_LABELS.get(s, s)))
        self._gesture.command.connect(self._on_gesture_command)
        self._gesture.hold_progress.connect(self.camera_bubble.set_progress)
        self._gesture.slice_delta.connect(self._on_gesture_slice)
        self._gesture.failed.connect(self._on_gesture_failed)
        self._gesture.start()
        self._gesture_timer.start()
        self._place_camera_bubble()
        self.camera_bubble.pop_in()
        self.brain_3d_panel.set_active_plane(self._gesture_axis())
        if not getattr(self, "_guide_shown", False):   # first time this run: show the guide
            self._guide_shown = True
            self.gesture_guide.pop_in()

    def _gesture_confirm(self, message: str):
        """Tick in the camera bubble + a short message in the island."""
        self.camera_bubble.flash_done()
        if not self._busy and self.island.mode != "sam":
            self.island.show_done(message, hold_ms=1400)

    def _on_gesture_command(self, name: str):
        brain, tumour = self.brain_3d_panel, self.tumor_3d_panel
        if name == "view":
            cur = brain._view_mode if brain._view_mode in self._VIEW_ORDER else "All"
            nxt = self._VIEW_ORDER[(self._VIEW_ORDER.index(cur) + 1) % len(self._VIEW_ORDER)]
            if brain.view_toggle is not None:
                brain.view_toggle.select(nxt)
            else:
                brain.set_view_mode(nxt)
            self._gesture_confirm(self._VIEW_MESSAGES[nxt])
        elif name == "preset":
            self._preset_i = (getattr(self, "_preset_i", -1) + 1) % len(self._PRESET_ORDER)
            view = self._PRESET_ORDER[self._preset_i]
            for panel in (brain, tumour):
                panel.fly_to(view)
            self._gesture_confirm(f"{view.title()} view")
        elif name == "focus":
            rank, n = brain.legend.focus_next() if brain.legend is not None else (0, 0)
            self._gesture_confirm("No tumours to focus" if n == 0
                                  else ("Showing all tumours" if rank == 0 else f"Focus: Tumour {rank}"))
        elif name == "plane":
            order = self._SLICE_ORDER
            axis = order[(order.index(self._gesture_axis()) + 1) % len(order)]
            self._slice_axis = axis
            self._slice_acc = 0.0
            brain.set_active_plane(axis)
            panel = self._panel_for_axis(axis)
            self.camera_bubble.set_state(
                f"\u261d {self._PLANE_NAMES[axis]} slice {panel.slider.value()} / {panel.slider.maximum()}")
            self._gesture_confirm(f"Slicing: {self._PLANE_NAMES[axis]}")
        elif name == "reset":
            if brain.view_toggle is not None:
                brain.view_toggle.select("All")
            if brain.legend is not None:
                brain.legend.clear_focus()
            self._preset_i = 0
            for panel in (brain, tumour):
                panel.fly_to("front", self.RESET_CAMERA_DISTANCE)
            self._gesture_confirm("View reset")

    def _gesture_axis(self) -> int:
        return getattr(self, "_slice_axis", 2)

    def _panel_for_axis(self, axis: int):
        return {0: self.sagittal_panel, 1: self.coronal_panel, 2: self.axial_panel}[axis]

    def _on_gesture_slice(self, delta: float):
        """Pointing finger moves the active slice (and its plane in 3D)."""
        axis = self._gesture_axis()
        slider = self._panel_for_axis(axis).slider
        if not slider.isEnabled():
            self.camera_bubble.set_state("Load a scan to slice")
            return
        self._slice_acc = getattr(self, "_slice_acc", 0.0) + delta
        step = int(self._slice_acc)
        if step:
            self._slice_acc -= step
            slider.setValue(max(slider.minimum(), min(slider.maximum(), slider.value() + step)))
        self.camera_bubble.set_state(
            f"\u261d {self._PLANE_NAMES[axis]} slice {slider.value()} / {slider.maximum()}")

    def moveEvent(self, event):
        super().moveEvent(event)
        if getattr(self, "gesture_guide", None) is not None and self.gesture_guide.isVisible():
            self._place_camera_bubble()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, "gesture_guide", None) is not None and self.gesture_guide.isVisible():
            self._place_camera_bubble()

    def changeEvent(self, event):
        super().changeEvent(event)
        guide = getattr(self, "gesture_guide", None)
        if guide is not None and self.isMinimized():
            guide.pop_out()

    def _place_camera_bubble(self):
        """Sit the camera bubble just right of the island, top edges aligned."""
        bubble = getattr(self, "camera_bubble", None)
        if bubble is None:
            return
        island, header = self.island, self.header
        m = CameraBubble.MARGIN
        x = header.x() + island.x() + island.width() + 10 - m
        y = header.y() + island.y() - m
        x = min(x, self.centralWidget().width() - bubble.width() - 4)
        bubble.move(x, y)
        bubble.raise_()
        guide = getattr(self, "gesture_guide", None)
        if guide is not None:
            # Right edge lined up with the bubble's, just below it.
            gx = x + m + CameraBubble.W - GestureGuideCard.W - GestureGuideCard.MARGIN
            gx = max(4, min(gx, self.centralWidget().width() - guide.width() - 4))
            gy = y + m + CameraBubble.H + 8 - GestureGuideCard.MARGIN
            # The guide is a floating window, so place it in screen coordinates.
            guide.move(self.centralWidget().mapToGlobal(QPoint(int(gx), int(gy))))
            guide.raise_()

    def _stop_gesture(self):
        self._gesture_timer.stop()
        if self._gesture is not None:
            self._gesture.stop()
            self._gesture = None
        self._gesture_acc = [0.0, 0.0, 1.0]
        self.camera_bubble.set_progress(0.0)
        self.camera_bubble.pop_out()
        self.gesture_guide.pop_out()
        self.brain_3d_panel.set_active_plane(None)

    def _on_gesture_failed(self, message: str):
        self.island.gesture_btn.setChecked(False)    # also stops the camera
        QMessageBox.warning(self, "Gesture control", message)

    def _on_gesture_delta(self, d_az: float, d_el: float, zoom: float):
        acc = self._gesture_acc
        acc[0] += d_az
        acc[1] += d_el
        acc[2] *= zoom

    def _flush_gesture(self):
        d_az, d_el, zoom = self._gesture_acc
        if d_az == 0.0 and d_el == 0.0 and zoom == 1.0:
            return
        self._gesture_acc = [0.0, 0.0, 1.0]
        for panel in (self.brain_3d_panel, self.tumor_3d_panel):   # both 3D views move together
            panel.send_gesture(d_az, d_el, zoom)

    def closeEvent(self, event):
        self._stop_gesture()      # release the webcam when the app closes
        super().closeEvent(event)

    def _sync_island_buttons(self):
        self.clear_btn.setEnabled(self.current_folder is not None and not self._busy)
        self.island.open_btn.setEnabled(not self._busy)
        sam_ready = any(p._sam_enabled for p in self.center_panels)
        self.island.refine_btn.setEnabled(sam_ready and not self._busy)
        if self.island.mode != "sam":
            return
        p = self._sam_panel
        if p is None or not p._sam_enabled:
            self._exit_sam_mode()
            return
        count = len(p._sam_points) + int(p._sam_box is not None)
        if p._sam_preview_available:
            hint = "Preview ready"
        elif count:
            hint = f"{count} prompt{'s' if count != 1 else ''}"
        else:
            hint = "Click or drag on the slice"
        self.island.update_sam(hint, {
            "undo": p.sam_undo_btn.isEnabled(),
            "clear": p.sam_clear_btn.isEnabled(),
            "preview": p.sam_run_btn.isEnabled(),
            "accept": p.sam_accept_btn.isEnabled(),
            "discard": p.sam_discard_btn.isEnabled(),
            "done": not p._sam_preview_available and not self._busy,
        }, p._sam_preview_available)

    # ---------------- MedSAM2 via the island -----------------------------------
    def _panel_for_plane(self, plane: str):
        return {"Sagittal": self.sagittal_panel, "Axial": self.axial_panel,
                "Coronal": self.coronal_panel}.get(plane, self.axial_panel)

    def _enter_sam_mode(self):
        if self._busy or not any(p._sam_enabled for p in self.center_panels):
            return
        self._choose_sam_plane("Axial", enter=True)

    def _choose_sam_plane(self, plane: str, enter: bool = False):
        panel = self._panel_for_plane(plane)
        current = self._sam_panel
        if current is not None and current is not panel and current._sam_preview_available:
            self.island.set_sam_plane([k for k in ("Sagittal", "Axial", "Coronal")
                                       if self._panel_for_plane(k) is current][0])
            return
        for p in self.center_panels:
            if p is not panel and p.sam_refine_btn.isChecked():
                p.sam_refine_btn.setChecked(False)
        if not panel.sam_refine_btn.isChecked():
            panel.sam_refine_btn.setChecked(True)
        self._sam_panel = panel
        if enter:
            self.island.enter_sam(plane)
        else:
            self.island.set_sam_plane(plane)
        self._sync_island_buttons()

    def _exit_sam_mode(self):
        for p in self.center_panels:
            if p.sam_refine_btn.isChecked():
                p.sam_refine_btn.setChecked(False)
        self._sam_panel = None
        self.island.exit_sam()

    def _on_island_sam_action(self, key: str):
        if key == "done":
            self._exit_sam_mode()
            return
        p = self._sam_panel
        if p is None:
            return
        btn = {"undo": p.sam_undo_btn, "clear": p.sam_clear_btn, "preview": p.sam_run_btn,
               "accept": p.sam_accept_btn, "discard": p.sam_discard_btn}.get(key)
        if btn is not None and btn.isEnabled():
            btn.click()
        self._sync_island_buttons()

    # ---------------- Drag a patient folder onto the window --------------------
    def dragEnterEvent(self, event):
        if not self._busy and DropVolumeZone._path_from_mime(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        folder = DropVolumeZone._path_from_mime(event)
        if folder is None or self._busy:
            event.ignore()
            return
        event.acceptProposedAction()
        self._load_folder(folder)

    def _browse_folder(self):
        if self._busy:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select patient folder", _default_browse_dir())
        if folder:
            self._load_folder(Path(folder))

    def toggle_maximize(self, panel):
        """Maximize one panel to fill the window body; click again to restore."""
        home = next(c for c, members in self._column_members.items() if panel in members)
        # Restore everything first.
        for container, members in self._column_members.items():
            container.show()
            for w in members:
                w.show()
        self.left_container.setMaximumWidth(LEFT_COL_MAX)
        self.right_container.setMaximumWidth(RIGHT_COL_MAX)

        if self._maximized_panel is panel:
            panel.set_maximized(False)
            self._maximized_panel = None
        else:
            if self._maximized_panel is not None:
                self._maximized_panel.set_maximized(False)
            for container, members in self._column_members.items():
                container.setVisible(container is home)
            for w in self._column_members[home]:
                w.setVisible(w is panel)
            home.setMaximumWidth(QT_MAX)
            panel.set_maximized(True)
            self._maximized_panel = panel
        for p in self.center_panels:
            if p.isVisible():
                QTimer.singleShot(0, p.refresh_after_layout_change)

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

    def on_run_segmentation(self):
        if self.current_folder is None or self._busy:
            return

        folder_path = self.current_folder
        survival_stats = self._survival_stats

        self.busy_bar.show("Running segmentation")
        self.run_seg_btn.setEnabled(False)
        self._set_2d_loading(True)

        def job():
            modality_paths = MainWindow._find_modality_files(folder_path)
            display_volume, model_input, affine, transform, axcodes = MainWindow._load_and_stack(modality_paths)
            model, device = _ensure_model()
            mask, probs, predicted_days = MainWindow._run_model_static(
                model, device, model_input, display_volume.shape, survival_stats
            )
            tumor_fig, brain_fig = build_3d_figures(display_volume, mask, probs, affine, axcodes)
            return {
                "display_volume": display_volume,
                "affine": affine,
                "transform": transform,
                "axcodes": axcodes,
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
        labels, components, color_map = self._label_tumor_components(mask, probs, voxel_vol_cm3)
        self.sagittal_panel.set_detection(probs, labels, color_map)
        self.axial_panel.set_detection(probs, labels, color_map)
        self.coronal_panel.set_detection(probs, labels, color_map)
        self._set_2d_loading(False)

        self._axcodes = result.get("axcodes")
        self._update_report(mask, labels, components, voxel_vol_cm3, display_volume)
        self.survival_panel.set_days(result["predicted_days"])
        self._display_volume = display_volume
        self._display_affine = affine
        self._volume_transform = result["transform"]
        self._committed_mask = mask.copy()
        self._committed_probs = probs.copy()
        self._preview_mask = None
        self._set_sam_available(True)

        self.busy_bar.hide()
        n_found = len(components)
        self.island.show_done(
            f"Segmentation complete \u00b7 {n_found} tumour{'s' if n_found != 1 else ''} found"
            if n_found else "Segmentation complete \u00b7 no tumour found"
        )
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
        return _component_table(mask, probs, voxel_vol_cm3)

    def _set_2d_loading(self, active: bool):
        """Shows/hides the spinner overlay on all three 2D panels at once."""
        for panel in self.center_panels:
            panel.set_loading(active)

    def _set_sam_available(self, available: bool):
        for panel in self.center_panels:
            panel.enable_sam(available)
        if not available:
            self._exit_sam_mode()

    def _set_sam_busy(self, busy: bool):
        self.island.open_btn.setEnabled(not busy)
        self.run_seg_btn.setEnabled(not busy and self.current_folder is not None)
        for panel in self.center_panels:
            panel.set_sam_busy(busy)

    def _set_sam_preview_available(self, available: bool):
        for panel in self.center_panels:
            panel.set_sam_preview_available(available)

    def _reset_report(self):
        self.patients_record_panel.reset()
        self.summary_panel.reset()
        self.brain_3d_panel.set_legend([])

    def _update_report(self, mask: np.ndarray, labels: np.ndarray, components: list,
                       voxel_vol_cm3: float, volume: Optional[np.ndarray]):
        """Refresh the per-tumor Report card and the Report summary together."""
        total_cm3 = float(mask.sum()) * voxel_vol_cm3
        self.patients_record_panel.set_tumors(components, total_cm3)
        self.brain_3d_panel.set_legend(
            [(f"Tumour {c['rank']}", c["color_hex"]) for c in components[:REPORT_MAX_TUMOR_ROWS]]
        )

        percent = None
        location = None
        bilateral = False
        if volume is not None and volume.shape == mask.shape:
            brain = _brain_tissue_mask(volume, mask)
            brain_voxels = int(brain.sum())
            if brain_voxels:
                percent = 100.0 * float((mask > 0.5).sum()) / brain_voxels
            if components:
                location = _rough_location(labels == components[0]["label_id"], brain, self._axcodes)
                sides = set()
                for comp in components:
                    loc = _rough_location(labels == comp["label_id"], brain, self._axcodes)
                    if loc and loc[0] != "Midline":
                        sides.add(loc[0])
                bilateral = len(sides) > 1
        self.summary_panel.set_summary(
            total_cm3, len(components), percent, location,
            components[0]["color_name"] if components else None, bilateral,
        )

    def _show_binary_mask(self, mask: np.ndarray):
        voxel_vol_cm3 = self._voxel_volume_cm3(self._display_affine)
        labels, components, color_map = self._label_tumor_components(mask, None, voxel_vol_cm3)
        for panel in self.center_panels:
            panel.set_refined_detection(mask, labels, color_map)
        self._update_report(mask, labels, components, voxel_vol_cm3, self._display_volume)

    def _show_committed_mask(self):
        if self._committed_mask is None:
            for panel in self.center_panels:
                panel.clear_detection()
            self._reset_report()
            return
        if self._committed_probs is not None:
            voxel_vol_cm3 = self._voxel_volume_cm3(self._display_affine)
            labels, components, color_map = self._label_tumor_components(
                self._committed_mask, self._committed_probs, voxel_vol_cm3
            )
            for panel in self.center_panels:
                panel.set_detection(self._committed_probs, labels, color_map)
            self._update_report(
                self._committed_mask, labels, components, voxel_vol_cm3, self._display_volume
            )
        else:
            self._show_binary_mask(self._committed_mask)

    def on_run_sam_refinement(self, prompt: dict):
        if self._busy or self._display_volume is None:
            return
        volume = self._display_volume.copy()
        seed = self._committed_mask.copy() if self._committed_mask is not None else None
        axcodes = self._axcodes
        self.busy_bar.show("MedSAM2 refining")
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
            tumor_fig, brain_fig = build_3d_figures(volume, mask, None, self._display_affine, axcodes)
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
        probs = self._committed_probs
        if probs is None or mask is None or probs.shape != mask.shape or mask is not self._committed_mask:
            probs = None
        try:
            tumor_fig, brain_fig = build_3d_figures(display_volume, mask, probs, affine, self._axcodes)
            if tumor_fig is not None:
                self.tumor_3d_panel.set_figure(tumor_fig)
            else:
                self.tumor_3d_panel.clear_figure()
            self.brain_3d_panel.set_figure(brain_fig)
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
        return display_volume, tensor, affine, transform, nib.aff2axcodes(flair_image.affine)

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

    def on_clear_views(self):
        """Unload the case: empty 2D and 3D views, report, survival and MedSAM2."""
        if self._busy:
            return
        self.current_folder = None
        self._nii_files = []
        self._display_volume = None
        self._display_affine = None
        self._volume_transform = None
        self._axcodes = None
        self._committed_mask = None
        self._committed_probs = None
        self._preview_mask = None
        self._pending_3d = None
        for panel in self.center_panels:
            panel.clear_volume()
        for panel3d in (self.tumor_3d_panel, self.brain_3d_panel):
            panel3d.set_empty_message(Panel3D.MSG_NO_CASE)
            panel3d.clear_figure()
        self._reset_report()
        self.survival_panel.set_days(None)
        self._set_sam_preview_available(False)
        self._set_sam_available(False)
        self.patient_info_panel.reset()
        self.island.set_case(None)
        self.run_seg_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

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
        try:
            header_img = nib.load(str(nii_files[0]))
            native_shape = tuple(int(v) for v in header_img.shape[:3])
            native_zooms = tuple(float(z) for z in header_img.header.get_zooms()[:3])
        except Exception:
            native_shape, native_zooms = None, None
        self.patient_info_panel.set_case(case_id, nii_files, native_shape, native_zooms)
        self.island.set_case(case_id)
        self.busy_bar.show("Loading scan")
        for panel3d in (self.tumor_3d_panel, self.brain_3d_panel):
            panel3d.set_empty_message(Panel3D.MSG_READY)
            panel3d.clear_figure()   # drop the previous patient's 3D view
        self._reset_report()
        self._axcodes = None
        self.survival_panel.set_days(None)
        self._committed_mask = None
        self._committed_probs = None
        self._preview_mask = None
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
            self.busy_bar.hide()
            self._set_2d_loading(False)
            self.run_seg_btn.setEnabled(True)
            return False
        return True

    def _on_preview_ready(self, result):
        volume, affine, transform, axcodes = result
        self._axcodes = axcodes
        self.sagittal_panel.set_volume(volume)
        self.axial_panel.set_volume(volume)
        self.coronal_panel.set_volume(volume)
        self._display_volume = volume
        self._display_affine = affine
        self._volume_transform = transform
        self._set_2d_loading(False)
        self.busy_bar.hide()
        self.run_seg_btn.setEnabled(True)
        self._set_sam_available(True)
        self._busy = False

    def _on_preview_failed(self, message: str):
        self.busy_bar.hide()
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
        return volume, affine, transform, nib.aff2axcodes(img.affine)


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