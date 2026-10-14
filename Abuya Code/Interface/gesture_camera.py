"""
Touchless 3D control for NeuroVista.

A webcam + MediaPipe hand tracking runs on a QThread and turns hand movement
into small camera deltas for the Plotly 3D views:

    open palm, move       -> rotate (orbit left/right, tilt up/down)
    pinch (OK sign), move up/down -> zoom
    fist                  -> hold (nothing moves)
    point (index only), up/down   -> scroll the active slice
    point, finger held still      -> command "plane" (Axial / Coronal / Sagittal)
    peace sign, hold      -> command "view"   (All / Tumour / Brain)
    thumbs up, hold       -> command "preset" (Front / Side / Top)
    three fingers, hold   -> command "focus"  (next tumour / all)
    open palm held still  -> command "reset"
    no hand               -> nothing moves

Commands need the pose held for HOLD_SECONDS (hold_progress drives the ring
in the camera bubble), then wait COOLDOWN_SECONDS and a pose change before
they can fire again, so a hand resting in one pose never repeats a command.

Uses MediaPipe's current "Tasks" HandLandmarker (works on new mediapipe
versions that no longer ship mp.solutions). It needs a small model file,
hand_landmarker.task (~8 MB). On first use it is downloaded automatically
into Interface/assets/. Download it once before the expo in case the venue
has no internet.

Nothing is recorded or saved: each frame only lives for one loop.
Put this file next to ui_layout.py (in the Interface/ folder).
"""
import math
import sys
import time
import urllib.request
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

ROTATE_GAIN = 5.0      # radians per full-frame swipe left/right
TILT_GAIN = 3.0        # radians per full-frame swipe up/down (make negative to invert)
ZOOM_GAIN = 3.0        # zoom strength while pinching
DEADZONE = 0.004       # ignore hand jitter smaller than this (fraction of frame)
PINCH_ON = 0.30        # thumb-index gap (as a fraction of hand size) to START a pinch
PINCH_OFF = 0.42       # gap needed to LEAVE a pinch (hysteresis stops flicker)
EXTENDED_RATIO = 1.15  # fingertip this much farther from the wrist than its middle joint = finger up
STABLE_FRAMES = 3      # a new gesture must hold this many frames before it takes over
PREVIEW_W = 320        # width of the preview frame sent to the UI
HOLD_SECONDS = 0.6     # how long a command pose must be held before it fires
COOLDOWN_SECONDS = 1.0 # pause after a command before another can fire
RESET_SECONDS = 1.5    # open palm held still this long -> reset
RESET_RING_AFTER = 0.5 # only show the reset ring after the palm has been still this long
STILL_EPS = 0.012      # palm movement (fraction of frame) still counted as "still"
SLICE_GAIN = 160.0     # slices per full-frame finger movement while pointing
PLANE_SECONDS = 1.0    # pointing finger held still this long -> switch slice plane
PLANE_RING_AFTER = 0.35
THUMB_UP_MARGIN = 0.25 # thumb tip must be this far (x hand size) above the knuckles

# Command poses: what each one fires, and the label shown under the ring.
COMMANDS = {
    "peace": ("view", "Peace sign \u00b7 hold for View mode"),
    "thumbs": ("preset", "Thumbs up \u00b7 hold for Next view"),
    "three": ("focus", "Three fingers \u00b7 hold for Next tumour"),
}

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "assets" / "hand_landmarker.task"

# The 21-point hand skeleton, for drawing on the preview.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # little
    (0, 17),                                   # palm edge
)


class GestureCamera(QThread):
    gesture_delta = Signal(float, float, float)   # d_azimuth (rad), d_elevation (rad), zoom factor
    frame_ready = Signal(QImage)                  # mirrored preview with hand skeleton drawn
    state_changed = Signal(str)                   # pose name ("rotate", "slice", ...) or status text
    command = Signal(str)                         # "view" / "preset" / "focus" / "reset" / "plane"
    hold_progress = Signal(float, str)            # 0-1 ring fill + label; 0 hides the ring
    slice_delta = Signal(float)                   # +/- slices while pointing (up = positive)
    failed = Signal(str)

    def __init__(self, cam_index: int = 0, parent=None):
        super().__init__(parent)
        self.cam_index = cam_index
        self._running = True
        self._prev = None
        self._state = None
        self._mode = "none"
        self._cand = "none"
        self._cand_n = 0
        self._mode_since = 0.0      # when the current pose started
        self._armed = True          # False after a command until the pose changes
        self._cool_until = 0.0
        self._still_since = None    # open palm: when it last stopped moving
        self._still_anchor = None
        self._progress = 0.0

    def stop(self):
        self._running = False
        self.wait(3000)

    def _set_state(self, state: str):
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)

    # ------------------------------------------------------------------ setup
    def _ensure_model(self) -> bool:
        if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 1_000_000:
            return True
        self._set_state("Downloading hand model…")
        try:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = MODEL_PATH.with_suffix(".part")
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.replace(MODEL_PATH)
            return True
        except Exception as exc:
            self.failed.emit(
                "Could not download the hand-tracking model "
                f"({exc}).\n\nDownload it manually from:\n{MODEL_URL}\n\n"
                f"and save it as:\n{MODEL_PATH}"
            )
            return False

    def _make_landmarker(self, mp):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        return vision.HandLandmarker.create_from_options(options)

    # ------------------------------------------------------------------- loop
    def run(self):
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            self.failed.emit(
                f"Missing package '{exc.name}'.\nRun: pip install opencv-python mediapipe"
            )
            return

        if not self._ensure_model() or not self._running:
            return
        try:
            landmarker = self._make_landmarker(mp)
        except Exception as exc:
            self.failed.emit(f"Could not start hand tracking:\n{exc}")
            return

        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.cam_index, backend)
        if not cap.isOpened():
            landmarker.close()
            self.failed.emit("No webcam found, or it is being used by another app.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._set_state("none")

        last_ts = 0
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    self.msleep(10)
                    continue
                frame = cv2.flip(frame, 1)   # mirror so movement feels natural
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                ts = int(time.monotonic() * 1000)
                if ts <= last_ts:            # VIDEO mode needs increasing timestamps
                    ts = last_ts + 1
                last_ts = ts
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
                )

                h, w, _ = rgb.shape
                if result.hand_landmarks:
                    lm = result.hand_landmarks[0]
                    self._handle(lm, w / h)
                    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
                    for a, b in HAND_CONNECTIONS:
                        cv2.line(rgb, pts[a], pts[b], (90, 162, 255), 2)
                    for x, y in pts:
                        cv2.circle(rgb, (x, y), 3, (255, 255, 255), -1)
                else:
                    self._prev = None
                    self._set_state("none")
                    self._lost_hand()

                small_h = int(h * PREVIEW_W / w)
                small = cv2.resize(rgb, (PREVIEW_W, small_h))
                img = QImage(small.data, PREVIEW_W, small_h, 3 * PREVIEW_W,
                             QImage.Format_RGB888).copy()
                self.frame_ready.emit(img)
        finally:
            cap.release()
            landmarker.close()

    # --------------------------------------------------------------- gestures
    @staticmethod
    def _classify(lm, aspect: float, current: str) -> str:
        """Decide the pose from the 21 landmarks.

        Distances are relative to the hand's own size, so it works near or far
        from the camera and with the hand slightly tilted.
          hold    fist: all four fingers curled, thumb not pointing up
          thumbs  all four curled, thumb clearly up
          zoom    thumb touching index while middle/ring/little stay up (OK sign)
          rotate  all four fingers up (open palm)
          slice   index up only (pointing)
          peace   index + middle up
          three   index + middle + ring up
        """
        def d(a, b):
            return math.hypot((lm[a].x - lm[b].x) * aspect, lm[a].y - lm[b].y)

        size = d(0, 9) or 1e-6                     # wrist -> middle knuckle
        up = [d(0, tip) > d(0, pip) * EXTENDED_RATIO
              for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))]
        index_up, middle_up, ring_up, little_up = up
        others_up = middle_up + ring_up + little_up
        gap = d(4, 8) / size                       # thumb tip <-> index tip
        pinch_limit = PINCH_OFF if current == "zoom" else PINCH_ON

        if others_up == 0 and not index_up:
            knuckles_top = min(lm[5].y, lm[9].y, lm[13].y, lm[17].y)
            thumb_up = lm[4].y < knuckles_top - THUMB_UP_MARGIN * size
            return "thumbs" if thumb_up else "hold"   # fist wins over pinch
        if gap < pinch_limit and others_up >= 2:
            return "zoom"
        if index_up and others_up == 3:
            return "rotate"
        if index_up and middle_up and ring_up and not little_up:
            return "three"
        if index_up and middle_up and not ring_up and not little_up:
            return "peace"
        if index_up and others_up == 0:
            return "slice"
        return "none"

    def _set_progress(self, value: float, label: str = ""):
        value = max(0.0, min(1.0, value))
        if abs(value - self._progress) >= 0.04 or (value == 0.0) != (self._progress == 0.0):
            self._progress = value
            self.hold_progress.emit(value, label)

    def _fire(self, name: str, now: float):
        self._armed = False
        self._cool_until = now + COOLDOWN_SECONDS
        self._set_progress(0.0)
        self.command.emit(name)

    def _lost_hand(self):
        self._mode = "none"
        self._cand, self._cand_n = "none", 0
        self._armed = True
        self._still_since = None
        self._still_anchor = None
        self._set_progress(0.0)

    def _handle(self, lm, aspect: float = 1.0):
        now = time.monotonic()
        raw = self._classify(lm, aspect, self._mode)
        # Debounce: only switch once the new pose is stable for a few frames.
        if raw == self._cand:
            self._cand_n += 1
        else:
            self._cand, self._cand_n = raw, 1
        if raw != self._mode and self._cand_n >= STABLE_FRAMES:
            self._mode = raw
            self._mode_since = now
            self._prev = None
            self._armed = True               # changing pose re-arms commands
            self._still_since = None
            self._still_anchor = None
            self._set_progress(0.0)
        mode = self._mode
        self._set_state(mode)
        cooling = now < self._cool_until

        # ---- hold-to-confirm command poses
        if mode in COMMANDS:
            self._prev = None
            name, label = COMMANDS[mode]
            if self._armed and not cooling:
                held = (now - self._mode_since) / HOLD_SECONDS
                if held >= 1.0:
                    self._fire(name, now)
                else:
                    self._set_progress(held, label)
            return

        if mode in ("hold", "none"):
            self._prev = None
            return

        cx, cy = lm[9].x, lm[9].y                  # palm centre (middle-finger knuckle)
        if mode == "slice":
            cx, cy = lm[8].x, lm[8].y              # follow the fingertip instead

        # ---- held still: open palm -> reset, pointing finger -> next slice plane
        if mode in ("rotate", "slice"):
            if mode == "rotate":
                total, ring_after, cmd, label = RESET_SECONDS, RESET_RING_AFTER, "reset", "Hold still to reset the view"
            else:
                total, ring_after, cmd, label = PLANE_SECONDS, PLANE_RING_AFTER, "plane", "Hold still to switch plane"
            if self._still_anchor is None or math.hypot(cx - self._still_anchor[0],
                                                        cy - self._still_anchor[1]) > STILL_EPS:
                self._still_anchor = (cx, cy)
                self._still_since = now
                self._set_progress(0.0)
                if not self._armed and not cooling:
                    self._armed = True             # moved again after a command
            elif self._armed and not cooling and self._still_since is not None:
                still = now - self._still_since
                if still >= total:
                    self._fire(cmd, now)
                elif still >= ring_after:
                    self._set_progress((still - ring_after) / (total - ring_after), label)

        if self._prev is None:
            self._prev = (cx, cy)
            return
        dx, dy = cx - self._prev[0], cy - self._prev[1]
        if abs(dx) < DEADZONE and abs(dy) < DEADZONE:
            return
        self._prev = (cx, cy)

        if mode == "zoom":
            self.gesture_delta.emit(0.0, 0.0, 1.0 + dy * ZOOM_GAIN)
        elif mode == "slice":
            self.slice_delta.emit(-dy * SLICE_GAIN)   # finger up = higher slice
        else:
            self.gesture_delta.emit(-dx * ROTATE_GAIN, dy * TILT_GAIN, 1.0)
