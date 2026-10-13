"""
Touchless 3D control for NeuroVista.

A webcam + MediaPipe Hands runs on a QThread and turns hand movement into
small camera deltas for the Plotly 3D views:

    open palm, move       -> rotate (orbit left/right, tilt up/down)
    pinch, move up/down   -> zoom
    fist                  -> hold (nothing moves)
    no hand               -> nothing moves

Nothing is recorded or saved: each frame only lives for one loop.
Put this file next to ui_layout.py (in the Interface/ folder).
"""
import math
import sys

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

ROTATE_GAIN = 5.0      # radians per full-frame swipe left/right
TILT_GAIN = 3.0        # radians per full-frame swipe up/down (make negative to invert)
ZOOM_GAIN = 3.0        # zoom strength while pinching
DEADZONE = 0.004       # ignore hand jitter smaller than this (fraction of frame)
PINCH_DIST = 0.05      # thumb-to-index distance that counts as a pinch
PREVIEW_W = 320        # width of the preview frame sent to the UI


class GestureCamera(QThread):
    gesture_delta = Signal(float, float, float)   # d_azimuth (rad), d_elevation (rad), zoom factor
    frame_ready = Signal(QImage)                  # mirrored preview with hand skeleton drawn
    state_changed = Signal(str)                   # "rotate" / "zoom" / "hold" / "none"
    failed = Signal(str)

    def __init__(self, cam_index: int = 0, parent=None):
        super().__init__(parent)
        self.cam_index = cam_index
        self._running = True
        self._prev = None
        self._state = None

    def stop(self):
        self._running = False
        self.wait(2000)

    def _set_state(self, state: str):
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)

    def run(self):
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            self.failed.emit(
                f"Missing package '{exc.name}'.\nRun: pip install opencv-python mediapipe"
            )
            return
        if not hasattr(mp, "solutions"):
            self.failed.emit(
                "This mediapipe version has no 'mp.solutions' API.\n"
                "Install an older mediapipe release, or switch this file to the Tasks API."
            )
            return

        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.cam_index, backend)
        if not cap.isOpened():
            self.failed.emit("No webcam found, or it is being used by another app.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        hands_mod = mp.solutions.hands
        hands = hands_mod.Hands(
            max_num_hands=1,
            model_complexity=0,            # fastest model; fine for one hand
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        drawer = mp.solutions.drawing_utils
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    self.msleep(10)
                    continue
                frame = cv2.flip(frame, 1)   # mirror so movement feels natural
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = hands.process(rgb)
                rgb.flags.writeable = True

                if res.multi_hand_landmarks:
                    hand = res.multi_hand_landmarks[0]
                    self._handle(hand.landmark)
                    drawer.draw_landmarks(rgb, hand, hands_mod.HAND_CONNECTIONS)
                else:
                    self._prev = None
                    self._set_state("none")

                h, w, _ = rgb.shape
                small_h = int(h * PREVIEW_W / w)
                small = cv2.resize(rgb, (PREVIEW_W, small_h))
                img = QImage(small.data, PREVIEW_W, small_h, 3 * PREVIEW_W,
                             QImage.Format_RGB888).copy()
                self.frame_ready.emit(img)
        finally:
            cap.release()
            hands.close()

    def _handle(self, lm):
        cx, cy = lm[9].x, lm[9].y   # palm centre (middle-finger knuckle)
        pinch = math.dist((lm[4].x, lm[4].y), (lm[8].x, lm[8].y)) < PINCH_DIST
        extended = sum(
            lm[tip].y < lm[pip].y for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
        )
        if pinch:
            mode = "zoom"
        elif extended >= 4:
            mode = "rotate"
        elif extended == 0:
            mode = "hold"
        else:
            mode = "none"            # in-between pose: ignore

        self._set_state(mode)
        if mode in ("hold", "none"):
            self._prev = None
            return
        if self._prev is None or self._prev[2] != mode:
            self._prev = (cx, cy, mode)
            return

        dx, dy = cx - self._prev[0], cy - self._prev[1]
        if abs(dx) < DEADZONE and abs(dy) < DEADZONE:
            return
        self._prev = (cx, cy, mode)

        if mode == "zoom":
            self.gesture_delta.emit(0.0, 0.0, 1.0 + dy * ZOOM_GAIN)
        else:
            self.gesture_delta.emit(-dx * ROTATE_GAIN, dy * TILT_GAIN, 1.0)
