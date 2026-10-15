"""
One-time installer: adds the opt-in "Gesture" button (webcam hand control
of the 3D views) to ui_layout.py.

HOW TO USE
  1. Put this file AND gesture_camera.py in the same folder as ui_layout.py
     (your Interface/ folder).
  2. Open a terminal in that folder and run:
         python add_gesture_control.py
  3. Start the app as usual:
         python ui_layout.py

Safety:
  - A backup is saved first as ui_layout.backup.py.
  - If any part of ui_layout.py doesn't look as expected, NOTHING is changed.
  - Running it twice does nothing the second time.
"""
import ast
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "ui_layout.py"
BACKUP = TARGET.with_name(TARGET.stem + ".backup.py")
MARKER = "_inject_gesture_hook"


# --------------------------------------------------------------------------
# 1. JavaScript camera hook (goes just above `class Panel3D`)
# --------------------------------------------------------------------------
HOOK_CODE = r'''# ---------------------------------------------------------------------------
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
        var k = unit(cam.up || {x: 0, y: 0, z: 1});
        var v = {x: cam.eye.x - c.x, y: cam.eye.y - c.y, z: cam.eye.z - c.z};
        v = rot(v, k, dAz);
        var r = unit({x: k.y*v.z - k.z*v.y, y: k.z*v.x - k.x*v.z, z: k.x*v.y - k.y*v.x});
        var t = rot(v, r, dEl);
        var tl = Math.sqrt(t.x*t.x + t.y*t.y + t.z*t.z) || 1;
        if (Math.abs((k.x*t.x + k.y*t.y + k.z*t.z) / tl) < 0.97) v = t;
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
        pass'''

# --------------------------------------------------------------------------
# 2. New Panel3D methods (webcam card + sending gestures into the page)
# --------------------------------------------------------------------------
PANEL_METHODS = r'''    # ---- webcam gesture preview (hidden until Gesture is switched on) ----
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

'''

# --------------------------------------------------------------------------
# 3. Island: hand icon + Gesture toggle button
# --------------------------------------------------------------------------
ICON_BRANCH = r'''    elif kind == "gesture":
        p.drawRoundedRect(QRectF(13, 24, 22, 16), 7, 7)          # palm
        for x in (16, 21, 26, 31):
            p.drawLine(QPointF(x, 24), QPointF(x, 11))            # fingers
        p.drawLine(QPointF(13, 31), QPointF(8, 24))               # thumb
'''

ISLAND_BUTTON = r'''        # Gesture toggle: OFF at startup. The webcam only starts when clicked.
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
'''

# --------------------------------------------------------------------------
# 4. MainWindow wiring
# --------------------------------------------------------------------------
MAIN_INIT = r'''        # Webcam gesture control (off until the island's Gesture button is clicked).
        self.island.gesture_toggled.connect(self._toggle_gesture)
        self._gesture = None
        self._gesture_acc = [0.0, 0.0, 1.0]
        self._gesture_timer = QTimer(self)
        self._gesture_timer.setInterval(33)          # ~30 camera updates per second
        self._gesture_timer.timeout.connect(self._flush_gesture)
'''

MAIN_METHODS = r'''    # ---------------- Webcam gesture control ----------------------------------
    _GESTURE_LABELS = {
        "rotate": "\u270b Rotating", "zoom": "\U0001F90F Zooming",
        "hold": "\u270a Holding", "none": "Show an open palm",
    }

    def _toggle_gesture(self, on: bool):
        if not on:
            self._stop_gesture()
            return
        here = str(Path(__file__).resolve().parent)
        if here not in sys.path:
            sys.path.insert(0, here)
        try:
            from gesture_camera import GestureCamera
        except ImportError:
            self.island.gesture_btn.setChecked(False)
            QMessageBox.warning(self, "Gesture control",
                                "gesture_camera.py was not found next to ui_layout.py.")
            return
        self._gesture = GestureCamera()
        self._gesture.gesture_delta.connect(self._on_gesture_delta)
        self._gesture.frame_ready.connect(self.brain_3d_panel.set_camera_frame)
        self._gesture.state_changed.connect(
            lambda s: self.brain_3d_panel.set_camera_state(self._GESTURE_LABELS.get(s, s)))
        self._gesture.failed.connect(self._on_gesture_failed)
        self._gesture.start()
        self._gesture_timer.start()
        self.brain_3d_panel.show_camera_preview(True)

    def _stop_gesture(self):
        self._gesture_timer.stop()
        if self._gesture is not None:
            self._gesture.stop()
            self._gesture = None
        self._gesture_acc = [0.0, 0.0, 1.0]
        self.brain_3d_panel.show_camera_preview(False)

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

'''

ROTATE_OLD = "if (paused || busy || document.hidden) { last = now; return; }"
ROTATE_NEW = "if (paused || busy || document.hidden || window.nvGestureActive) { last = now; return; }"

ISLAND_LOOP = (
    "        for btn in (self.open_btn, self.run_btn, self.refine_btn, self.clear_btn):\n"
    "            row.addWidget(btn)\n"
)
ICON_END = "    p.end()\n    return QIcon(pix)\n\n\nclass DynamicIsland(QFrame):"

# (what it is, text to find, text to put in its place)
EDITS = [
    ("auto-rotate pause line", ROTATE_OLD, ROTATE_NEW),
    ("gesture JavaScript hook", "\n\nclass Panel3D(HivePanel):",
     "\n\n" + HOOK_CODE + "\n\n\nclass Panel3D(HivePanel):"),
    ("Panel3D webcam card setup", "        view_stack.addWidget(self.web_view)\n",
     "        view_stack.addWidget(self.web_view)\n"
     "        self._view_stack = view_stack\n"
     "        self._build_camera_preview()\n"),
    ("Panel3D hook injection", "        _inject_middle_click_pan(tmp.name)\n",
     "        _inject_middle_click_pan(tmp.name)\n"
     "        _inject_gesture_hook(tmp.name)\n"),
    ("Panel3D new methods", "    def set_empty_message(self, message: str):",
     PANEL_METHODS + "    def set_empty_message(self, message: str):"),
    ("island hand icon", ICON_END, ICON_BRANCH + ICON_END),
    ("island gesture signal", "    sam_action = Signal(str)\n",
     "    sam_action = Signal(str)\n    gesture_toggled = Signal(bool)\n"),
    ("island Gesture button", ISLAND_LOOP, ISLAND_BUTTON),
    ("MainWindow wiring", "        self.island.sam_action.connect(self._on_island_sam_action)\n",
     "        self.island.sam_action.connect(self._on_island_sam_action)\n" + MAIN_INIT),
    ("MainWindow gesture methods", "    def _sync_island_buttons(self):",
     MAIN_METHODS + "    def _sync_island_buttons(self):"),
]


def main() -> int:
    if not TARGET.is_file():
        print(f"Could not find {TARGET.name} in {TARGET.parent}")
        print("Put this script in the same folder as ui_layout.py and run it again.")
        return 1
    if not (TARGET.parent / "gesture_camera.py").is_file():
        print("Note: gesture_camera.py is not in this folder yet - copy it here too.")

    raw = TARGET.read_text(encoding="utf-8")
    if MARKER in raw:
        print("Gesture control is already added. Nothing to do.")
        return 0

    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")

    problems = []
    for name, old, _new in EDITS:
        count = text.count(old)
        if count != 1:
            problems.append(f"  - {name}: expected 1 match, found {count}")
    if problems:
        print("ui_layout.py doesn't look like the version this was written for.")
        print("Nothing was changed. Details:")
        print("\n".join(problems))
        print("Send these lines to Claude and the script can be adjusted.")
        return 1

    for _name, old, new in EDITS:
        text = text.replace(old, new, 1)

    try:
        ast.parse(text)
    except SyntaxError as exc:
        print(f"Result would not be valid Python (line {exc.lineno}). Nothing was changed.")
        return 1

    shutil.copy2(TARGET, BACKUP)
    if crlf:
        text = text.replace("\n", "\r\n")
    TARGET.write_bytes(text.encode("utf-8"))

    print("Done! Gesture control added to ui_layout.py")
    print(f"Backup of the original: {BACKUP.name}")
    print("Next: pip install opencv-python mediapipe   then   python ui_layout.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
