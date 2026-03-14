"""
tracker.py — GUI video tracker with smooth zoom and auto-zoom-out on tracking loss.

Usage:  python tracker.py
  (or pass an input path as the first argument)

Controls during preview:
    SPACE   -> pause / resume
    R       -> reselect subject (works while running or paused)
    Q / ESC -> stop early and save
"""

import sys
from typing import Any
import os
import glob
import queue
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────
OUTPUT_RESOLUTION    = (1280, 720)
PADDING_EXPAND_RATE  = 0.02        # extra padding added per lost frame
MAX_PADDING_MULT     = 3.0         # zoom-out cap = user_padding * this
TRACKER_TYPES        = ["CSRT", "KCF", "MOSSE", "MIL", "BOOSTING", "TLD", "MEDIANFLOW"]
# ─────────────────────────────────────────────────────────────────────────────


# ── Utility functions ──────────────────────────────────────────────────────────

def build_tracker(tracker_type: str):
    t = tracker_type.upper()
    NAMES = {
        "CSRT":       "TrackerCSRT_create",
        "KCF":        "TrackerKCF_create",
        "MOSSE":      "TrackerMOSSE_create",
        "MIL":        "TrackerMIL_create",
        "BOOSTING":   "TrackerBoosting_create",
        "TLD":        "TrackerTLD_create",
        "MEDIANFLOW": "TrackerMedianFlow_create",
    }
    if t not in NAMES:
        raise ValueError(f"Unknown tracker '{tracker_type}'. Choose from: {', '.join(NAMES)}")
    fn_name = NAMES[t]
    for module in (cv2, cv2.legacy):
        fn = getattr(module, fn_name, None)
        if fn is not None:
            return fn()
    raise RuntimeError(f"Tracker '{tracker_type}' not found in cv2 or cv2.legacy.")


def smooth_rect(current: list, target, alpha: float) -> list:
    """EMA toward target. alpha=1 → instant snap, alpha→0 → very slow."""
    return [current[i] * (1 - alpha) + target[i] * alpha for i in range(4)]


def crop_frame(frame: np.ndarray, box: list, pad: float,
               zoom_w: float, zoom_h: float,
               out_w: int, out_h: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    # Use separately-controlled zoom size instead of the (jittery) tracker bbox size
    w, h = zoom_w, zoom_h
    out_ar = out_w / out_h
    pw, ph = w * pad, h * pad
    if pw / ph > out_ar:
        ph = pw / out_ar
    else:
        pw = ph * out_ar
    x1 = int(cx - pw / 2)
    y1 = int(cy - ph / 2)
    x2 = int(cx + pw / 2)
    y2 = int(cy + ph / 2)
    if x1 < 0:     x2 -= x1; x1 = 0
    if y1 < 0:     y2 -= y1; y1 = 0
    if x2 > fw:    x1 -= (x2 - fw); x2 = fw
    if y2 > fh:    y1 -= (y2 - fh); y2 = fh
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Letterbox: scale to fit output without squishing
    ch, cw = crop.shape[:2]
    scale  = min(out_w / cw, out_h / ch)
    nw, nh = int(cw * scale), int(ch * scale)
    resized = cv2.resize(crop, (nw, nh))
    canvas  = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    ox, oy  = (out_w - nw) // 2, (out_h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def select_roi_on_frame(frame: np.ndarray, title: str):
    """
    Show the full frame scaled to fit the screen for ROI selection.
    Returns the ROI in original frame coordinates, or (0,0,0,0) on cancel.
    """
    fh, fw = frame.shape[:2]
    max_w, max_h = 1280, 720
    scale = min(max_w / fw, max_h / fh, 1.0)
    display = cv2.resize(frame, (int(fw * scale), int(fh * scale))) if scale < 1.0 else frame
    roi = cv2.selectROI(title, display, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(title)
    # Map back to original coordinates
    rx, ry, rw, rh = roi
    return (int(rx / scale), int(ry / scale), int(rw / scale), int(rh / scale))


def do_reselect(frame: np.ndarray, tracker_type: str):
    """
    Show the full-frame ROI picker. Returns a new (tracker, smooth_box) on
    success, or (None, None) if the user cancelled.
    """
    x, y, w, h = select_roi_on_frame(frame, "Reselect Subject - drag a box, then SPACE/ENTER")
    if w == 0 or h == 0:
        return None, None
    tracker = build_tracker(tracker_type)
    tracker.init(frame, (x, y, w, h))
    smooth_box = [float(x), float(y), float(w), float(h)]
    return tracker, smooth_box


# ── Worker thread ──────────────────────────────────────────────────────────────

class ProcessingWorker(threading.Thread):
    def __init__(self, input_path, output_path, tracker_type,
                 config, config_lock, q, pause_event, stop_event, reselect_event):
        super().__init__(daemon=True)
        self.input_path      = input_path
        self.output_path     = output_path
        self.tracker_type    = tracker_type
        self.config          = config
        self.config_lock     = config_lock
        self.q               = q
        self.pause_event     = pause_event
        self.stop_event      = stop_event
        self.reselect_event  = reselect_event

    def _put(self, **kwargs):
        self.q.put(kwargs)

    def run(self):
        out_w, out_h = OUTPUT_RESOLUTION
        cap = writer = None
        try:
            # ── Open video ────────────────────────────────────────────────
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                self._put(type="error", message=f"Cannot open video:\n{self.input_path}")
                return

            fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            ret, first_frame = cap.read()
            if not ret:
                self._put(type="error", message="Cannot read first frame.")
                return

            # ── Initial ROI selection ─────────────────────────────────────
            self._put(type="status", text="Select your subject in the preview window, then press SPACE or ENTER.")
            self._put(type="roi_open")
            x, y, w, h = select_roi_on_frame(first_frame, "Select Subject - drag a box, then SPACE/ENTER")
            self._put(type="roi_closed")

            if self.stop_event.is_set():
                self._put(type="done", stopped_early=True, output=self.output_path)
                return

            if w == 0 or h == 0:
                w, h = 100, 100
                x = max(0, x - w // 2)
                y = max(0, y - h // 2)

            tracker = build_tracker(self.tracker_type)
            tracker.init(first_frame, (x, y, w, h))

            # ── Init writer (temp file; audio muxed in after) ─────────────
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(tmp_fd)
            fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
            writer = cv2.VideoWriter(tmp_path, fourcc, fps, (out_w, out_h))

            with self.config_lock:
                padding       = self.config["padding"]
                alpha_pos     = self.config["alpha_pos"]
                alpha_zoom    = self.config["alpha_zoom"]
                zoom_interval = self.config["zoom_interval"]

            smooth_box     = [float(x), float(y), float(w), float(h)]
            smooth_padding = padding
            lost_frames    = 0
            frame_idx      = 1
            current_frame  = first_frame

            # Zoom size is sampled periodically — decoupled from per-frame bbox jitter
            smooth_zoom_w  = float(w)
            smooth_zoom_h  = float(h)
            zoom_target_w  = float(w)
            zoom_target_h  = float(h)
            zoom_counter       = 0
            full_frame_mode    = True
            full_frame_target  = 1.0   # 1 = full frame, 0 = tracked
            transition_alpha   = 1.0   # current blend position

            writer.write(crop_frame(first_frame, smooth_box, smooth_padding,
                                    smooth_zoom_w, smooth_zoom_h, out_w, out_h))
            self._put(type="status", text="Tracking...")

            # ── Frame loop ────────────────────────────────────────────────
            while not self.stop_event.is_set():

                # ── Reselect (works paused or running) ────────────────────
                if self.reselect_event.is_set():
                    self.reselect_event.clear()
                    self._put(type="status", text="Reselecting — drag a box, then SPACE/ENTER.")
                    self._put(type="roi_open")
                    new_tracker, new_box = do_reselect(current_frame, self.tracker_type)
                    self._put(type="roi_closed")
                    if new_tracker is not None and new_box is not None:
                        tracker            = new_tracker
                        full_frame_target  = 0.0
                        zoom_target_w   = new_box[2]
                        zoom_target_h   = new_box[3]
                        lost_frames     = 0
                        zoom_counter    = 0
                        with self.config_lock:
                            smooth_padding = self.config["padding"]
                        self._put(type="reselected")
                    else:
                        status = "Paused" if self.pause_event.is_set() else "Tracking..."
                        self._put(type="status", text=status)
                    continue

                # ── Pause ─────────────────────────────────────────────────
                if self.pause_event.is_set():
                    key = cv2.waitKey(50) & 0xFF
                    if key in (ord('q'), 27):
                        self.stop_event.set()
                    elif key == ord(' '):
                        self.pause_event.clear()
                        self._put(type="resumed")
                    elif key == ord('r'):
                        self.reselect_event.set()
                    elif key == ord('f'):
                        full_frame_target = 1.0
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                current_frame = frame  # keep reference for potential reselect

                # Read live config
                with self.config_lock:
                    padding       = self.config["padding"]
                    alpha_pos     = self.config["alpha_pos"]
                    alpha_zoom    = self.config["alpha_zoom"]
                    zoom_interval = self.config["zoom_interval"]

                if full_frame_mode:
                    # Skip tracking to speed up processing; press R to reselect and resume
                    ok         = False
                    target_pad = padding
                else:
                    ok, bbox = tracker.update(frame)

                    if ok:
                        lost_frames  = 0
                        smooth_box   = smooth_rect(smooth_box, bbox, alpha_pos)
                        target_pad   = padding
                        zoom_counter += 1
                        zoom_sample_frames = max(1, int(zoom_interval * fps))
                        if zoom_counter >= zoom_sample_frames:
                            zoom_counter  = 0
                            zoom_target_w = bbox[2]
                            zoom_target_h = bbox[3]
                    else:
                        lost_frames += 1
                        target_pad   = min(
                            padding + PADDING_EXPAND_RATE * lost_frames,
                            padding * MAX_PADDING_MULT
                        )

                # Smooth zoom size toward its target
                smooth_zoom_w += (zoom_target_w - smooth_zoom_w) * alpha_zoom
                smooth_zoom_h += (zoom_target_h - smooth_zoom_h) * alpha_zoom
                smooth_padding += (target_pad - smooth_padding) * alpha_zoom

                # Smoothly transition transition_alpha toward full_frame_target
                transition_alpha += (full_frame_target - transition_alpha) * 0.06
                transition_alpha  = max(0.0, min(1.0, transition_alpha))
                full_frame_mode   = transition_alpha > 0.999

                # Build full-frame view (letterboxed)
                fh, fw  = frame.shape[:2]
                scale   = min(out_w / fw, out_h / fh)
                nw, nh  = int(fw * scale), int(fh * scale)
                canvas  = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                canvas[(out_h - nh) // 2:(out_h - nh) // 2 + nh,
                       (out_w - nw) // 2:(out_w - nw) // 2 + nw] = cv2.resize(frame, (nw, nh))

                # Build tracked crop view
                tracked_frame = crop_frame(frame, smooth_box, smooth_padding,
                                           smooth_zoom_w, smooth_zoom_h, out_w, out_h)

                # Blend between the two based on transition_alpha
                if transition_alpha <= 0.001:
                    out_frame = tracked_frame
                elif transition_alpha >= 0.999:
                    out_frame = canvas
                else:
                    out_frame = cv2.addWeighted(tracked_frame, 1.0 - transition_alpha,
                                                canvas, transition_alpha, 0)
                writer.write(out_frame)

                # Preview overlay (not written to file)
                preview = out_frame.copy()
                if full_frame_mode:
                    label, color = f"FULL FRAME  {frame_idx}/{total}", (200, 200, 0)
                elif ok:
                    label, color = f"TRACKING  {frame_idx}/{total}", (0, 220, 0)
                else:
                    label, color = f"LOST ({lost_frames})  {frame_idx}/{total}", (0, 0, 220)
                cv2.putText(preview, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                hint = ("R=reselect to resume  SPACE=pause  Q=quit" if full_frame_target >= 1.0
                        else "F=full frame  R=reselect  SPACE=pause  Q=quit")
                cv2.putText(preview, hint, (10, out_h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                cv2.imshow("Tracking Preview", preview)

                frame_idx += 1

                if frame_idx % 5 == 0:
                    self._put(
                        type="progress",
                        frame=frame_idx, total=total,
                        pct=frame_idx / max(total, 1) * 100,
                        tracking=ok
                    )

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    self.stop_event.set()
                elif key == ord(' '):
                    self.pause_event.set()
                    self._put(type="paused")
                elif key == ord('r'):
                    self.reselect_event.set()
                elif key == ord('f'):
                    full_frame_mode = not full_frame_mode

        except Exception as e:
            self._put(type="error", message=str(e))
            return
        finally:
            if cap:    cap.release()
            if writer: writer.release()
            cv2.destroyAllWindows()

        stopped_early = self.stop_event.is_set()

        # ── Mux audio from original into the output file ──────────────────
        self._put(type="status", text="Muxing audio...")
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_path,
                "-i", self.input_path,
                "-map", "0:v:0",
                "-map", "1:a?",        # copy audio if present (? = optional)
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                self.output_path,
            ]
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # ffmpeg not available or failed — fall back to video-only output
            os.replace(tmp_path, self.output_path)
            self._put(type="done", stopped_early=stopped_early, output=self.output_path,
                      warning="ffmpeg not found — output has no audio.")
            return
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        self._put(type="done", stopped_early=stopped_early, output=self.output_path)


# ── GUI ────────────────────────────────────────────────────────────────────────

class TrackerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Video Tracker")
        self.root.resizable(False, False)

        self._worker         = None
        self._pause_event    = threading.Event()
        self._stop_event     = threading.Event()
        self._reselect_event = threading.Event()
        self._config_lock    = threading.Lock()
        self._config: dict[str, Any] = {
            "padding":       1.5,
            "alpha_pos":     0.15,
            "alpha_zoom":    0.08,
            "zoom_interval": 5.0,
        }
        self._q               = queue.Queue()
        self._polling         = False
        self._paused          = False
        self._saved_geometry  = ""

        self._build_ui()
        self._autofill_input()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        P: dict[str, Any] = dict(padx=8, pady=4)

        # Files
        ff = ttk.LabelFrame(self.root, text="Files")
        ff.grid(row=0, column=0, sticky="ew", **P)

        ttk.Label(ff, text="Input video:").grid(row=0, column=0, sticky="w", **P)
        self.input_var = tk.StringVar()
        ttk.Entry(ff, textvariable=self.input_var, width=52).grid(row=0, column=1, **P)
        ttk.Button(ff, text="Browse...", command=self._browse_input).grid(row=0, column=2, **P)

        ttk.Label(ff, text="Output video:").grid(row=1, column=0, sticky="w", **P)
        self.output_var = tk.StringVar(value="tracked_output.mp4")
        ttk.Entry(ff, textvariable=self.output_var, width=52).grid(row=1, column=1, **P)
        ttk.Button(ff, text="Browse...", command=self._browse_output).grid(row=1, column=2, **P)

        # Settings
        sf = ttk.LabelFrame(self.root, text="Settings")
        sf.grid(row=1, column=0, sticky="ew", **P)

        ttk.Label(sf, text="Tracker algorithm:").grid(row=0, column=0, sticky="w", **P)
        self.tracker_var = tk.StringVar(value="CSRT")
        ttk.Combobox(sf, textvariable=self.tracker_var,
                     values=TRACKER_TYPES, width=14,
                     state="readonly").grid(row=0, column=1, sticky="w", **P)

        self._add_slider(sf, row=1,
                         label="Movement smoothing:",
                         hint="Higher = slower/smoother camera movement",
                         var_name="smooth_pos_var", from_=0.01, to=0.99,
                         default=0.85, fmt=".2f", config_key="alpha_pos",
                         invert=True)

        self._add_slider(sf, row=2,
                         label="Zoom smoothing:",
                         hint="Higher = slower zoom transitions",
                         var_name="smooth_zoom_var", from_=0.01, to=0.99,
                         default=0.92, fmt=".2f", config_key="alpha_zoom",
                         invert=True)

        self._add_slider(sf, row=3,
                         label="Padding / zoom level:",
                         hint="Space around subject (1.0 = tight, 4.0 = wide)",
                         var_name="padding_var", from_=1.0, to=4.0,
                         default=1.5, fmt=".1f", config_key="padding")

        self._add_slider(sf, row=4,
                         label="Zoom update interval (s):",
                         hint="Seconds between zoom level recalculations",
                         var_name="zoom_interval_var", from_=1.0, to=15.0,
                         default=5.0, fmt=".1f", config_key="zoom_interval")

        # Controls
        cf = ttk.Frame(self.root)
        cf.grid(row=2, column=0, **P)

        self.start_btn = ttk.Button(cf, text="Start", command=self._start)
        self.start_btn.grid(row=0, column=0, padx=6)

        self.pause_btn = ttk.Button(cf, text="Pause", command=self._toggle_pause,
                                    state="disabled")
        self.pause_btn.grid(row=0, column=1, padx=6)

        self.reselect_btn = ttk.Button(cf, text="Reselect", command=self._reselect,
                                       state="disabled")
        self.reselect_btn.grid(row=0, column=2, padx=6)

        self.stop_btn = ttk.Button(cf, text="Stop", command=self._stop,
                                   state="disabled")
        self.stop_btn.grid(row=0, column=3, padx=6)

        # Progress
        pf = ttk.LabelFrame(self.root, text="Progress")
        pf.grid(row=3, column=0, sticky="ew", **P)

        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(pf, variable=self.progress_var,
                        maximum=100, length=540).grid(row=0, column=0, columnspan=2, **P)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(pf, textvariable=self.status_var, width=40).grid(row=1, column=0, sticky="w", **P)

        self.frame_var = tk.StringVar(value="")
        ttk.Label(pf, textvariable=self.frame_var).grid(row=1, column=1, sticky="e", **P)

    def _add_slider(self, parent, row, label, hint, var_name,
                    from_, to, default, fmt, config_key, invert=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=2)
        var = tk.DoubleVar(value=default)
        setattr(self, var_name, var)

        val_lbl = ttk.Label(parent, text=f"{default:{fmt}}", width=6)

        def on_change(*_):
            v = var.get()
            val_lbl.config(text=f"{v:{fmt}}")
            with self._config_lock:
                self._config[config_key] = (1.0 - v) if invert else v

        var.trace_add("write", on_change)

        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="w", padx=8, pady=2)
        ttk.Scale(frame, variable=var, from_=from_, to=to,
                  orient="horizontal", length=220).pack(side="left")
        ttk.Label(frame, text=hint, foreground="gray").pack(side="left", padx=6)

        val_lbl.grid(row=row, column=2, padx=4, pady=2)

    # ── File browsers ─────────────────────────────────────────────────────────

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            self.input_var.set(path)
            self.output_var.set(os.path.splitext(path)[0] + "_tracked.mp4")

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save output as",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")]
        )
        if path:
            self.output_var.set(path)

    def _autofill_input(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mp4s = glob.glob(os.path.join(script_dir, "*.mp4"))
        if mp4s:
            self.input_var.set(mp4s[0])
            self.output_var.set(os.path.splitext(mp4s[0])[0] + "_tracked.mp4")

    # ── Controls ──────────────────────────────────────────────────────────────

    def _start(self):
        input_path = self.input_var.get().strip()
        if not input_path:
            messagebox.showerror("No input", "Please select an input video.")
            return
        if not os.path.exists(input_path):
            messagebox.showerror("File not found", f"Cannot find:\n{input_path}")
            return

        self._pause_event.clear()
        self._stop_event.clear()
        self._reselect_event.clear()
        self._paused = False
        self.progress_var.set(0)
        self.frame_var.set("")
        self.status_var.set("Starting...")
        self.start_btn.config(state="disabled")
        self.pause_btn.config(state="normal", text="Pause")
        self.reselect_btn.config(state="normal")
        self.stop_btn.config(state="normal")

        self._worker = ProcessingWorker(
            input_path     = input_path,
            output_path    = self.output_var.get().strip() or "tracked_output.mp4",
            tracker_type   = self.tracker_var.get(),
            config         = self._config,
            config_lock    = self._config_lock,
            q              = self._q,
            pause_event    = self._pause_event,
            stop_event     = self._stop_event,
            reselect_event = self._reselect_event,
        )
        self._worker.start()
        self._polling = True
        self._poll_queue()

    def _toggle_pause(self):
        if self._paused:
            self._pause_event.clear()
            self._paused = False
            self.pause_btn.config(text="Pause")
            self.status_var.set("Tracking...")
        else:
            self._pause_event.set()
            self._paused = True
            self.pause_btn.config(text="Resume")
            self.status_var.set("Paused")

    def _reselect(self):
        self._reselect_event.set()

    def _stop(self):
        self._stop_event.set()
        self._pause_event.clear()
        self._reselect_event.clear()

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                msg = self._q.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass
        if self._polling:
            self.root.after(50, self._poll_queue)

    def _handle_msg(self, msg):
        kind = msg.get("type")

        if kind == "progress":
            self.progress_var.set(msg["pct"])
            self.frame_var.set(f"Frame {msg['frame']} / {msg['total']}")
            self.status_var.set("Tracking" if msg["tracking"] else "Lost — zooming out...")

        elif kind == "status":
            self.status_var.set(msg["text"])

        elif kind == "paused":
            self._paused = True
            self.pause_btn.config(text="Resume")
            self.status_var.set("Paused")

        elif kind == "resumed":
            self._paused = False
            self.pause_btn.config(text="Pause")
            self.status_var.set("Tracking...")

        elif kind == "roi_open":
            self._saved_geometry = self.root.wm_geometry()

        elif kind == "roi_closed":
            if self._saved_geometry:
                self.root.geometry(self._saved_geometry)
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()

        elif kind == "reselected":
            self._paused = False
            self.pause_btn.config(text="Pause")
            self.status_var.set("Tracking new subject...")

        elif kind == "done":
            self._polling = False
            self.progress_var.set(100)
            self.start_btn.config(state="normal")
            self.pause_btn.config(state="disabled", text="Pause")
            self.reselect_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
            if msg.get("stopped_early"):
                self.status_var.set("Stopped early.")
            else:
                self.status_var.set("Done!")
                note = f"\n\nNote: {msg['warning']}" if msg.get("warning") else ""
                messagebox.showinfo("Done", f"Output saved to:\n{msg['output']}{note}")

        elif kind == "error":
            self._polling = False
            self.status_var.set("Error")
            self.start_btn.config(state="normal")
            self.pause_btn.config(state="disabled", text="Pause")
            self.reselect_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
            messagebox.showerror("Error", msg["message"])


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    TrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
