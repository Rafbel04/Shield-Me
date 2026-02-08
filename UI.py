import tkinter as tk
from tkinter import ttk, filedialog
import threading
import os
import json

import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path

from pipeline_steps import phase_a_detect, phase_b_generate, phase_c_save_scad


class ShieldMeApp:
    """Main tkinter application for Shield-Me IO Shield Generator."""

    # Colors per port type (RGB for tkinter canvas -- converted from BGR OpenCV colors)
    PORT_COLORS = {
        'usb':           '#0000FF',
        'usb_c':         '#00FFFF',
        'hdmi':          '#00FF00',
        'dp':            '#FFA500',
        'vga':           '#800080',
        'dvi':           '#FFC0CB',
        'ps2':           '#FF14B3',
        'audio':         '#FF00FF',
        'optical_audio': '#FF69B4',
        'ethernet':      '#FFFF00',
        'screw':         '#808000',
        'wifi':          '#FF8000',
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Shield-Me IO Shield Generator")
        self.root.geometry("1100x800")
        self.root.minsize(900, 600)

        # State
        self.image_path = None
        self.official_shield_path = None
        self.official_shield_image = None
        self.phase_a_result = None
        self.phase_b_result = None
        self.cutout_scales = {}  # {port_type: float}

        # Contour selection / movement
        self.selected_contour_idx = None
        self.move_step_mm = 0.25
        self.contour_offsets = {}       # {int index: [dx_mm, dy_mm]}
        self._pending_contour_offsets = None  # loaded from settings, applied after first regen

        # Preview coordinate transform (set during _update_shield_preview)
        self._preview_ox = 0
        self._preview_oy = 0
        self._preview_s = 1.0
        self._preview_plate_h = 48.5

        # Tkinter variables
        self.ppm_multiplier = tk.DoubleVar(value=1.00)
        self.left_anchor_mm = tk.DoubleVar(value=6.0)
        self.bottom_anchor_mm = tk.DoubleVar(value=4.0)
        self.cutout_scale_var = tk.DoubleVar(value=1.0)
        self.selected_port_type = tk.StringVar(value='')

        self._build_ui()
        self._set_state_initial()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Top bar
        top = tk.Frame(self.root, padx=10, pady=8)
        top.pack(fill=tk.X)

        self.select_btn = tk.Button(top, text="Select IO Image", command=self._select_image)
        self.select_btn.pack(side=tk.LEFT)

        self.file_label = tk.Label(top, text="No image selected", fg="gray",
                                   anchor="w", padx=10)
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.select_official_btn = tk.Button(top, text="Load Official Shield",
                                             command=self._select_official_shield)
        self.select_official_btn.pack(side=tk.RIGHT, padx=(0, 5))

        self.detect_btn = tk.Button(top, text="Detect Ports", command=self._start_detection)
        self.detect_btn.pack(side=tk.RIGHT)

        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # Main area -- left (previews) / right (controls)
        main = tk.Frame(self.root, padx=10, pady=5)
        main.pack(fill=tk.BOTH, expand=True)

        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        left = tk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        right = tk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # -- Left: Mask preview --
        tk.Label(left, text="Mask Preview", font=("Arial", 11, "bold")).pack(anchor="w")
        self.mask_label = tk.Label(left, bg="#1a1a1a", relief=tk.SUNKEN)
        self.mask_label.pack(fill=tk.BOTH, expand=True, pady=(2, 5))

        # -- Left: 2D Shield preview --
        tk.Label(left, text="IO Shield Preview", font=("Arial", 11, "bold")).pack(anchor="w")
        self.shield_canvas = tk.Canvas(left, bg="#2a2a2a", height=300, relief=tk.SUNKEN,
                                        takefocus=True)
        self.shield_canvas.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        self.shield_canvas.bind("<Configure>", lambda e: self._update_shield_preview())
        self.shield_canvas.bind("<Button-1>", self._on_shield_click)
        self.shield_canvas.bind("<Left>", self._on_arrow_key)
        self.shield_canvas.bind("<Right>", self._on_arrow_key)
        self.shield_canvas.bind("<Up>", self._on_arrow_key)
        self.shield_canvas.bind("<Down>", self._on_arrow_key)

        # -- Right: Detection summary --
        tk.Label(right, text="Detection Summary", font=("Arial", 11, "bold")).pack(anchor="w")
        self.summary_text = tk.Text(right, height=6, state=tk.DISABLED, wrap=tk.WORD,
                                     font=("Consolas", 10))
        self.summary_text.pack(fill=tk.X, pady=(2, 8))

        # -- Right: Parameter controls --
        controls = tk.LabelFrame(right, text="Parameters", padx=8, pady=8)
        controls.pack(fill=tk.X, pady=(0, 8))

        # Cutout scale
        tk.Label(controls, text="Cutout Scale (per port type):").pack(anchor="w")
        cs_row = tk.Frame(controls)
        cs_row.pack(fill=tk.X, pady=(2, 0))

        self.port_combo = ttk.Combobox(cs_row, textvariable=self.selected_port_type,
                                        state="readonly", width=14)
        self.port_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.port_combo.bind("<<ComboboxSelected>>", self._on_port_type_selected)

        self.cutout_slider = tk.Scale(cs_row, from_=0.5, to=2.0, resolution=0.05,
                                       orient=tk.HORIZONTAL, variable=self.cutout_scale_var,
                                       command=self._on_cutout_scale_changed, showvalue=False)
        self.cutout_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.cutout_entry = tk.Entry(cs_row, textvariable=self.cutout_scale_var, width=6)
        self.cutout_entry.pack(side=tk.LEFT)

        # Left anchor offset
        tk.Label(controls, text="Left Anchor Offset (mm):").pack(anchor="w", pady=(8, 0))
        anch_row = tk.Frame(controls)
        anch_row.pack(fill=tk.X, pady=(2, 0))

        self.anchor_slider = tk.Scale(anch_row, from_=0.0, to=20.0, resolution=0.5,
                                       orient=tk.HORIZONTAL, variable=self.left_anchor_mm,
                                       showvalue=False)
        self.anchor_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.anchor_entry = tk.Entry(anch_row, textvariable=self.left_anchor_mm, width=6)
        self.anchor_entry.pack(side=tk.LEFT)

        # Bottom anchor offset
        tk.Label(controls, text="Bottom Anchor Offset (mm):").pack(anchor="w", pady=(8, 0))
        bot_row = tk.Frame(controls)
        bot_row.pack(fill=tk.X, pady=(2, 0))

        self.bottom_anchor_slider = tk.Scale(bot_row, from_=0.0, to=20.0, resolution=0.5,
                                              orient=tk.HORIZONTAL, variable=self.bottom_anchor_mm,
                                              showvalue=False)
        self.bottom_anchor_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.bottom_anchor_entry = tk.Entry(bot_row, textvariable=self.bottom_anchor_mm, width=6)
        self.bottom_anchor_entry.pack(side=tk.LEFT)

        # Scale multiplier
        tk.Label(controls, text="Scale Multiplier (pixels/mm):").pack(anchor="w", pady=(8, 0))
        ppm_row = tk.Frame(controls)
        ppm_row.pack(fill=tk.X, pady=(2, 0))

        self.ppm_slider = tk.Scale(ppm_row, from_=0.50, to=2.00, resolution=0.001,
                                    orient=tk.HORIZONTAL, variable=self.ppm_multiplier,
                                    showvalue=False)
        self.ppm_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.ppm_entry = tk.Entry(ppm_row, textvariable=self.ppm_multiplier, width=6)
        self.ppm_entry.pack(side=tk.LEFT)

        # Apply button
        self.apply_btn = tk.Button(controls, text="Apply / Regenerate Preview",
                                    command=self._regenerate)
        self.apply_btn.pack(fill=tk.X, pady=(12, 0))

        # Bottom bar
        ttk.Separator(self.root).pack(fill=tk.X, pady=(5, 0))
        bottom = tk.Frame(self.root, padx=10, pady=8)
        bottom.pack(fill=tk.X)

        self.save_btn = tk.Button(bottom, text="Save SCAD File", command=self._save_scad)
        self.save_btn.pack(side=tk.LEFT)

        self.save_settings_btn = tk.Button(bottom, text="Save Settings",
                                            command=self._save_settings)
        self.save_settings_btn.pack(side=tk.LEFT, padx=(5, 0))

        self.status_label = tk.Label(bottom, text="Ready", fg="gray", anchor="w", padx=10)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _set_state_initial(self):
        self.detect_btn.config(state=tk.DISABLED)
        self._set_controls_state(tk.DISABLED)
        self.apply_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.save_settings_btn.config(state=tk.DISABLED)

    def _set_state_image_loaded(self):
        self.detect_btn.config(state=tk.NORMAL)
        self._set_controls_state(tk.DISABLED)
        self.apply_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.save_settings_btn.config(state=tk.NORMAL)  # can save globals even before detection
        self.phase_a_result = None
        self.phase_b_result = None
        self.cutout_scales.clear()

    def _set_state_detected(self):
        self.detect_btn.config(state=tk.NORMAL)
        self._set_controls_state(tk.NORMAL)
        self.apply_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)

    def _set_controls_state(self, state):
        self.port_combo.config(state="readonly" if state == tk.NORMAL else tk.DISABLED)
        self.cutout_slider.config(state=state)
        self.cutout_entry.config(state=state)
        self.anchor_slider.config(state=state)
        self.anchor_entry.config(state=state)
        self.bottom_anchor_slider.config(state=state)
        self.bottom_anchor_entry.config(state=state)
        self.ppm_slider.config(state=state)
        self.ppm_entry.config(state=state)

    def _set_status(self, msg):
        self.status_label.config(text=msg, fg="black")

    # ------------------------------------------------------------------
    # Image selection
    # ------------------------------------------------------------------

    def _select_image(self):
        path = filedialog.askopenfilename(
            title="Select motherboard IO image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.image_path = path
        self.file_label.config(text=os.path.basename(path), fg="black")
        self._set_state_image_loaded()
        self._load_settings()
        self._set_status(f"Image selected: {os.path.basename(path)}")

    def _select_official_shield(self):
        path = filedialog.askopenfilename(
            title="Select official IO shield image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        # Load the image
        img = cv2.imread(path)
        if img is None:
            self._set_status(f"Error: Could not load official shield image")
            return

        self.official_shield_path = path
        self.official_shield_image = img
        self._update_shield_preview()
        self._set_status(f"Official shield loaded: {os.path.basename(path)}")

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _settings_path(self):
        """Return the path to the settings JSON for the current image."""
        if not self.image_path:
            return None
        stem = Path(self.image_path).stem
        return Path(__file__).parent / "saved_configs" / f"{stem}.settings.json"

    def _save_settings(self):
        """Save UI parameters and contour offsets to disk."""
        path = self._settings_path()
        if path is None:
            return

        data = {
            "ppm_multiplier": self.ppm_multiplier.get(),
            "left_anchor_mm": self.left_anchor_mm.get(),
            "bottom_anchor_mm": self.bottom_anchor_mm.get(),
            "cutout_scales": dict(self.cutout_scales),
            "official_shield_path": self.official_shield_path or "",
            "contour_offsets": {str(k): v for k, v in self.contour_offsets.items()},
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        self._set_status(f"Settings saved: {path.name}")

    def _load_settings(self):
        """Load saved settings for the current image, if they exist."""
        path = self._settings_path()
        if path is None or not path.exists():
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        # Restore slider values
        if "ppm_multiplier" in data:
            self.ppm_multiplier.set(data["ppm_multiplier"])
        if "left_anchor_mm" in data:
            self.left_anchor_mm.set(data["left_anchor_mm"])
        if "bottom_anchor_mm" in data:
            self.bottom_anchor_mm.set(data["bottom_anchor_mm"])

        # Restore cutout scales
        if "cutout_scales" in data:
            self.cutout_scales = {k: float(v) for k, v in data["cutout_scales"].items()}

        # Restore official shield image
        if data.get("official_shield_path") and os.path.isfile(data["official_shield_path"]):
            img = cv2.imread(data["official_shield_path"])
            if img is not None:
                self.official_shield_path = data["official_shield_path"]
                self.official_shield_image = img

        # Store contour offsets for later application (after regeneration)
        if data.get("contour_offsets"):
            self._pending_contour_offsets = {
                int(k): v for k, v in data["contour_offsets"].items()
            }

        self._set_status(f"Settings loaded: {path.name}")

    # ------------------------------------------------------------------
    # Detection (threaded)
    # ------------------------------------------------------------------

    def _start_detection(self):
        if not self.image_path:
            return

        self._set_status("Detecting ports... (OpenCV window is open)")
        self.detect_btn.config(state=tk.DISABLED)
        self.select_btn.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._run_detection_thread, daemon=True)
        thread.start()

    def _run_detection_thread(self):
        result = phase_a_detect(
            self.image_path,
            detection_mode="manual",
        )
        self.root.after(0, self._on_detection_complete, result)

    def _on_detection_complete(self, result):
        self.select_btn.config(state=tk.NORMAL)

        if result is None:
            self._set_status("Detection failed or cancelled.")
            self.detect_btn.config(state=tk.NORMAL)
            return

        self.phase_a_result = result
        total = len(result["detections"])
        self._set_status(f"Detected {total} port(s). Adjust parameters and preview.")

        # Populate summary
        self._update_summary()

        # Populate port type dropdown with only detected types
        detected_types = sorted(result["port_summary"].keys())
        self.port_combo["values"] = detected_types
        if detected_types:
            self.selected_port_type.set(detected_types[0])
            self.cutout_scale_var.set(self.cutout_scales.get(detected_types[0], 1.0))

        self._set_state_detected()

        # Auto-generate first preview
        self._regenerate()

        # Re-apply saved contour offsets if we have pending ones
        if self._pending_contour_offsets and self.phase_b_result:
            fitted = self.phase_b_result["contours_fitted"]
            for idx, (dx, dy) in self._pending_contour_offsets.items():
                if 0 <= idx < len(fitted):
                    fitted[idx][:, :, 0] += dx
                    fitted[idx][:, :, 1] += dy
            self.contour_offsets = dict(self._pending_contour_offsets)
            self._pending_contour_offsets = None
            self._update_shield_preview()

    def _update_summary(self):
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        if self.phase_a_result:
            summary = self.phase_a_result["port_summary"]
            total = sum(summary.values())
            self.summary_text.insert(tk.END, f"{total} port(s) detected:\n")
            for pt, count in sorted(summary.items()):
                self.summary_text.insert(tk.END, f"  {pt}: {count}\n")
            ppm = self.phase_a_result["pixels_per_mm"]
            if ppm is not None:
                self.summary_text.insert(tk.END, f"\nCalibrated: {ppm:.3f} px/mm")
            else:
                self.summary_text.insert(tk.END, "\nNo USB calibration (auto-scale)")
        self.summary_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Cutout scale per-port-type
    # ------------------------------------------------------------------

    def _on_port_type_selected(self, event=None):
        pt = self.selected_port_type.get()
        self.cutout_scale_var.set(self.cutout_scales.get(pt, 1.0))

    def _on_cutout_scale_changed(self, *args):
        pt = self.selected_port_type.get()
        if pt:
            self.cutout_scales[pt] = self.cutout_scale_var.get()

    # ------------------------------------------------------------------
    # Regeneration
    # ------------------------------------------------------------------

    def _regenerate(self):
        if self.phase_a_result is None:
            return

        self.selected_contour_idx = None
        self.contour_offsets = {}

        self._set_status("Generating mask and fitting contours...")
        self.root.update_idletasks()

        scales = dict(self.cutout_scales) if self.cutout_scales else None

        try:
            self.phase_b_result = phase_b_generate(
                self.phase_a_result,
                cutout_scales=scales,
                ppm_multiplier=self.ppm_multiplier.get(),
                left_anchor_mm=self.left_anchor_mm.get(),
                bottom_anchor_mm=self.bottom_anchor_mm.get(),
            )
        except Exception as e:
            self._set_status(f"Error: {e}")
            return

        self._update_mask_preview()
        self._update_shield_preview()
        self._set_status("Preview updated. Adjust parameters or save SCAD.")

    # ------------------------------------------------------------------
    # Mask preview
    # ------------------------------------------------------------------

    def _update_mask_preview(self):
        if self.phase_b_result is None:
            return

        mask = self.phase_b_result["mask"]

        # Resize to fit the label
        label_w = self.mask_label.winfo_width() or 500
        label_h = self.mask_label.winfo_height() or 250
        h, w = mask.shape[:2]
        scale = min(label_w / w, label_h / h, 1.0)
        disp_w = max(int(w * scale), 1)
        disp_h = max(int(h * scale), 1)
        resized = cv2.resize(mask, (disp_w, disp_h))

        pil_img = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_img)
        self.mask_label.config(image=photo)
        self.mask_label.image = photo  # prevent GC

    # ------------------------------------------------------------------
    # Contour selection and movement
    # ------------------------------------------------------------------

    def _canvas_to_mm(self, cx, cy):
        """Convert canvas pixel coordinates back to mm coordinates."""
        mx = (cx - self._preview_ox) / self._preview_s
        my = self._preview_plate_h - (cy - self._preview_oy) / self._preview_s
        return mx, my

    def _on_shield_click(self, event):
        """Handle click on shield canvas to select/deselect a contour."""
        self.shield_canvas.focus_set()

        if not (self.phase_b_result and self.phase_b_result["contours_fitted"]):
            return

        mx, my = self._canvas_to_mm(event.x, event.y)
        click_pt = (mx, my)

        # Check each contour for hit
        for i, cnt in enumerate(self.phase_b_result["contours_fitted"]):
            # Reshape to (N, 2) for pointPolygonTest
            pts = cnt.reshape(-1, 2).astype(np.float32)
            if cv2.pointPolygonTest(pts, click_pt, False) >= 0:
                self.selected_contour_idx = i
                self._update_shield_preview()
                self._set_status(f"Selected contour {i + 1} -- use arrow keys to move")
                return

        # Clicked empty space -- deselect
        self.selected_contour_idx = None
        self._update_shield_preview()
        self._set_status("Preview updated. Adjust parameters or save SCAD.")

    def _on_arrow_key(self, event):
        """Move the selected contour with arrow keys."""
        if self.selected_contour_idx is None:
            return
        if not (self.phase_b_result and self.phase_b_result["contours_fitted"]):
            return

        idx = self.selected_contour_idx
        cnt = self.phase_b_result["contours_fitted"][idx]
        step = self.move_step_mm

        dx, dy = 0.0, 0.0
        if event.keysym == "Left":
            dx = -step
        elif event.keysym == "Right":
            dx = step
        elif event.keysym == "Up":
            dy = step   # Y increases upward in SCAD coords
        elif event.keysym == "Down":
            dy = -step

        cnt[:, :, 0] += dx
        cnt[:, :, 1] += dy

        # Accumulate offset for persistence
        prev = self.contour_offsets.get(idx, [0.0, 0.0])
        self.contour_offsets[idx] = [prev[0] + dx, prev[1] + dy]

        self._update_shield_preview()

    # ------------------------------------------------------------------
    # 2D Shield preview (canvas)
    # ------------------------------------------------------------------

    def _update_shield_preview(self):
        canvas = self.shield_canvas
        canvas.delete("all")

        # Full IO shield plate dimensions (mm)
        plate_w_mm = 164.0
        plate_h_mm = 48.5

        # Usable area is 3.8mm inset from physical edges
        inset_mm = 3.8
        usable_x_min = inset_mm
        usable_x_max = plate_w_mm - inset_mm
        usable_y_min = inset_mm
        usable_y_max = plate_h_mm - inset_mm

        cw = canvas.winfo_width() or 600
        ch = canvas.winfo_height() or 300
        margin = 15

        sx = (cw - 2 * margin) / plate_w_mm
        sy = (ch - 2 * margin) / plate_h_mm
        s = min(sx, sy)

        # Center the drawing
        total_w = plate_w_mm * s
        total_h = plate_h_mm * s
        ox = (cw - total_w) / 2
        oy = (ch - total_h) / 2

        # Store transform for click-to-mm conversion
        self._preview_ox = ox
        self._preview_oy = oy
        self._preview_s = s
        self._preview_plate_h = plate_h_mm

        def mm_to_canvas(mx, my):
            cx = ox + mx * s
            cy = oy + (plate_h_mm - my) * s  # flip Y
            return cx, cy

        # BACKGROUND: Draw official shield image if loaded
        if self.official_shield_image is not None:
            target_w = int(total_w)
            target_h = int(total_h)
            resized = cv2.resize(self.official_shield_image, (target_w, target_h))

            # Convert BGR to RGB for tkinter
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(pil_img)

            # Display the image centered on the canvas
            canvas.create_image(ox, oy, image=photo, anchor=tk.NW)
            canvas.image = photo  # prevent GC

        # Outer edge: Physical shield plate outline (SOLID)
        x0, y0 = mm_to_canvas(0, 0)
        x1, y1 = mm_to_canvas(plate_w_mm, plate_h_mm)
        canvas.create_rectangle(x0, y1, x1, y0, outline="#666", width=2)

        # Inner edge: Usable area 3.8mm inset from outer edge (DASHED)
        ux0, uy0 = mm_to_canvas(usable_x_min, usable_y_min)
        ux1, uy1 = mm_to_canvas(usable_x_max, usable_y_max)
        canvas.create_rectangle(ux0, uy1, ux1, uy0, outline="#4488cc", width=1, dash=(4, 4))

        # Draw fitted contours
        if self.phase_b_result and self.phase_b_result["contours_fitted"]:
            for i, cnt in enumerate(self.phase_b_result["contours_fitted"]):
                points = []
                for pt in cnt:
                    mx, my = float(pt[0][0]), float(pt[0][1])
                    px, py = mm_to_canvas(mx, my)
                    points.extend([px, py])
                if len(points) >= 6:
                    if i == self.selected_contour_idx:
                        canvas.create_polygon(points, outline="#00ff00", fill="",
                                               width=3)
                    else:
                        canvas.create_polygon(points, outline="#ff4444", fill="",
                                               width=2)

    # ------------------------------------------------------------------
    # Save SCAD
    # ------------------------------------------------------------------

    def _save_scad(self):
        if self.phase_b_result is None:
            return

        stem = Path(self.image_path).stem
        default_name = f"{stem}_shield.scad"

        filepath = filedialog.asksaveasfilename(
            defaultextension=".scad",
            filetypes=[("OpenSCAD files", "*.scad"), ("All files", "*.*")],
            initialfile=default_name,
        )
        if not filepath:
            return

        ok = phase_c_save_scad(self.phase_b_result, filepath)
        if ok:
            self._save_settings()
            self._set_status(f"Saved: {filepath}")
        else:
            self._set_status("Error saving SCAD file.")


def main():
    root = tk.Tk()
    ShieldMeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
