"""
Interactive port selector for Shield-Me.

The user draws rough bounding boxes around ports on the motherboard image
and selects the port type via keyboard shortcut. Template matching runs
within each drawn region to refine the exact position and size.
"""

import cv2
import numpy as np


class InteractivePortSelector:
    """
    OpenCV-based interactive window for marking ports on a motherboard
    IO panel image. Each drawn rectangle is refined via template matching
    within the drawn region.

    Produces detections in the same tuple format as
    PortMaskApplicator.detect_ports():
        (x, y, w, h, scale, port_type, confidence)
    """

    WINDOW_NAME = "Shield-Me: Mark Ports"
    MAX_DISPLAY_WIDTH = 1400
    MAX_DISPLAY_HEIGHT = 900

    KEY_MAP = {
        ord('u'): 'usb',
        ord('c'): 'usb_c',
        ord('h'): 'hdmi',
        ord('d'): 'dp',
        ord('v'): 'vga',
        ord('i'): 'dvi',
        ord('p'): 'ps2',
        ord('a'): 'audio',
        ord('o'): 'optical_audio',
        ord('e'): 'ethernet',
        ord('s'): 'screw',
    }

    COLORS = {
        'usb':           (255, 0, 0),
        'vga':           (128, 0, 128),
        'hdmi':          (0, 255, 0),
        'dp':            (0, 165, 255),
        'usb_c':         (255, 255, 0),
        'dvi':           (203, 192, 255),
        'ps2':           (147, 20, 255),
        'audio':         (255, 0, 255),
        'optical_audio': (180, 105, 255),
        'ethernet':      (0, 255, 255),
        'screw':         (0, 128, 128),
    }

    def __init__(self, image, applicator):
        """
        Args:
            image: Original BGR image (full resolution).
            applicator: PortMaskApplicator instance (for region-scoped detection).
        """
        self._original_image = image
        self._applicator = applicator
        self._detections = []       # final detection tuples (original coords)
        self._is_refined = []       # True if CV-refined, False if fallback
        self._current_port = 'usb'
        self._drawing = False
        self._drag_start = None     # (x, y) in display coords
        self._drag_current = None
        self._show_help = True
        self._cancelled = False
        self._threshold = 0.5

        # Compute display scaling
        orig_h, orig_w = image.shape[:2]
        scale_w = self.MAX_DISPLAY_WIDTH / orig_w
        scale_h = self.MAX_DISPLAY_HEIGHT / orig_h
        self._scale_factor = min(scale_w, scale_h, 1.0)

        disp_w = int(orig_w * self._scale_factor)
        disp_h = int(orig_h * self._scale_factor)
        if self._scale_factor < 1.0:
            self._display_image = cv2.resize(image, (disp_w, disp_h))
        else:
            self._display_image = image.copy()

    def run(self):
        """
        Display the interactive window. Blocks until the user finalizes
        or cancels.

        Returns:
            List of detection tuples in original image coordinates, or
            empty list if cancelled.
        """
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)

        print("\n" + "=" * 70)
        print("INTERACTIVE PORT SELECTOR")
        print("=" * 70)
        print("Select port type with keyboard, then click+drag to mark it.")
        print("[]/[] to adjust CV threshold, [?] for help, [Enter/q] done, [Esc] cancel.\n")

        while True:
            frame = self._render_frame()
            cv2.imshow(self.WINDOW_NAME, frame)

            key = cv2.waitKey(30) & 0xFF

            if key == 13 or key == ord('q'):  # Enter or q
                break
            elif key == 27:  # Escape
                self._cancelled = True
                break
            elif key == ord('z'):
                if self._detections:
                    removed = self._detections.pop()
                    self._is_refined.pop()
                    print(f"  Undo: removed {removed[5]}")
            elif key == ord('?'):
                self._show_help = not self._show_help
            elif key == ord(']'):
                self._threshold = min(0.95, self._threshold + 0.05)
                print(f"  Threshold: {self._threshold:.2f}")
            elif key == ord('['):
                self._threshold = max(0.05, self._threshold - 0.05)
                print(f"  Threshold: {self._threshold:.2f}")
            elif key in self.KEY_MAP:
                self._current_port = self.KEY_MAP[key]
                print(f"  Selected: {self._current_port}")

        cv2.destroyWindow(self.WINDOW_NAME)

        if self._cancelled:
            print("Selection cancelled.")
            return []

        print(f"\nFinalized {len(self._detections)} port(s).")
        return list(self._detections)

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._drag_start = (x, y)
            self._drag_current = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._drawing:
                self._drag_current = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if not self._drawing:
                return
            self._drawing = False
            self._drag_current = (x, y)
            self._commit_rectangle()

    def _commit_rectangle(self):
        """Convert drawn rectangle to original coords and run CV refinement."""
        if self._drag_start is None or self._drag_current is None:
            return

        dx1, dy1 = self._drag_start
        dx2, dy2 = self._drag_current

        # Reject tiny rectangles (accidental clicks)
        if abs(dx2 - dx1) < 3 or abs(dy2 - dy1) < 3:
            return

        # Convert to original image coordinates
        ox1, oy1 = self._display_to_original(min(dx1, dx2), min(dy1, dy2))
        ox2, oy2 = self._display_to_original(max(dx1, dx2), max(dy1, dy2))
        region = (ox1, oy1, ox2 - ox1, oy2 - oy1)

        # Try CV refinement within the drawn region
        refined = self._applicator.detect_port_in_region(
            self._original_image, region, self._current_port,
            threshold=self._threshold
        )

        if refined is not None:
            self._detections.append(refined)
            self._is_refined.append(True)
            print(f"  + {self._current_port} (refined, conf={refined[6]:.2f})")
        else:
            # Fallback: use the user-drawn box as-is
            fallback = (ox1, oy1, ox2 - ox1, oy2 - oy1,
                        1.0, self._current_port, 1.0)
            self._detections.append(fallback)
            self._is_refined.append(False)
            print(f"  + {self._current_port} (no CV match, using drawn box)")

        self._drag_start = None
        self._drag_current = None

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _display_to_original(self, dx, dy):
        ox = int(dx / self._scale_factor)
        oy = int(dy / self._scale_factor)
        # Clamp to image bounds
        img_h, img_w = self._original_image.shape[:2]
        return max(0, min(ox, img_w)), max(0, min(oy, img_h))

    def _original_to_display(self, ox, oy):
        return int(ox * self._scale_factor), int(oy * self._scale_factor)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_frame(self):
        frame = self._display_image.copy()

        # Draw committed detections
        for idx, det in enumerate(self._detections):
            x, y, w, h, _, port_type, conf = det
            dx1, dy1 = self._original_to_display(x, y)
            dx2, dy2 = self._original_to_display(x + w, y + h)
            color = self.COLORS.get(port_type, (0, 0, 255))
            refined = self._is_refined[idx]

            if refined:
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 2)
            else:
                # Dashed rectangle for fallback detections
                self._draw_dashed_rect(frame, (dx1, dy1), (dx2, dy2), color)

            label = f"{port_type} ({conf:.2f})"
            cv2.putText(frame, label, (dx1, dy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Rubber-band rectangle while dragging
        if self._drawing and self._drag_start and self._drag_current:
            color = self.COLORS.get(self._current_port, (0, 0, 255))
            pt1 = (min(self._drag_start[0], self._drag_current[0]),
                    min(self._drag_start[1], self._drag_current[1]))
            pt2 = (max(self._drag_start[0], self._drag_current[0]),
                    max(self._drag_start[1], self._drag_current[1]))
            cv2.rectangle(frame, pt1, pt2, color, 1)

        # Status bar
        self._draw_status_bar(frame)

        # Help overlay
        if self._show_help:
            self._draw_help_overlay(frame)

        return frame

    def _draw_status_bar(self, frame):
        bar_h = 30
        cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_h), (0, 0, 0), -1)
        color = self.COLORS.get(self._current_port, (255, 255, 255))
        text = (f"Port: {self._current_port}  |  "
                f"Thresh: {self._threshold:.2f}  |  "
                f"Placed: {len(self._detections)}  |  "
                f"[?] Help  [z] Undo  [Enter/q] Done  [Esc] Cancel")
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_help_overlay(self, frame):
        sorted_keys = sorted(self.KEY_MAP.items(), key=lambda kv: kv[1])
        line_h = 22
        overlay_w = 240
        overlay_h = (len(sorted_keys) + 2) * line_h
        x0 = frame.shape[1] - overlay_w - 10
        y0 = frame.shape[0] - overlay_h - 10

        # Clamp to frame
        x0 = max(0, x0)
        y0 = max(0, y0)

        # Semi-transparent background
        x1 = min(x0 + overlay_w, frame.shape[1])
        y1 = min(y0 + overlay_h, frame.shape[0])
        roi = frame[y0:y1, x0:x1]
        dark = np.zeros_like(roi)
        frame[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.3, dark, 0.7, 0)

        # Header
        cv2.putText(frame, "KEYBOARD SHORTCUTS", (x0 + 10, y0 + line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Port type lines
        for i, (key_ord, port_type) in enumerate(sorted_keys):
            y_text = y0 + (i + 2) * line_h
            color = self.COLORS.get(port_type, (255, 255, 255))
            # Color swatch
            cv2.rectangle(frame, (x0 + 10, y_text - 10),
                          (x0 + 22, y_text), color, -1)
            label = f"[{chr(key_ord)}] {port_type}"
            if port_type == self._current_port:
                label += "  <--"
            cv2.putText(frame, label, (x0 + 28, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    @staticmethod
    def _draw_dashed_rect(frame, pt1, pt2, color, dash_len=8, gap_len=5):
        """Draw a dashed rectangle (used for fallback/unrefined detections)."""
        x1, y1 = pt1
        x2, y2 = pt2
        # Draw dashed lines for each edge
        for start, end in [((x1, y1), (x2, y1)),   # top
                           ((x2, y1), (x2, y2)),    # right
                           ((x2, y2), (x1, y2)),    # bottom
                           ((x1, y2), (x1, y1))]:   # left
            _draw_dashed_line(frame, start, end, color, dash_len, gap_len)


def _draw_dashed_line(frame, pt1, pt2, color, dash_len=8, gap_len=5):
    """Draw a dashed line between two points."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    length = int(np.hypot(dx, dy))
    if length == 0:
        return

    for i in range(0, length, dash_len + gap_len):
        t_start = i / length
        t_end = min((i + dash_len) / length, 1.0)
        sx = int(x1 + dx * t_start)
        sy = int(y1 + dy * t_start)
        ex = int(x1 + dx * t_end)
        ey = int(y1 + dy * t_end)
        cv2.line(frame, (sx, sy), (ex, ey), color, 2)


def interactive_detect_ports(image, applicator):
    """
    Drop-in replacement for PortMaskApplicator.detect_ports() that uses
    interactive user selection with CV refinement.

    Args:
        image: BGR image (full resolution)
        applicator: PortMaskApplicator instance

    Returns:
        List of (x, y, w, h, scale, port_type, confidence) tuples
    """
    selector = InteractivePortSelector(image, applicator)
    return selector.run()


if __name__ == "__main__":
    import sys
    from port_mask_applicator import PortMaskApplicator

    path = sys.argv[1] if len(sys.argv) > 1 else "IO pics/MSI-A520m.png"
    img = cv2.imread(path)
    if img is None:
        print(f"Could not load: {path}")
        sys.exit(1)

    app = PortMaskApplicator("templates", "cutouts")
    dets = interactive_detect_ports(img, app)
    print(f"\nDetections ({len(dets)}):")
    for d in dets:
        print(f"  {d[5]}: pos=({d[0]},{d[1]}) size={d[2]}x{d[3]} "
              f"conf={d[6]:.2f}")
