import cv2
import json
import numpy as np
import os
from pathlib import Path

class PortMaskApplicator:
    """
    Detects ports on motherboard IO panels and applies pre-made cutout masks.

    Directory structure expected:
    templates/
        usb/
            reference.png       # Reference image of the port for detection
        hdmi/
            reference.png
        ...
    cutouts/
        usb.png                # Pre-made cutout mask (white with black port shape)
        hdmi.png
        ...
    """

    # Known physical dimensions (width_mm, height_mm) of what each template depicts.
    # These must match what the template image actually shows (housing, bezel, etc.),
    # not just the bare connector opening.
    PORT_PHYSICAL_DIMENSIONS = {
        'usb': (12.4, 4.5),       # USB Type-A receptacle housing
        'usb_c': (8.9, 3.2),      # USB Type-C receptacle
        'hdmi': (14.0, 4.6),      # HDMI Type-A
        'dp': (16.1, 4.4),        # DisplayPort
        'vga': (30.8, 12.0),      # VGA D-Sub 15 housing
        'dvi': (37.0, 13.0),      # DVI housing
        'ps2': (13.0, 13.0),      # PS/2 circular housing
        'audio': (6.5, 6.5),      # 3.5mm audio jack
        'optical_audio': (8.0, 8.0),  # Optical S/PDIF
        'ethernet': (15.9, 13.1),  # RJ45 jack housing
        'screw': (5.0, 5.0),      # Screw hole
        'wifi': (10.0, 10.0),     # WiFi antenna connector
    }

    def __init__(self, templates_dir="templates", cutouts_dir="cutouts"):
        self.templates_dir = templates_dir
        self.cutouts_dir = cutouts_dir
        self.port_configs = {}
        self.load_port_configurations()
    
    def load_port_configurations(self):
        """Load all port templates and their corresponding cutouts.

        Each port type folder can contain multiple template .png files
        (e.g. ethernet.png, ethernet2.png, ethernet_angled.png).
        All will be used during detection to improve matching across
        different manufacturers and angles.
        """
        if not os.path.exists(self.templates_dir):
            print(f"Warning: Templates directory '{self.templates_dir}' not found.")
            return

        if not os.path.exists(self.cutouts_dir):
            print(f"Warning: Cutouts directory '{self.cutouts_dir}' not found.")
            return

        # Port types to look for
        port_types = ['usb', 'vga', 'hdmi', 'dp', 'usb_c', 'dvi', 'ps2',
                     'audio', 'optical_audio', 'ethernet', 'screw']

        for port_type in port_types:
            template_dir = os.path.join(self.templates_dir, port_type)
            cutout_path = os.path.join(self.cutouts_dir, f"{port_type}.png")

            if not os.path.isdir(template_dir) or not os.path.exists(cutout_path):
                continue

            cutout = cv2.imread(cutout_path, cv2.IMREAD_GRAYSCALE)
            if cutout is None:
                continue

            # Load all .png files from the template directory
            templates_gray = []
            for fname in sorted(os.listdir(template_dir)):
                if not fname.lower().endswith('.png'):
                    continue
                tpath = os.path.join(template_dir, fname)
                timg = cv2.imread(tpath)
                if timg is not None:
                    templates_gray.append(cv2.cvtColor(timg, cv2.COLOR_BGR2GRAY))

            if templates_gray:
                self.port_configs[port_type] = {
                    'templates_gray': templates_gray,
                    'cutout': cutout,
                }
                sizes = ", ".join(f"{t.shape[1]}x{t.shape[0]}" for t in templates_gray)
                print(f"Loaded {port_type}: {len(templates_gray)} template(s) [{sizes}], "
                      f"cutout {cutout.shape[1]}x{cutout.shape[0]}")

        if not self.port_configs:
            print("\nNo port configurations loaded!")
            print("Please set up the following directory structure:")
            print("  templates/")
            print("    usb/usb.png, usb2.png, ...")
            print("    hdmi/hdmi.png, hdmi2.png, ...")
            print("    ...")
            print("  cutouts/")
            print("    usb.png")
            print("    hdmi.png")
            print("    ...")
    
    def detect_ports(self, image, threshold=0.7, scales=None, nms_threshold=0.3):
        """
        Detect all configured ports in the image.
        
        Args:
            image: Input BGR image
            threshold: Matching confidence threshold (0-1)
            scales: List of scale factors to try
            nms_threshold: IoU threshold for non-maximum suppression
            
        Returns:
            List of detections: [(x, y, w, h, scale, port_type, confidence), ...]
        """
        if scales is None:
            scales = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        all_detections = []
        
        print("\nDetecting ports...")
        print("-" * 70)
        
        for port_type, config in self.port_configs.items():
            port_detections = []

            for template_gray in config['templates_gray']:
                for scale in scales:
                    # Resize template
                    scaled_w = int(template_gray.shape[1] * scale)
                    scaled_h = int(template_gray.shape[0] * scale)

                    if scaled_w > gray_image.shape[1] or scaled_h > gray_image.shape[0]:
                        continue

                    scaled_template = cv2.resize(template_gray, (scaled_w, scaled_h))

                    # Template matching
                    result = cv2.matchTemplate(gray_image, scaled_template, cv2.TM_CCOEFF_NORMED)

                    # Find matches above threshold
                    locations = np.where(result >= threshold)

                    for pt in zip(*locations[::-1]):
                        confidence = result[pt[1], pt[0]]
                        port_detections.append((
                            pt[0], pt[1],           # x, y
                            scaled_w, scaled_h,     # width, height
                            scale,                  # scale factor
                            port_type,
                            float(confidence)
                        ))
            
            if port_detections:
                # Apply NMS per port type
                port_detections = self._non_max_suppression(port_detections, nms_threshold)
                all_detections.extend(port_detections)
                print(f"  {port_type}: {len(port_detections)} detected")
        
        print(f"\nTotal detections: {len(all_detections)}")
        return all_detections

    def detect_port_in_region(self, image, region, port_type, threshold=0.5):
        """
        Run template matching for a single port type within a user-drawn region.

        The user draws a rough bounding box; this method searches inside it
        with a wide scale range to find the best match.

        Args:
            image: Full BGR image
            region: (x, y, w, h) bounding box in original image coordinates
            port_type: Which port type to search for (e.g., 'usb')
            threshold: Minimum confidence to accept a match

        Returns:
            Detection tuple (x, y, w, h, scale, port_type, confidence) in
            full-image coordinates, or None if no match found.
        """
        if port_type not in self.port_configs:
            return None

        rx, ry, rw, rh = region
        # Clamp region to image bounds
        img_h, img_w = image.shape[:2]
        rx = max(0, rx)
        ry = max(0, ry)
        rw = min(rw, img_w - rx)
        rh = min(rh, img_h - ry)

        if rw < 5 or rh < 5:
            return None

        crop = image[ry:ry+rh, rx:rx+rw]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Wide scale range -- safe because the search region is small
        scales = [i / 10.0 for i in range(3, 31)]  # 0.3 to 3.0

        best = None  # (confidence, x, y, w, h, scale)
        best_area = 0

        for template_gray in self.port_configs[port_type]['templates_gray']:
            for scale in scales:
                scaled_w = int(template_gray.shape[1] * scale)
                scaled_h = int(template_gray.shape[0] * scale)

                if scaled_w > gray_crop.shape[1] or scaled_h > gray_crop.shape[0]:
                    continue
                if scaled_w < 3 or scaled_h < 3:
                    continue

                scaled_template = cv2.resize(template_gray, (scaled_w, scaled_h))
                result = cv2.matchTemplate(gray_crop, scaled_template,
                                           cv2.TM_CCOEFF_NORMED)

                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val >= threshold:
                    area = scaled_w * scaled_h
                    if best is None or area > best_area:
                        best = (max_val, max_loc[0], max_loc[1],
                                scaled_w, scaled_h, scale)
                        best_area = area

        if best is None:
            return None

        conf, local_x, local_y, det_w, det_h, det_scale = best
        # Convert local crop coordinates to full-image coordinates
        full_x = rx + local_x
        full_y = ry + local_y

        return (full_x, full_y, det_w, det_h, det_scale, port_type, float(conf))

    def calibrate_from_detections(self, detections):
        """
        Derive pixels_per_mm from detected USB Type-A port.

        Uses the highest-confidence USB detection and its known
        physical width to calculate the image scale.

        Args:
            detections: List of (x, y, w, h, scale, port_type, confidence)

        Returns:
            pixels_per_mm (float) or None if no USB detected
        """
        usb_dets = [d for d in detections if d[5] == 'usb']

        if not usb_dets:
            print("WARNING: No USB Type-A port detected. Cannot calibrate scale.")
            return None

        # Pick the highest-confidence detection
        best = max(usb_dets, key=lambda d: d[6])
        det_x, det_y, det_w, det_h, det_scale, det_type, det_conf = best

        phys_w, phys_h = self.PORT_PHYSICAL_DIMENSIONS['usb']

        # Use width as primary calibration axis
        pixels_per_mm = det_w / phys_w

        print(f"\nUSB calibration: {det_w}px wide / {phys_w}mm = "
              f"{pixels_per_mm:.4f} px/mm (confidence: {det_conf:.3f})")

        return pixels_per_mm

    def filter_by_physical_size(self, detections, pixels_per_mm, tolerance=0.02):
        """
        Reject detections whose real-world size doesn't match expectations.

        For each detection, convert its pixel width to mm using the calibrated
        scale and compare against the expected physical width for that port
        type. Detections outside the tolerance are discarded.

        Args:
            detections: List of (x, y, w, h, scale, port_type, confidence)
            pixels_per_mm: Calibrated scale factor
            tolerance: Allowed fractional deviation (0.4 = +/-40%)

        Returns:
            Filtered list of detections
        """
        kept = []
        rejected = []

        for det in detections:
            x, y, det_w, det_h, scale, port_type, confidence = det

            if port_type not in self.PORT_PHYSICAL_DIMENSIONS:
                kept.append(det)
                continue

            expected_w, expected_h = self.PORT_PHYSICAL_DIMENSIONS[port_type]
            actual_w_mm = det_w / pixels_per_mm

            # Check against expected width, and also expected height
            # to handle rotated ports (e.g. vertical HDMI)
            ratio_w = actual_w_mm / expected_w
            ratio_h = actual_w_mm / expected_h
            if abs(ratio_w - 1.0) <= tolerance or abs(ratio_h - 1.0) <= tolerance:
                kept.append(det)
            else:
                rejected.append((port_type, actual_w_mm, expected_w, confidence))

        if rejected:
            print(f"\nFiltered out {len(rejected)} false detection(s):")
            for ptype, actual, expected, conf in rejected:
                print(f"  {ptype}: measured {actual:.1f}mm, "
                      f"expected {expected:.1f}mm (conf {conf:.3f})")

        print(f"Detections after size filter: {len(kept)}")
        return kept

    def _non_max_suppression(self, detections, threshold=0.3):
        """Apply non-maximum suppression to remove overlapping detections."""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x[6], reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            detections = [
                det for det in detections
                if self._iou(current, det) < threshold
            ]
        
        return keep
    
    def _iou(self, det1, det2):
        """Calculate Intersection over Union between two detections."""
        x1, y1, w1, h1 = det1[:4]
        x2, y2, w2, h2 = det2[:4]
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def create_clean_mask(self, image_shape, detections, cutout_scales=None):
        """
        Create a clean binary mask by applying pre-made cutouts at detected positions.
        
        Args:
            image_shape: Shape of original image (height, width, channels)
            detections: List of port detections
            
        Returns:
            Clean binary mask (BLACK background, WHITE ports - CV detects white as objects)
        """
        h, w = image_shape[:2]

        # Start with BLACK background (CV detects white objects)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Pre-compute a uniform cutout size per port type.
        # Use the (width, height) from the widest detection of each type so
        # that all ports of the same kind get identical cutouts.
        uniform_sizes = {}
        for x, y, det_w, det_h, scale, port_type, confidence in detections:
            if port_type not in self.port_configs:
                continue
            if port_type not in uniform_sizes or det_w > uniform_sizes[port_type][0]:
                uniform_sizes[port_type] = (det_w, det_h)

        print("\nApplying cutouts...")
        print("-" * 70)
        for port_type, (uw, uh) in sorted(uniform_sizes.items()):
            print(f"  Uniform size for {port_type}: {uw}x{uh}px")

        for x, y, det_w, det_h, scale, port_type, confidence in detections:
            if port_type not in self.port_configs:
                continue

            cutout = self.port_configs[port_type]['cutout']
            uni_w, uni_h = uniform_sizes[port_type]

            if cutout_scales and port_type in cutout_scales:
                cs = cutout_scales[port_type]
                uni_w = int(uni_w * cs)
                uni_h = int(uni_h * cs)

            # Scale the cutout to the uniform size
            scaled_cutout = cv2.resize(cutout, (uni_w, uni_h),
                                      interpolation=cv2.INTER_NEAREST)

            # Center the uniform cutout on the detection's center
            det_cx = x + det_w // 2
            det_cy = y + det_h // 2
            place_x = det_cx - uni_w // 2
            place_y = det_cy - uni_h // 2

            # Calculate placement bounds
            y1 = max(0, place_y)
            y2 = min(h, place_y + uni_h)
            x1 = max(0, place_x)
            x2 = min(w, place_x + uni_w)

            # Calculate cutout bounds (in case placement is at image edge)
            cy1 = 0 if place_y >= 0 else -place_y
            cy2 = scaled_cutout.shape[0] if place_y + uni_h <= h else h - place_y
            cx1 = 0 if place_x >= 0 else -place_x
            cx2 = scaled_cutout.shape[1] if place_x + uni_w <= w else w - place_x

            # Apply cutout: where cutout is WHITE (255), make mask WHITE
            cutout_region = scaled_cutout[cy1:cy2, cx1:cx2]
            mask_region = mask[y1:y2, x1:x2]

            # Blend: keep the BRIGHTER value (white ports win)
            mask[y1:y2, x1:x2] = np.maximum(mask_region, cutout_region)

            print(f"  Applied {port_type} cutout at ({place_x}, {place_y}), "
                  f"size {uni_w}x{uni_h}, confidence {confidence:.3f}")

        return mask
    
    def visualize_detections(self, image, detections):
        """Draw bounding boxes on image for visualization."""
        vis = image.copy()
        
        colors = {
            'usb': (255, 0, 0),          # Blue
            'vga': (128, 0, 128),        # Purple
            'hdmi': (0, 255, 0),         # Green
            'dp': (0, 165, 255),         # Orange
            'usb_c': (255, 255, 0),      # Cyan
            'dvi': (203, 192, 255),      # Pink
            'ps2': (147, 20, 255),       # Deep pink
            'audio': (255, 0, 255),      # Magenta
            'optical_audio': (180, 105, 255),  # Hot pink
            'ethernet': (0, 255, 255),   # Yellow
            'screw': (0, 128, 128),      # Dark yellow
        }
        
        for x, y, w, h, scale, port_type, confidence in detections:
            color = colors.get(port_type, (0, 0, 255))
            
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            
            label = f"{port_type}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(vis, (x, y - label_size[1] - 10),
                         (x + label_size[0], y), color, -1)
            cv2.putText(vis, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis


def process_motherboard_io(image_path, output_mask_path="output_mask.png",
                           templates_dir="templates", cutouts_dir="cutouts",
                           threshold=0.7, show_preview=True):
    """
    Main function to detect ports and create a clean mask.
    
    Args:
        image_path: Path to motherboard IO image
        output_mask_path: Where to save the generated mask
        templates_dir: Directory containing port reference images
        cutouts_dir: Directory containing pre-made cutout masks
        threshold: Detection confidence threshold (0-1)
        show_preview: Whether to display preview windows
    """
    print("="*70)
    print("PORT DETECTION AND MASK GENERATION")
    print("="*70)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"\nInput image: {image_path}")
    print(f"Size: {image.shape[1]}x{image.shape[0]}")
    
    # Initialize applicator
    applicator = PortMaskApplicator(templates_dir, cutouts_dir)
    
    if not applicator.port_configs:
        print("\nERROR: No port configurations loaded!")
        print("Please set up templates and cutouts directories.")
        return None
    
    # Detect ports
    detections = applicator.detect_ports(image, threshold=threshold)
    
    if not detections:
        print("\nWARNING: No ports detected!")
        print("Try lowering the threshold (current: {})".format(threshold))
        return None
    
    # Create clean mask
    mask = applicator.create_clean_mask(image.shape, detections)
    
    # Save mask
    cv2.imwrite(output_mask_path, mask)
    print(f"\n✓ Mask saved to: {output_mask_path}")
    
    # Show preview
    if show_preview:
        vis = applicator.visualize_detections(image, detections)
        
        # Resize for display if needed
        max_width = 1200
        if vis.shape[1] > max_width:
            scale = max_width / vis.shape[1]
            vis = cv2.resize(vis, (max_width, int(vis.shape[0] * scale)))
            mask_display = cv2.resize(mask, (max_width, int(mask.shape[0] * scale)))
        else:
            mask_display = mask
        
        cv2.imshow("Detected Ports", vis)
        cv2.imshow("Generated Mask", mask_display)
        print("\nPress any key to close preview...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return mask


if __name__ == "__main__":
    # Example usage
    mask = process_motherboard_io(
        image_path="motherboard_io.jpg",
        output_mask_path="clean_mask.png",
        templates_dir="templates",
        cutouts_dir="cutouts",
        threshold=0.7,
        show_preview=True
    )
