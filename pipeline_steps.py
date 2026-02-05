"""
Decomposed pipeline stages for Shield-Me.

Splits the monolithic full_pipeline() into independently callable phases
so the UI can run detection once, then re-run generation with different
parameters without repeating the interactive selection step.
"""

import cv2
import numpy as np
from pathlib import Path

from port_mask_applicator import PortMaskApplicator
from interactive_selector import interactive_detect_ports
from opencv_handler import add_bottom_padding, fit_into_IO_Shield, contours_to_scad


def phase_a_detect(image_path, templates_dir="templates", cutouts_dir="cutouts",
                   detection_mode="manual", detection_threshold=0.7):
    """
    Phase A: Load image, run detection, calibrate scale.

    Args:
        image_path: Path to motherboard IO photo.
        templates_dir: Directory with port reference images.
        cutouts_dir: Directory with pre-made cutout masks.
        detection_mode: "auto" or "manual" (interactive).
        detection_threshold: Confidence threshold for auto mode.

    Returns:
        dict with keys {image, image_path, detections, pixels_per_mm,
        applicator, port_summary}, or None on failure.
    """
    input_path = Path(image_path)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {image_path}")
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return None

    print(f"Image loaded: {image.shape[1]}x{image.shape[0]}")

    applicator = PortMaskApplicator(templates_dir, cutouts_dir)
    if not applicator.port_configs:
        print("ERROR: No port configurations loaded!")
        return None

    if detection_mode == "manual":
        detections = interactive_detect_ports(
            image, applicator, image_name=input_path.name
        )
        if not detections:
            print("No ports were marked (cancelled or empty).")
            return None
    else:
        detections = applicator.detect_ports(
            image, threshold=detection_threshold
        )
        if not detections:
            print("No ports detected at current threshold.")
            return None

    pixels_per_mm = applicator.calibrate_from_detections(detections)

    if pixels_per_mm is not None and detection_mode == "auto":
        detections = applicator.filter_by_physical_size(detections, pixels_per_mm)

    # Build port summary
    port_summary = {}
    for det in detections:
        pt = det[5]
        port_summary[pt] = port_summary.get(pt, 0) + 1

    return {
        "image": image,
        "image_path": str(image_path),
        "detections": detections,
        "pixels_per_mm": pixels_per_mm,
        "applicator": applicator,
        "port_summary": port_summary,
    }


def phase_b_generate(phase_a_result, cutout_scales=None,
                     ppm_multiplier=1.0, left_anchor_mm=6.0,
                     bottom_anchor_mm=4.0):
    """
    Phase B: Create mask, extract contours, fit to IO shield.

    Pure computation -- no GUI calls.  Fast enough to call synchronously
    from the tkinter main thread.

    Args:
        phase_a_result: dict returned by phase_a_detect().
        cutout_scales: {port_type: float} per-type cutout scale (default 1.0).
        ppm_multiplier: Multiplier applied to pixels_per_mm (default 1.0).
        left_anchor_mm: Left anchor offset in mm (default 6.0).
        bottom_anchor_mm: Bottom anchor offset in mm (default 4.0).

    Returns:
        dict with keys {mask, contours_fitted, pixels_per_mm_adjusted},
        or None on failure.
    """
    image = phase_a_result["image"]
    detections = phase_a_result["detections"]
    applicator = phase_a_result["applicator"]
    raw_ppm = phase_a_result["pixels_per_mm"]

    # Adjust pixels_per_mm
    pixels_per_mm = raw_ppm * ppm_multiplier if raw_ppm is not None else None

    # Create clean mask with optional per-type cutout scaling
    mask = applicator.create_clean_mask(
        image.shape, detections, cutout_scales=cutout_scales
    )

    # Extract contours from mask
    padded = add_bottom_padding(mask, pad_h=1, color=(0, 0, 0))
    processed = cv2.medianBlur(padded, 3)
    _, processed = cv2.threshold(processed, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=2)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    straightened = [cv2.approxPolyDP(c, 0.02, True) for c in contours]

    # Fit to IO shield dimensions
    fitted = fit_into_IO_Shield(
        straightened,
        dim=[[3.8, 160.2], [3.8, 44.7]],
        pixels_per_mm=pixels_per_mm,
        left_anchor_mm=left_anchor_mm,
        bottom_anchor_mm=bottom_anchor_mm,
    )

    return {
        "mask": mask,
        "contours_fitted": fitted,
        "pixels_per_mm_adjusted": pixels_per_mm,
    }


def phase_c_save_scad(phase_b_result, output_filename):
    """
    Phase C: Write fitted contours to an OpenSCAD file.

    Args:
        phase_b_result: dict returned by phase_b_generate().
        output_filename: Path for the .scad file.

    Returns:
        True on success, False on failure.
    """
    try:
        contours_to_scad(
            phase_b_result["contours_fitted"],
            scale=1,
            extrude_h=5.0,
            filename=output_filename,
        )
        print(f"SCAD saved: {output_filename}")
        return True
    except Exception as e:
        print(f"ERROR saving SCAD: {e}")
        return False
