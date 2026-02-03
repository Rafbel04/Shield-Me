#!/usr/bin/env python3

# TODO: port cutouts are being placed on the very left of the IO shield, but that may be ok since
# there will be more auto-generated ports being added to the program

"""
Complete automated pipeline: Motherboard photo -> OpenSCAD IO shield

This integrates:
1. Port detection and clean mask generation (port_mask_applicator.py)
2. Contour extraction and fitting (opencv_handler.py)
3. OpenSCAD generation
"""

import cv2
import os
import sys
from pathlib import Path

# Import the port detector
from port_mask_applicator import PortMaskApplicator
from interactive_selector import interactive_detect_ports

# Import your existing OpenSCAD functions
# Adjust the import path if needed
try:
    from opencv_handler import (
        add_bottom_padding,
        fit_into_IO_Shield,
        contours_to_scad
    )
except ImportError:
    print("Warning: Could not import opencv_handler functions")
    print("Make sure opencv_handler.py is in the same directory or adjust the import")


def full_pipeline(motherboard_image_path, output_scad_name=None,
                  templates_dir="templates", cutouts_dir="cutouts",
                  detection_threshold=0.7, show_steps=True,
                  detection_mode="auto"):
    """
    Complete pipeline from motherboard photo to OpenSCAD file.

    Args:
        motherboard_image_path: Path to raw motherboard IO photo
        output_scad_name: Output .scad filename (auto-generated if None)
        templates_dir: Directory with port reference images
        cutouts_dir: Directory with pre-made cutout masks
        detection_threshold: Template matching threshold (0-1)
        show_steps: Show visualization at each step
        detection_mode: "auto" for template matching, "manual" for interactive selection

    Returns:
        True if successful, False otherwise
    """
    
    print("\n" + "="*70)
    print("AUTOMATED IO SHIELD GENERATION - COMPLETE PIPELINE")
    print("="*70 + "\n")
    
    # Validate input
    input_path = Path(motherboard_image_path)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {motherboard_image_path}")
        return False
    
    # Generate output filename
    if output_scad_name is None:
        output_scad_name = f"{input_path.stem}_shield.scad"
    
    temp_mask_path = f"temp_mask_{input_path.stem}.png"
    
    print(f"Input:  {motherboard_image_path}")
    print(f"Output: {output_scad_name}\n")
    
    # ========================================================================
    # STEP 1: Load image
    # ========================================================================
    print("STEP 1: Loading image...")
    print("-" * 70)
    
    image = cv2.imread(motherboard_image_path)
    if image is None:
        print(f"ERROR: Could not load image")
        return False
    
    print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    # ========================================================================
    # STEP 2: Detect ports and generate clean mask
    # ========================================================================
    print("\nSTEP 2: Detecting ports...")
    print("-" * 70)

    # Always initialize applicator (needed for cutouts and region detection)
    applicator = PortMaskApplicator(templates_dir, cutouts_dir)

    if not applicator.port_configs:
        print("\nERROR: No port configurations loaded!")
        print("\nSetup required:")
        print("  1. Run: python generate_cutouts.py")
        print("  2. Create templates/ directory with reference images")
        print("  3. See SETUP_GUIDE.md for details")
        return False

    if detection_mode == "manual":
        print("MODE: Interactive selection (draw boxes, CV refines)")
        detections = interactive_detect_ports(image, applicator, image_name=input_path.name)
        if not detections:
            print("\nNo ports were marked (cancelled or empty).")
            return False
    else:
        print("MODE: Automatic template matching")
        detections = applicator.detect_ports(image, threshold=detection_threshold)
        if not detections:
            print("\nWARNING: No ports detected!")
            print(f"Current threshold: {detection_threshold}")
            print("Try:")
            print("  - Lower threshold (e.g., 0.6 or 0.65)")
            print("  - Check that templates match your motherboard")
            print("  - Verify templates/ and cutouts/ are set up correctly")
            return False

    print(f"\n✓ {len(detections)} port(s) marked")

    # ========================================================================
    # STEP 2.5: Calibrate scale from USB port
    # ========================================================================
    print("\nSTEP 2.5: Calibrating scale from USB port...")
    print("-" * 70)

    pixels_per_mm = applicator.calibrate_from_detections(detections)

    if pixels_per_mm is not None:
        print(f"Calibrated: {pixels_per_mm:.4f} pixels/mm")
    else:
        print("Proceeding without calibration (auto-scale to width)")

    # ========================================================================
    # STEP 2.6: Filter false detections by physical size
    # ========================================================================
    if pixels_per_mm is not None and detection_mode == "auto":
        print("\nSTEP 2.6: Filtering detections by physical size...")
        print("-" * 70)
        detections = applicator.filter_by_physical_size(detections, pixels_per_mm)

    # Generate mask from filtered detections
    mask = applicator.create_clean_mask(image.shape, detections)
    cv2.imwrite(temp_mask_path, mask)
    print(f"✓ Clean mask generated: {temp_mask_path}")

    # Show visualization
    if show_steps:
        vis = applicator.visualize_detections(image, detections)

        # Resize for display if needed
        max_width = 1200
        if vis.shape[1] > max_width:
            scale = max_width / vis.shape[1]
            vis = cv2.resize(vis, (max_width, int(vis.shape[0] * scale)))
            mask_display = cv2.resize(mask, (max_width, int(mask.shape[0] * scale)))
        else:
            mask_display = mask

        cv2.imshow("Step 2: Detected Ports", vis)
        cv2.imshow("Step 2: Generated Mask", mask_display)
    
    # ========================================================================
    # STEP 3: Process mask to extract contours
    # ========================================================================
    print("\nSTEP 3: Extracting contours from mask...")
    print("-" * 70)
    
    # Add padding to ensure closed edges
    padded_mask = add_bottom_padding(mask, pad_h=1, color=(0, 0, 0))
    
    # Apply median blur
    processed = cv2.medianBlur(padded_mask, 3)
    _, processed = cv2.threshold(processed, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=2)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Approximate to straighter lines
    straightened = [cv2.approxPolyDP(c, 0.02, True) for c in contours]
    
    print(f"✓ Extracted {len(straightened)} contours")
    
    # Show contours if requested
    if show_steps:
        contour_vis = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_vis, straightened, -1, (0, 255, 0), 2)
        
        if contour_vis.shape[1] > 1200:
            scale = 1200 / contour_vis.shape[1]
            contour_vis = cv2.resize(contour_vis, 
                                    (1200, int(contour_vis.shape[0] * scale)))
        
        cv2.imshow("Step 3: Extracted Contours", contour_vis)
    
    # ========================================================================
    # STEP 4: Fit contours to IO shield dimensions
    # ========================================================================
    print("\nSTEP 4: Fitting contours to IO shield dimensions...")
    print("-" * 70)
    
    try:
        # Standard ATX IO shield dimensions in mm
        fitted_contours = fit_into_IO_Shield(
            straightened,
            dim=[[8, 156], [4, 44.5]],  # [[x_min, x_max], [y_min, y_max]]
            pixels_per_mm=pixels_per_mm,
            left_anchor_mm=6.0  # Reduced from default 8.0 to prevent right-edge clipping
        )
        print("✓ Contours fitted to standard ATX IO shield")
    except Exception as e:
        print(f"ERROR: Failed to fit contours: {e}")
        if show_steps:
            print("\nPress any key to close all windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return False
    
    # ========================================================================
    # STEP 5: Generate OpenSCAD file
    # ========================================================================
    print("\nSTEP 5: Generating OpenSCAD file...")
    print("-" * 70)
    
    try:
        contours_to_scad(fitted_contours, scale=1, extrude_h=5.0, 
                        filename=output_scad_name)
        print(f"✓ OpenSCAD file generated: {output_scad_name}")
    except Exception as e:
        print(f"ERROR: Failed to generate OpenSCAD: {e}")
        if show_steps:
            print("\nPress any key to close all windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return False
    
    # ========================================================================
    # Cleanup and finish
    # ========================================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  • OpenSCAD: {output_scad_name}")
    print(f"  • Debug mask: {temp_mask_path}")
    
    print(f"\nNext steps:")
    print(f"  1. Open {output_scad_name} in OpenSCAD")
    print(f"  2. Press F5 to preview")
    print(f"  3. Press F6 to render")
    print(f"  4. Export to STL for 3D printing")
    
    print(f"\nPort detection summary:")
    port_counts = {}
    for det in detections:
        port_type = det[5]
        port_counts[port_type] = port_counts.get(port_type, 0) + 1
    
    for port_type, count in sorted(port_counts.items()):
        print(f"  • {port_type}: {count}")
    
    if show_steps:
        print(f"\nPress any key to close all windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Optionally remove temp mask
    # os.remove(temp_mask_path)
    
    return True


def batch_process(input_dir, output_dir="output", 
                 templates_dir="templates", cutouts_dir="cutouts",
                 detection_threshold=0.7):
    """
    Process multiple motherboard images in batch.
    
    Args:
        input_dir: Directory containing motherboard images
        output_dir: Directory for output .scad files
        templates_dir: Port reference templates
        cutouts_dir: Port cutout masks
        detection_threshold: Detection threshold
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images
    input_path = Path(input_dir)
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    images = [f for f in input_path.iterdir() if f.suffix.lower() in image_exts]
    
    if not images:
        print(f"No images found in {input_dir}")
        return
    
    print(f"\nBatch processing {len(images)} images...")
    print("="*70)
    
    successful = 0
    failed = 0
    
    for img_path in images:
        output_scad = str(Path(output_dir) / f"{img_path.stem}_shield.scad")
        
        print(f"\n\nProcessing: {img_path.name}")
        print("-" * 70)
        
        result = full_pipeline(
            str(img_path),
            output_scad,
            templates_dir=templates_dir,
            cutouts_dir=cutouts_dir,
            detection_threshold=detection_threshold,
            show_steps=False  # No visualization for batch
        )
        
        if result:
            successful += 1
        else:
            failed += 1
    
    print("\n\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Successful: {successful}/{len(images)}")
    print(f"Failed: {failed}/{len(images)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Shield-Me IO Shield Generator")
    parser.add_argument("image", nargs="?", default="IO pics\\asus-tuf-b365m-gaming.png",
                        help="Path to motherboard IO photo")
    parser.add_argument("-o", "--output", default=None,
                        help="Output .scad filename")
    parser.add_argument("-m", "--mode", choices=["auto", "manual"], default="auto",
                        help="Detection mode: 'auto' for template matching, "
                             "'manual' for interactive selection")
    parser.add_argument("-t", "--threshold", type=float, default=0.70,
                        help="Detection threshold for auto mode (0-1)")
    args = parser.parse_args()

    full_pipeline(
        motherboard_image_path=args.image,
        output_scad_name=args.output,
        templates_dir="templates",
        cutouts_dir="cutouts",
        detection_threshold=args.threshold,
        show_steps=True,
        detection_mode="manual",
    )