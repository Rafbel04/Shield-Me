#!/usr/bin/env python3
"""
Helper script to generate standard cutout masks for common ports.

This creates basic rectangular cutouts with slight padding.
You can customize these further in an image editor if needed.
"""

import cv2
import numpy as np
import os

def create_rectangular_cutout(width, height, filename, padding_percent=0.1, 
                              rounded_corners=0):
    """
    Create a rectangular cutout mask.
    
    Args:
        width: Cutout width in pixels
        height: Cutout height in pixels  
        filename: Output filename
        padding_percent: Extra clearance around port (0.1 = 10%)
        rounded_corners: Radius for rounded corners (0 = sharp corners)
    """
    # BLACK background (CV detects white objects)
    cutout = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate padding
    pad_w = int(width * padding_percent / 2)
    pad_h = int(height * padding_percent / 2)
    
    if rounded_corners > 0:
        # Create mask for rounded rectangle
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Draw filled rounded rectangle on mask
        cv2.rectangle(mask, 
                     (pad_w + rounded_corners, pad_h),
                     (width - pad_w - rounded_corners, height - pad_h),
                     255, -1)
        cv2.rectangle(mask,
                     (pad_w, pad_h + rounded_corners),
                     (width - pad_w, height - pad_h - rounded_corners),
                     255, -1)
        
        # Draw circles at corners
        cv2.circle(mask, (pad_w + rounded_corners, pad_h + rounded_corners), 
                  rounded_corners, 255, -1)
        cv2.circle(mask, (width - pad_w - rounded_corners, pad_h + rounded_corners), 
                  rounded_corners, 255, -1)
        cv2.circle(mask, (pad_w + rounded_corners, height - pad_h - rounded_corners), 
                  rounded_corners, 255, -1)
        cv2.circle(mask, (width - pad_w - rounded_corners, height - pad_h - rounded_corners), 
                  rounded_corners, 255, -1)
        
        # Use the mask directly (white port on black background)
        cutout = mask
    else:
        # Simple rectangle - draw WHITE rectangle on BLACK background
        cv2.rectangle(cutout, 
                     (pad_w, pad_h), 
                     (width - pad_w, height - pad_h), 
                     255, -1)
    
    cv2.imwrite(filename, cutout)
    print(f"Created: {filename} ({width}x{height}px)")


def create_circular_cutout(diameter, filename, padding_percent=0.1):
    """Create a circular cutout (for audio jacks, PS/2, etc.)"""
    size = diameter
    cutout = np.zeros((size, size), dtype=np.uint8)  # BLACK background
    
    # Calculate radius with padding
    radius = int((diameter / 2) * (1 + padding_percent))
    center = (size // 2, size // 2)
    
    cv2.circle(cutout, center, radius, 255, -1)  # WHITE circle
    
    cv2.imwrite(filename, cutout)
    print(f"Created: {filename} ({size}x{size}px, circular)")


def create_vga_cutout(filename):
    """Create a VGA/D-Sub shaped cutout (trapezoid-ish)."""
    width, height = 42, 28
    cutout = np.zeros((height, width), dtype=np.uint8)  # BLACK background
    
    # VGA is roughly trapezoidal
    # Define polygon points
    padding = 2
    pts = np.array([
        [padding + 2, padding],                    # Top left
        [width - padding - 2, padding],            # Top right  
        [width - padding, height - padding],       # Bottom right
        [padding, height - padding]                # Bottom left
    ], dtype=np.int32)
    
    cv2.fillPoly(cutout, [pts], 255)  # WHITE fill
    
    cv2.imwrite(filename, cutout)
    print(f"Created: {filename} ({width}x{height}px, VGA shape)")


def create_displayport_cutout(filename):
    """Create DisplayPort shaped cutout (one corner cut)."""
    width, height = 44, 22
    cutout = np.zeros((height, width), dtype=np.uint8)  # BLACK background
    
    padding = 2
    corner_cut = 4
    
    # DisplayPort has one corner cut at an angle
    pts = np.array([
        [padding, padding + corner_cut],           # Top left (cut corner)
        [padding + corner_cut, padding],           # Top left diagonal
        [width - padding, padding],                # Top right
        [width - padding, height - padding],       # Bottom right
        [padding, height - padding]                # Bottom left
    ], dtype=np.int32)
    
    cv2.fillPoly(cutout, [pts], 255)  # WHITE fill
    
    cv2.imwrite(filename, cutout)
    print(f"Created: {filename} ({width}x{height}px, DisplayPort shape)")


def generate_all_standard_cutouts(output_dir="cutouts"):
    """Generate standard cutouts for all common port types."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("GENERATING STANDARD PORT CUTOUTS")
    print("="*70)
    print()
    
    # USB Type-A (rectangular with slight rounding)
    create_rectangular_cutout(
        50, 22, 
        os.path.join(output_dir, "usb.png"),
        padding_percent=0.12,
        rounded_corners=2
    )
    
    # USB Type-C (smaller, more rounded)
    create_rectangular_cutout(
        30, 14,
        os.path.join(output_dir, "usb_c.png"),
        padding_percent=0.15,
        rounded_corners=3
    )
    
    # HDMI
    create_rectangular_cutout(
        44, 16,
        os.path.join(output_dir, "hdmi.png"),
        padding_percent=0.12,
        rounded_corners=1
    )
    
    # DisplayPort (special shape)
    create_displayport_cutout(os.path.join(output_dir, "dp.png"))
    
    # Ethernet RJ45 (taller rectangle)
    create_rectangular_cutout(
        44, 36,
        os.path.join(output_dir, "ethernet.png"),
        padding_percent=0.1,
        rounded_corners=1
    )
    
    # Audio jack (circular)
    create_circular_cutout(
        20,
        os.path.join(output_dir, "audio.png"),
        padding_percent=0.15
    )
    
    # Optical audio (square-ish)
    create_rectangular_cutout(
        22, 22,
        os.path.join(output_dir, "optical_audio.png"),
        padding_percent=0.1,
        rounded_corners=1
    )
    
    # PS/2 (circular, slightly larger)
    create_circular_cutout(
        22,
        os.path.join(output_dir, "ps2.png"),
        padding_percent=0.15
    )
    
    # VGA/D-Sub (special trapezoid shape)
    create_vga_cutout(os.path.join(output_dir, "vga.png"))
    
    # DVI (larger rectangle)
    create_rectangular_cutout(
        52, 30,
        os.path.join(output_dir, "dvi.png"),
        padding_percent=0.1,
        rounded_corners=1
    )
    
    print()
    print("="*70)
    print(f"All cutouts created in: {output_dir}/")
    print("="*70)
    print()
    print("You can now:")
    print("1. View these cutouts in an image viewer")
    print("2. Customize them in Photoshop/GIMP if needed")
    print("3. Create reference templates for detection")
    print("4. Run port detection on your motherboard images")


def create_cutout_preview(cutouts_dir="cutouts", output_file="cutouts_preview.png"):
    """Create a preview image showing all cutouts."""
    
    cutout_files = sorted([f for f in os.listdir(cutouts_dir) if f.endswith('.png')])
    
    if not cutout_files:
        print(f"No cutouts found in {cutouts_dir}")
        return
    
    # Load all cutouts
    cutouts = []
    max_h = 0
    for filename in cutout_files:
        path = os.path.join(cutouts_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            cutouts.append((filename, img))
            max_h = max(max_h, img.shape[0])
    
    # Create canvas
    cell_h = max_h + 60
    cell_w = 200
    cols = 3
    rows = (len(cutouts) + cols - 1) // cols
    
    canvas = np.ones((rows * cell_h, cols * cell_w), dtype=np.uint8) * 200
    
    # Place cutouts
    for idx, (filename, cutout) in enumerate(cutouts):
        row = idx // cols
        col = idx % cols
        
        # Center in cell
        y_offset = row * cell_h + 30
        x_offset = col * cell_w + (cell_w - cutout.shape[1]) // 2
        
        h, w = cutout.shape
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = cutout
        
        # Label
        label = filename.replace('.png', '').replace('_', ' ').title()
        size_label = f"{w}x{h}px"
        
        cv2.putText(canvas, label, (col * cell_w + 10, y_offset - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        cv2.putText(canvas, size_label, (col * cell_w + 10, y_offset - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, 100, 1)
    
    cv2.imwrite(output_file, canvas)
    print(f"\n✓ Preview saved: {output_file}")
    
    # Display
    cv2.imshow("Cutout Preview", canvas)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    # Generate all standard cutouts
    generate_all_standard_cutouts("cutouts")
    
    # Create preview
    if "--preview" in sys.argv or "-p" in sys.argv:
        create_cutout_preview("cutouts")
    else:
        print("\nTip: Run with --preview to see all cutouts in one image")
