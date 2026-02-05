#!/usr/bin/env python3
"""
Diagnostic script to analyze mask and contour dimensions.
This will help identify why "mask too tall" error occurs.
"""

import cv2
import numpy as np
import sys

def analyze_mask(mask_path):
    """Analyze a mask file and show dimensions of detected contours."""
    
    print("="*70)
    print("MASK ANALYSIS")
    print("="*70)
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"ERROR: Could not load {mask_path}")
        return
    
    print(f"\nMask file: {mask_path}")
    print(f"Image size: {mask.shape[1]}x{mask.shape[0]} pixels")
    print(f"Min value: {mask.min()}, Max value: {mask.max()}")
    
    # Count white and black pixels
    white_pixels = np.sum(mask == 255)
    black_pixels = np.sum(mask == 0)
    total_pixels = mask.shape[0] * mask.shape[1]
    
    print(f"\nPixel distribution:")
    print(f"  White (255): {white_pixels:,} ({100*white_pixels/total_pixels:.1f}%)")
    print(f"  Black (0): {black_pixels:,} ({100*black_pixels/total_pixels:.1f}%)")
    print(f"  Other: {total_pixels - white_pixels - black_pixels:,}")
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\nContours detected: {len(contours)}")
    
    if len(contours) == 0:
        print("\n⚠ WARNING: No contours detected!")
        print("  This usually means:")
        print("  - The mask is all black (no white objects)")
        print("  - Or all white (entire image is one object)")
        return
    
    # Analyze each contour
    print("\nContour details:")
    print("-" * 70)
    
    all_points = []
    for i, cnt in enumerate(contours):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # Collect all points
        for pt in cnt:
            all_points.append(pt[0])
        
        print(f"Contour {i+1}:")
        print(f"  Position: ({x}, {y})")
        print(f"  Size: {w}x{h} pixels")
        print(f"  Area: {area:.0f} px²")
    
    # Calculate overall bounds
    if all_points:
        all_points = np.array(all_points)
        min_x = all_points[:, 0].min()
        max_x = all_points[:, 0].max()
        min_y = all_points[:, 1].min()
        max_y = all_points[:, 1].max()
        
        total_width = max_x - min_x
        total_height = max_y - min_y
        
        print("\n" + "="*70)
        print("OVERALL BOUNDS (all contours combined)")
        print("="*70)
        print(f"X range: {min_x} to {max_x} (width: {total_width} pixels)")
        print(f"Y range: {min_y} to {max_y} (height: {total_height} pixels)")
        
        # Calculate aspect ratio
        aspect_ratio = total_width / total_height if total_height > 0 else 0
        print(f"\nAspect ratio: {aspect_ratio:.2f} (width/height)")
        
        # ATX IO shield dimensions in mm
        shield_width_mm = 156 - 8  # 148mm usable width
        shield_height_mm = 44.5 - 4  # 40.5mm usable height
        shield_aspect = shield_width_mm / shield_height_mm
        
        print(f"ATX IO Shield aspect ratio: {shield_aspect:.2f}")
        
        # Check if it will fit
        print("\n" + "="*70)
        print("FIT CHECK")
        print("="*70)
        
        # Calculate scale needed to fit width
        scale = shield_width_mm / total_width
        scaled_height = total_height * scale
        
        print(f"\nTo fit width ({total_width}px → {shield_width_mm}mm):")
        print(f"  Scale factor: {scale:.4f}")
        print(f"  Scaled height: {scaled_height:.2f} mm")
        print(f"  Available height: {shield_height_mm:.2f} mm")
        
        if scaled_height <= shield_height_mm:
            print(f"  ✓ FITS! ({scaled_height:.2f} < {shield_height_mm:.2f})")
        else:
            print(f"  ✗ TOO TALL! ({scaled_height:.2f} > {shield_height_mm:.2f})")
            overflow = scaled_height - shield_height_mm
            print(f"  Overflow: {overflow:.2f} mm ({100*overflow/shield_height_mm:.1f}% over limit)")
            
            print(f"\n  Possible issues:")
            print(f"  1. Mask includes parts of the motherboard above/below IO ports")
            print(f"  2. Detection caught non-port objects")
            print(f"  3. Ports are actually stacked unusually tall")
            print(f"  4. Image perspective/angle is off")
    
    # Visualize
    print("\n" + "="*70)
    print("VISUALIZATION")
    print("="*70)
    
    # Draw all contours on the mask
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    
    # Draw overall bounding box
    if all_points.any():
        cv2.rectangle(vis, (min_x, min_y), (max_x, max_y), (0, 0, 255), 3)
        
        # Add dimension labels
        cv2.putText(vis, f"{total_width}px", 
                   (min_x, min_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(vis, f"{total_height}px", 
                   (max_x + 10, (min_y + max_y) // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Resize for display if too large
    max_display_width = 1200
    if vis.shape[1] > max_display_width:
        scale = max_display_width / vis.shape[1]
        vis = cv2.resize(vis, (max_display_width, int(vis.shape[0] * scale)))
    
    cv2.imshow("Mask Analysis", vis)
    cv2.imshow("Original Mask", cv2.resize(mask, (vis.shape[1], vis.shape[0])) if mask.shape[1] > max_display_width else mask)
    
    print("\nGreen: Individual contours")
    print("Red: Overall bounding box")
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mask_path = sys.argv[1]
    else:
        # Default to looking for temp mask
        import glob
        temp_masks = glob.glob("temp_mask_*.png")
        if temp_masks:
            mask_path = temp_masks[0]
            print(f"Found: {mask_path}")
        else:
            print("Usage: python analyze_mask.py <mask_file.png>")
            print("Or place a temp_mask_*.png file in current directory")
            sys.exit(1)
    
    analyze_mask(mask_path)
