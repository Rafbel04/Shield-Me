# Shield-Me

Shield-Me is a computer vision tool that generates 3D-printable ATX IO shields from photographs of motherboard IO panels. It uses OpenCV template matching and contour extraction to detect ports and produce cutout masks, which are applied to a blank IO shield model and exported as an openSCAD file. 

## Introduction

Standard ATX IO shields are metal plates (164mm x 48.5mm) that cover the gap between a motherboard's rear IO panel and the PC case. When building with motherboards that lack a matching shield, or when replacing a lost one, Shield-Me can help by letting you 3D print one instead of buying one online.

The program provides a tkinter-based GUI where users load a motherboard IO image, mark detected ports with computer vision, adjust positioning and scale parameters, preview the result in real time, and export an OpenSCAD file.

Due to the proprietary nature and dimensions of IO Shields, it is *highly* recommended to find a decent image of an official IO shield to be used with the **Load Official Shield** feature. It helps immensely with lining up the ports properly. Without it, you may have to do a healthy amount of trial-and-error.

### Supported Port Types

Shield-Me recognizes 12 port types, each with known physical dimensions used for calibration and cutout sizing:

| Port Type | Key | Dimensions (mm) |
|---|---|---|
| USB Type-A | `u` | 12.4 x 4.5 |
| USB Type-C | `c` | 8.9 x 3.2 |
| HDMI | `h` | 14.0 x 4.6 |
| DisplayPort | `d` | 16.1 x 4.4 |
| VGA | `v` | 30.8 x 12.0 |
| DVI | `i` | 37.0 x 13.0 |
| PS/2 | `p` | 13.0 x 13.0 |
| Audio (3.5mm) | `a` | 6.5 x 6.5 |
| Optical Audio | `o` | 8.0 x 8.0 |
| Ethernet (RJ45) | `e` | 15.9 x 13.1 |
| Screw Hole | `s` | 5.0 x 5.0 |
| Wi-Fi Antenna | `w` | 10.0 x 10.0 |

### Dependencies

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Pillow (PIL)
- tkinter (included with standard Python installations)
- OpenSCAD (https://openscad.org/downloads.html)

## Process

Shield-Me operates in three sequential phases: detection, generation, and export.

### Phase A: Port Detection

1. **Image loading.** The user selects a photograph of the motherboard IO panel through the GUI file picker.

2. **Interactive marking.** An OpenCV window opens where the user draws bounding boxes around each port. Before drawing, the user selects the port type using the corresponding keyboard shortcut (listed in the on-screen overlay). The drawn region is passed to the template matching engine for refinement.

3. **Template matching.** For each user-drawn region, the program performs multi-scale template matching (`cv2.matchTemplate` with `TM_CCOEFF_NORMED`) against all reference images for that port type. The scale range spans 0.3x to 3.0x in 0.1 increments. The best match above the confidence threshold (adjustable with `[` and `]` keys) is used. If no match exceeds the threshold, the raw user-drawn box is used as a fallback (displayed with a dashed outline). Right-clicking a detection removes it; pressing `z` undoes the last action.

4. **Scale calibration.** If a USB Type-A port is among the detections, its pixel width is divided by the known physical width (12.4mm) to derive a `pixels_per_mm` calibration factor. This factor is used in Phase B to convert pixel coordinates to millimeter coordinates. 

5. **Configuration persistence.** Detections are saved to `saved_configs/<image_name>.json` so the user can reload them later without re-marking.

### Phase B: Mask Generation and Contour Fitting

1. **Cutout mask creation.** Pre-made binary cutout masks from the `cutouts/` directory are scaled to match each detection's size and composited onto a blank image. All detections of the same port type receive a uniform cutout size based on the widest detection for visual consistency.

2. **Image processing.** The composite mask undergoes median blur (3x3), binary thresholding, and morphological closing/opening to clean noise and close small gaps.

3. **Contour extraction.** `cv2.findContours` with `RETR_EXTERNAL` extracts outer contours from the cleaned mask. Each contour is simplified using `cv2.approxPolyDP` with an epsilon of 2% of the arc length.

4. **Coordinate transformation.** Contours are converted from pixel space to millimeter space using the calibrated `pixels_per_mm` factor. The Y-axis is flipped (image coordinates are Y-down; OpenSCAD is Y-up). Contours are anchored relative to the shield's bottom-left corner using configurable offsets (left anchor, bottom anchor).

5. **Boundary fitting.** The usable area of the shield is defined as a 3.8mm inset from the physical plate edges, yielding a region from (3.8, 3.8) to (160.2, 44.7) in mm. If contours exceed the right boundary, a correction scale is applied to fit them within the usable area.

6. **UI adjustments.** The GUI provides sliders for:
   - Scale multiplier: adjusts the `pixels_per_mm` factor (values > 1.0 grow contours, < 1.0 shrink them)
   - Left/bottom anchor offsets: shift all contours relative to the plate edges
   - Per-port-type cutout scale: independently resize cutouts for specific port types
   - Contour repositioning: click a contour in the shield preview to select it, then use arrow keys to nudge it in 0.25mm increments

7. **Live preview.** The shield preview canvas renders the physical plate outline (solid), the usable area boundary (dashed blue), and all fitted contours (red, or green for the selected contour). An optional (but highly recommended) official shield image can be loaded as a background overlay for visual comparison.

### Phase C: SCAD Export

1. **File generation.** The fitted contours are written to an OpenSCAD `.scad` file. Each contour becomes a `polygon()` call inside a `ports()` module with coordinates in millimeters.

2. **Embedded shield geometry.** The blank IO shield is embedded directly in the SCAD file as an OpenSCAD `polyhedron()` definition. This geometry was generated from a reference STL using `stl_to_polyhedron.py` and stored in `shield_polyhedron_template.py`. This was done to ensure a one-file solution by removing the dependency on the STL file.

3. **Boolean subtraction.** The final SCAD structure uses a CSG `difference()` operation: the shield body minus the extruded port cutouts (`linear_extrude` at 5mm height). This produces the shield with port openings cut through.

4. **Settings persistence.** UI parameters (scale multiplier, anchor offsets, cutout scales, contour offsets) are saved alongside the detection config in `saved_configs/<image_name>.settings.json` automatically when exporting to SCAD, allowing the user to reload and continue refining a previous session.

## Output

The primary output is a standalone `.scad` file that can be opened directly in [OpenSCAD](https://openscad.org/). The file structure is:

```
// Auto-generated by Shield-Me

module ports() {
  polygon(points=[[x1, y1], [x2, y2], ...]);   // Port cutout 1
  polygon(points=[[x1, y1], [x2, y2], ...]);   // Port cutout 2
  ...
}

// Blank IO Shield Geometry (162mm x 48mm x 3mm)
module shield() {
  polyhedron(
    points = [ ... ],
    faces = [ ... ]
  );
}

difference() {
  shield();
  linear_extrude(height=5.0) ports();
}
```

In OpenSCAD, the user can render the model (`F6`), inspect it visually, and export to STL (`F7`) for slicing and 3D printing.

### Physical Specifications

| Parameter | Value |
|---|---|
| Shield plate dimensions | 164mm x 48.5mm |
| Usable cutout area | 156.4mm x 40.9mm (3.8mm inset) |
| Shield thickness | 3mm |
| Cutout extrusion height | 5mm |

### Usage

```
python UI.py OR shield-me-vx.x.bat
```

1. Click **Select IO Image** and choose a motherboard IO panel photograph. (Can be found in the "gallery" section of the MoBo's manufacturer webpage)
2. Click **Detect Ports** to open the interactive marking window.
3. Select port types with keyboard shortcuts and draw bounding boxes around each port. Use **[** and **]** to adjust CV detection threshold.
4. Press **Enter** or **q** to finalize detections.
5. Click **Load Official Shield** and select a cropped photo of a genuine IO shield.
6. Adjust parameters (scale, anchors, cutout sizes) in the GUI and click **Apply / Regenerate Preview** to line up with official shield image.
7. Click individual contours in the shield preview and use arrow keys to fine-tune positioning.
8. Click **Save SCAD File** to export.
9. Open the `.scad` file in OpenSCAD, render, and export to STL for printing.
