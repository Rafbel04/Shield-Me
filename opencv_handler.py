import cv2
import json
import numpy as np

def _get_contour_count(sil, min_area=100):
    """
    Count meaningful contours in a silhouette image.
    Filters out tiny noise contours below min_area pixels.
    """
    mask = cv2.medianBlur(sil, 3)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter out tiny contours (noise)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    return len(contours), contours


def calibrate_and_crop(image, min_contours=3, step=5):
    """
    Remove the reference line from the bottom of manufacturer images by
    progressively cropping until the single connected contour (ports + line)
    splits into multiple independent port shapes.

    Args:
        image: Input BGR image with reference line at bottom
        min_contours: Minimum number of contours to consider the line removed
        step: Pixel step size for coarse scan (refined to 1px after)

    Returns:
        tuple: (cropped_image, crop_y) or (None, None) if no line found
    """
    h, w = image.shape[:2]

    # Convert to silhouette for contour analysis
    sil_full = make_silhouette(image)

    # Get baseline contour count with full image
    baseline_count, baseline_cnts = _get_contour_count(sil_full)
    print(f"Full image contour count: {baseline_count}")

    if baseline_count >= min_contours:
        # Already has multiple contours - no reference line to remove
        print("Image already has multiple contours. No reference line to remove.")
        return None, None

    if not baseline_cnts:
        print("Warning: No contours found in image.")
        return None, None

    # Coarse scan: crop from bottom in steps, looking for contour split
    crop_y_coarse = None
    for crop_y in range(h - 1, int(h * 0.5), -step):
        sil_cropped = make_silhouette(image[:crop_y, :])
        sil_cropped = add_bottom_padding(sil_cropped)
        count, _ = _get_contour_count(sil_cropped)

        if count >= min_contours:
            crop_y_coarse = crop_y
            break

    if crop_y_coarse is None:
        print("Warning: Could not find reference line by contour splitting.")
        return None, None

    # Fine scan: refine to single-pixel accuracy
    fine_start = min(crop_y_coarse + step, h - 1)
    crop_y_final = crop_y_coarse

    for crop_y in range(fine_start, crop_y_coarse - 1, -1):
        sil_cropped = make_silhouette(image[:crop_y, :])
        sil_cropped = add_bottom_padding(sil_cropped)
        count, _ = _get_contour_count(sil_cropped)

        if count >= min_contours:
            crop_y_final = crop_y
            break

    print(f"\nReference line removed at Y={crop_y_final}")

    # Crop the original image at the split point
    cropped_image = image[:crop_y_final, :]

    print(f"Cropped image from {h} to {crop_y_final} pixels height")
    print(f"Removed {h - crop_y_final} pixels from bottom\n")

    return cropped_image, crop_y_final

def process_image(image_path, auto_crop=True):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    # Try to crop the reference line from the bottom
    if auto_crop:
        cropped, crop_y = calibrate_and_crop(image)
        if cropped is not None:
            image = cropped

    # turns image into silhouette
    sil = make_silhouette(image)

    # add padding to bottom of image to ensure closed edges
    sil = add_bottom_padding(sil)

    mask = cv2.medianBlur(sil, 3)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # create kernel used to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))

    # close small breaks
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_closed = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # generate contour lines
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    straightened = [cv2.approxPolyDP(c, 0.02, True) for c in contours]

    bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(bgr, straightened, -1, (0,255,0), 2)
    cv2.imshow("Contours", bgr)
    print(len(contours))
    return straightened

def add_bottom_padding(img, pad_h=10, color=(0,0,0)):
    """
    Add a constant-color border to the bottom of an image.
    pad_h: number of pixels to add
    color: BGR tuple for the border (white = (255,255,255))
    """
    # top, bottom, left, right
    top, bottom, left, right = 0, pad_h, 0, 0
    padded = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )
    return padded

def contours_to_scad(contours, scale=1.0, extrude_h=5.0, filename="output.scad"):
    """
    contours: list of numpy arrays from approxPolyDP
    img_h: original image height (needed to flip Y axis)
    scale: map pixel→SCAD unit scale
    extrude_h: how tall to extrude
    """
    scad = []
    scad.append("// Auto-generated from OpenCV contours")
    scad.append("module ports() {")
    for cnt in contours:
        # build list of [x, y] flipping y so SVG/OpenSCAD coords match
        pts = [[float(pt[0][0]) * scale, (float(pt[0][1]) * scale)] 
               for pt in cnt]
        scad.append(f"  polygon(points={json.dumps(pts)});")
    scad.append("}")
    #scad.append(f"linear_extrude(height={extrude_h}) ports();")
    scad.append("""
    module shield(){
        import("Blank_IO_Shield-No_Grill.stl", center=true);
    }
    difference(){
        shield();
        linear_extrude(height=5.0) ports();
    }""")

    with open(filename, "w") as f:
        f.write("\n".join(scad))

    print(f"OpenSCAD script written to {filename}")

def make_silhouette(image):
    # Handle grayscale images (2D arrays)
    if len(image.shape) == 2:
        # Already grayscale
        _, mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    elif image.shape[2] == 4:
        # RGBA image - use alpha channel
        b, g, r, a = cv2.split(image)
        _, mask = cv2.threshold(a, 1, 255, cv2.THRESH_BINARY)
    else:
        # RGB image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    h, w = mask.shape
    sil = np.ones((h, w), dtype=np.uint8) * 0
    sil[mask==255] = 255
    return sil

def fit_into_IO_Shield(cnts, dim=[[8, 156], [4, 44.5]], pixels_per_mm=None, left_anchor_mm=8.0):
    """
    Scale and position contours to fit within standard IO shield dimensions.

    Args:
        cnts: List of contours
        dim: [[x_min, x_max], [y_min, y_max]] in mm
        pixels_per_mm: Calibrated scale factor. If None, will auto-calculate from contours.
        left_anchor_mm: When calibrated, distance from the plate's left edge to the
                        leftmost port (mm). This corresponds to the "top" of the IO
                        shield when the PC is standing upright.

    Returns:
        Scaled and positioned contours in mm coordinates
    """
    cnts = [c.astype(np.float64, copy=True) for c in cnts]

    extLeft, extBot, extRight, extTop = getContourExtremes(cnts)

    if pixels_per_mm is not None:
        # Use calibrated scale: convert pixels to mm
        scale = 1.0 / pixels_per_mm
        print(f"\nUsing calibrated scale: {scale:.6f} mm/pixel")
        print(f"(pixels_per_mm: {pixels_per_mm:.4f})")
    else:
        # Fallback: auto-calculate scale to fit width
        scale = (dim[0][1]-dim[0][0]) / (extRight-extLeft)
        print(f"\nNo calibration data - auto-scaling to fit width")
        print(f"Scale: {scale:.6f} mm/pixel")

    fitCheck = (extTop-extBot)*scale < (dim[1][1] - dim[1][0])
    if not fitCheck:
        raise ValueError("ERROR: Mask is too tall, cannot fit into IO shield!")

    for cnt in cnts:
        cnt[:,:,:2] = cnt[:,:,:2] * scale

    newExtLeft, extBot, extRight, extTop = getContourExtremes(cnts)

    # Flip Y axis (image Y increases downward, SCAD Y increases upward)
    pivotPoint = (extTop + extBot)/2

    for cnt in cnts:
        cnt[:,:,1] = 2.0 * pivotPoint - cnt[:, :, 1]

    # Recalculate extremes after flip
    newExtLeft, extBot, extRight, extTop = getContourExtremes(cnts)

    if pixels_per_mm is not None and left_anchor_mm is not None:
        # Calibrated mode: anchor leftmost port at left_anchor_mm from plate edge
        xOffset = left_anchor_mm - newExtLeft
    else:
        # Uncalibrated fallback: align to dim left boundary
        xOffset = dim[0][0] - newExtLeft

    # Bottom-align ports
    yOffset = dim[1][0] - extBot

    for cnt in cnts:
        cnt[:,:,0] = cnt[:,:,0] + xOffset
        cnt[:,:,1] = cnt[:,:,1] + yOffset

    # Check if ports exceed right boundary and scale down if needed
    newExtLeft, extBot, extRight, extTop = getContourExtremes(cnts)
    if extRight > dim[0][1]:
        overflow = extRight - dim[0][1]
        print(f"\nWARNING: Ports extend {overflow:.2f}mm beyond right edge!")
        print(f"Applying correction scale to fit within boundaries...")

        # Calculate correction scale to fit within boundaries
        available_width = dim[0][1] - dim[0][0]
        actual_width = extRight - newExtLeft
        correction_scale = available_width / actual_width

        print(f"Correction scale: {correction_scale:.4f}")

        # Re-scale and re-position
        for cnt in cnts:
            # Scale around left edge
            cnt[:,:,0] = (cnt[:,:,0] - newExtLeft) * correction_scale + dim[0][0]

        # Verify final position
        finalLeft, finalBot, finalRight, finalTop = getContourExtremes(cnts)
        print(f"Final position: X=[{finalLeft:.2f}, {finalRight:.2f}]mm")

    return cnts

def getContourExtremes(cnts):
    
    """
    Given a list of OpenCV contours (each shaped [N,1,2]),
    return (min_x, min_y, max_x, max_y) across **all** points.
    """
    if not cnts:
        raise ValueError("No contours passed in")

    # Stack all points → shape [total_pts, 2]
    pts = np.vstack([c.reshape(-1, 2) for c in cnts])

    min_x = float(pts[:, 0].min())
    max_x = float(pts[:, 0].max())
    min_y = float(pts[:, 1].min())
    max_y = float(pts[:, 1].max())

    return min_x, min_y, max_x, max_y

if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "IO pics/MSI-A520m.png"
    image_path = name
    contours = process_image(image_path, auto_crop=True)

    # Note: For calibrated scale, use the complete_pipeline.py which
    # derives pixels_per_mm from RJ45 detection via template matching.
    contours = fit_into_IO_Shield(contours)

    # Generate SCAD file
    output_name = name.split('/')[-1].replace('.png', '.scad')
    contours_to_scad(contours, 1, 5, output_name)

    # Display the result
    cv2.waitKey(0)
    cv2.destroyAllWindows()
