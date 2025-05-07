import cv2
import json
import numpy as np

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # get dimensions of the image

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

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
    if image.shape[2] == 4:
        b, g, r, a = cv2.split(image)
        _, mask = cv2.threshold(a, 1, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    h, w = mask.shape
    sil = np.ones((h, w), dtype=np.uint8) * 0
    sil[mask==255] = 255
    return sil

def fit_into_IO_Shield(cnts, dim=[[5, 156], [4, 44.5]]):
    cnts = [c.astype(np.float64, copy=True) for c in cnts]

    extLeft, extBot, extRight, extTop = getContourExtremes(cnts)

    scale = (dim[0][1]-dim[0][0]) / (extRight-extLeft)
    fitCheck = (extTop-extBot)*scale < (dim[1][1] - dim[1][0])
    if not fitCheck:
        raise("ERROR: Mask is too tall, cannot fit into IO shield!")
    
    for cnt in cnts:
        cnt[:,:,:2] = cnt[:,:,:2] * scale

    newExtLeft, extBot, extRight, extTop = getContourExtremes(cnts)

    pivotPoint = (extTop + extBot)/2

    for cnt in cnts:
        cnt[:,:,1] = 2.0 * pivotPoint - cnt[:, :, 1]

    xOffset = dim[0][0] - newExtLeft
    yOffset = dim[1][0] - extBot

    for cnt in cnts:
        cnt[:,:,0] = cnt[:,:,0] + xOffset
        cnt[:,:,1] = cnt[:,:,1] + yOffset
    
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
    name = "A320M-HDV.png"
    # Example usage
    image_path = name
    contours = process_image(image_path)
    contours = fit_into_IO_Shield(contours)
    contours_to_scad(contours, 1, 5, "a320m.scad") #asus uses 0.213 scaling

    # Display the result
    cv2.waitKey(0)
    cv2.destroyAllWindows()