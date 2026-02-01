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
    
    def __init__(self, templates_dir="templates", cutouts_dir="cutouts"):
        self.templates_dir = templates_dir
        self.cutouts_dir = cutouts_dir
        self.port_configs = {}
        self.load_port_configurations()
    
    def load_port_configurations(self):
        """Load all port templates and their corresponding cutouts."""
        if not os.path.exists(self.templates_dir):
            print(f"Warning: Templates directory '{self.templates_dir}' not found.")
            return
        
        if not os.path.exists(self.cutouts_dir):
            print(f"Warning: Cutouts directory '{self.cutouts_dir}' not found.")
            return
        
        # Port types to look for
        port_types = ['usb', 'vga', 'hdmi', 'dp', 'usb_c', 'dvi', 'ps2', 
                     'audio', 'optical_audio', 'ethernet']
        
        for port_type in port_types:
            template_path = os.path.join(self.templates_dir, port_type, f"{port_type}.png")
            cutout_path = os.path.join(self.cutouts_dir, f"{port_type}.png")
            
            if os.path.exists(template_path) and os.path.exists(cutout_path):
                template = cv2.imread(template_path)
                cutout = cv2.imread(cutout_path, cv2.IMREAD_GRAYSCALE)
                
                if template is not None and cutout is not None:
                    self.port_configs[port_type] = {
                        'template': template,
                        'cutout': cutout,
                        'template_gray': cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    }
                    print(f"✓ Loaded {port_type}: template {template.shape[1]}x{template.shape[0]}, "
                          f"cutout {cutout.shape[1]}x{cutout.shape[0]}")
        
        if not self.port_configs:
            print("\nNo port configurations loaded!")
            print("Please set up the following directory structure:")
            print("  templates/")
            print("    usb/reference.png")
            print("    hdmi/reference.png")
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
            # Tighter scale range for cleaner results
            scales = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        all_detections = []
        
        print("\nDetecting ports...")
        print("-" * 70)
        
        for port_type, config in self.port_configs.items():
            template_gray = config['template_gray']
            port_detections = []
            
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
    
    def create_clean_mask(self, image_shape, detections):
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
        
        print("\nApplying cutouts...")
        print("-" * 70)
        
        for x, y, det_w, det_h, scale, port_type, confidence in detections:
            if port_type not in self.port_configs:
                continue
            
            cutout = self.port_configs[port_type]['cutout']
            
            # Scale the cutout to match the detection size
            scaled_cutout = cv2.resize(cutout, (det_w, det_h), 
                                      interpolation=cv2.INTER_NEAREST)
            
            # Calculate placement bounds
            y1 = max(0, y)
            y2 = min(h, y + det_h)
            x1 = max(0, x)
            x2 = min(w, x + det_w)
            
            # Calculate cutout bounds (in case detection is at image edge)
            cy1 = 0 if y >= 0 else -y
            cy2 = scaled_cutout.shape[0] if y + det_h <= h else h - y
            cx1 = 0 if x >= 0 else -x
            cx2 = scaled_cutout.shape[1] if x + det_w <= w else w - x
            
            # Apply cutout: where cutout is WHITE (255), make mask WHITE
            cutout_region = scaled_cutout[cy1:cy2, cx1:cx2]
            mask_region = mask[y1:y2, x1:x2]
            
            # Blend: keep the BRIGHTER value (white ports win)
            mask[y1:y2, x1:x2] = np.maximum(mask_region, cutout_region)
            
            print(f"  Applied {port_type} cutout at ({x}, {y}), "
                  f"size {det_w}x{det_h}, confidence {confidence:.3f}")
        
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
