import cv2
import numpy as np

def calculate_colors(img1_path, img2_path, inlier_p1, inlier_p2, inlier_idx):
    img1_color = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2_color = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    # Initialize an array to store the averaged colors for each inlier correspondence
    colors = []
    # For each corresponding inlier point, compute the average color from both images.
    for pt1, pt2 in zip(inlier_p1[inlier_idx], inlier_p2[inlier_idx]):
        # Convert float coordinates to integer indices
        x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
        x2, y2 = int(round(pt2[0])), int(round(pt2[1]))
        # Get the pixel color values from each image (BGR format)
        color1 = img1_color[y1, x1, :].astype(np.float32)
        color2 = img2_color[y2, x2, :].astype(np.float32)
        # Compute the average color
        avg_color = ((color1 + color2) / 2).astype(np.uint8)
        colors.append(avg_color.tolist())
    colors = np.array(colors)
    
    return colors