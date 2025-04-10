import numpy as np
from tqdm import tqdm
import matlab.engine

def essential_matrix_estimation(kp1, kp2, matches, camera_intrinsic, eng, 
                                max_iter=5000, threshold=1e-5):
    """
    Step2: Estimates the Essential Matrix using the 5-Point Algorithm with RANSAC.
    It takes the camera intrinsic matrix, keypoints from two images, their matches,
    and a MATLAB engine instance (which must already be started). The function uses
    a RANSAC loop to find the best Essential Matrix candidate that fits the normalized
    matched keypoints.

    Allow functions:
        numpy
        tqdm (for progress tracking)
        eng.calibrated_fivepoint() (please read ./Step2/calibrated_fivepoint.m)

    Deny functions:
        cv2

    Parameters:
        kp1 (list): List of cv2.KeyPoint objects from the first image.
        kp2 (list): List of cv2.KeyPoint objects from the second image.
        matches (list): List of cv2.DMatch objects representing the matches between the images.
        camera_intrinsic (np.ndarray): Camera intrinsic matrix (3x3).
        eng: MATLAB engine object (already started).
        max_iter (int): Maximum number of RANSAC iterations (default 5000).
        threshold (float): Inlier threshold for error (default 1e-5).

    Returns:
        E_est (np.ndarray): The estimated Essential Matrix (3x3).
        inlier_p1 (np.ndarray): Inlier keypoint coordinates from the first image (N x 2).
        inlier_p2 (np.ndarray): Inlier keypoint coordinates from the second image (N x 2).
        best_inlier_idx (np.ndarray): Inlier matching index (N, )
    """
    # TODO: Fill this function
    
    return E_est, inlier_p1, inlier_p2, best_inlier_idx