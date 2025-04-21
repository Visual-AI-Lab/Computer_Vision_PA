import numpy as np
from tqdm import tqdm
import random
import matlab.engine

def three_point_algorithm(matches, next_matches, inlier_idx, initial_point, add_image_kp, camera_intrinsic, eng, threepoint_threshold=1e-4, threepoint_max_iter=1000):
    """
    Estimate the projection matrix of a third image via P3P and RANSAC.

    Parameters:
        matches (list[cv2.DMatch]): Matches between image1 and image2.
        next_matches (list[cv2.DMatch]): Matches between image2 and image3.
        inlier_idx (np.ndarray): Indices of inliers from calculate_inlier_points.
        initial_point (np.ndarray): 3D points from stereo reconstruction (Nx3).
        add_image_kp (list[cv2.KeyPoint]): Keypoints from image3.
        camera_intrinsic (np.ndarray): Intrinsic camera matrix (3x3).
        eng (matlab.engine): MATLAB engine instance.
        threepoint_threshold (float): Reprojection error threshold.
        threepoint_max_iter (int): Maximum RANSAC iterations.

    Returns:
        best_P (np.ndarray): Best estimated projection matrix (3x4).
    """
    # TODO: Fill this function
    
    return best_P

def calculate_inlier_points(EM1, EM2, kp1, kp2, matches, camera_intrinsic, threshold=1e-2):
    """
    Identify inlier keypoint pairs between two images given their essential matrices.

    Parameters:
        EM1 (np.ndarray): Projection matrix of image 1 (3x4).
        EM2 (np.ndarray): Projection matrix of image 2 (3x4).
        kp1 (list[cv2.KeyPoint]): Keypoints from image 1.
        kp2 (list[cv2.KeyPoint]): Keypoints from image 2.
        matches (list[cv2.DMatch]): Matches between descriptors of image1 and image2.
        camera_intrinsic (np.ndarray): Intrinsic camera matrix (3x3).
        threshold (float): Epipolar error threshold for inlier selection.

    Returns:
        inlier_p1 (np.ndarray): Array of inlier points from image 1 (Nx2).
        inlier_p2 (np.ndarray): Array of inlier points from image 2 (Nx2).
        inlier_idx (np.ndarray): Indices of inlier matches.
    """
    # TODO: Fill this function
    E = convert_camera_matrix(EM1, EM2)
    
    return inlier_p1, inlier_p2, inlier_idx

def convert_camera_matrix(Rt0, Rt1):
    """
    Convert two camera projection matrices to the essential matrix E.

    Parameters:
        Rt0 (np.ndarray): Projection matrix of camera 0 (3x4).
        Rt1 (np.ndarray): Projection matrix of camera 1 (3x4).

    Returns:
        E (np.ndarray): Essential matrix (3x3).
    """
    R0, t0 = Rt0[:, :3], Rt0[:, 3]
    R1, t1 = Rt1[:, :3], Rt1[:, 3]
    R = R1 @ R0.T
    t = t1 - R @ t0
    t_cross = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    E = t_cross @ R
    return E