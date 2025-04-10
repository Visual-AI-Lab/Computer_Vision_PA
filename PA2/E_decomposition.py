import numpy as np

def essential_matrix_decomposition(E, inlier_p1, inlier_p2, camera_intrinsic):
    """
    Step3: Decomposes the Essential Matrix and performs triangulation to compute the poses of the two cameras.
    The function returns the pose of the first camera (P0, which is [I | 0]) and the selected pose for
    the second camera (P1) based on the cheirality condition.

    Allow functions:
        numpy
        
    Deny functions:
        cv2

    Parameters:
        E (np.ndarray): Essential Matrix (3x3 numpy array).
        inlier_p1 (np.ndarray): Inlier keypoint coordinates from the first image (N x 2 numpy array).
        inlier_p2 (np.ndarray): Inlier keypoint coordinates from the second image (N x 2 numpy array).
        camera_intrinsic (np.ndarray): Camera intrinsic matrix (3x3 numpy array).

    Returns:
        P0 (np.ndarray): Pose of the first camera ([I | 0], 3x4 numpy array).
        P1 (np.ndarray): Pose of the selected second camera (3x4 numpy array).
    """
    #TODO: Fill this functions
    
    return P0, P1
