import numpy as np

def triangulate_points(EM0, EM1, inlier_p1, inlier_p2, camera_intrinsic):
    """
    Step4: Computes 3D points via linear triangulation using the given camera poses (EM0, EM1)
    and the corresponding inlier keypoint coordinates from two images.
    
    Allow functions:
        numpy
        
    Deny functions:
        cv2

    Parameters:
        EM0             : Pose of the first camera ([I|0], 3x4 numpy array).
        EM1             : Pose of the second camera (3x4 numpy array).
        inlier_p1       : Inlier keypoints from the first image (N x 2 numpy array, [x, y]).
        inlier_p2       : Inlier keypoints from the second image (N x 2 numpy array, [x, y]).
        camera_intrinsic: Camera intrinsic matrix (3x3 numpy array).
        
    Returns:
        points_3d (np.ndarray): (N x 3) numpy array where each row is the triangulated 3D coordinate (X, Y, Z).
        inlier_idx (np.ndarray): (N,) numpy array containing the indices of the inlier points used.
    """
    #TODO: Fill this functions
    
    return points_3d, inlier_idx
