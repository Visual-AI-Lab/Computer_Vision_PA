import cv2
import numpy as np
import os
from tqdm import tqdm

def camera_calibaration(images_files, checker_files, checkerboard):
    """
        Step7: Accepts a list of image filenames, a directory path containing the chessboard images, 
        and a tuple defining the chessboard dimensions (number of inner corners per row and column).
        The function performs camera calibration by:
            - Creating the 3D object points for the chessboard pattern.
            - Iterating through each image to convert it to grayscale.
            - Detecting the chessboard corners using cv2.findChessboardCorners.
            - Refining corner positions with cv2.cornerSubPix.
            - Collecting corresponding object points and image points.
            - Computing the camera intrinsic matrix using cv2.calibrateCamera.

        Allow functions:
            numpy
            cv2.imread()
            cv2.cvtColor()
            cv2.findChessboardCorners()
            cv2.cornerSubPix()
            cv2.calibrateCamera()
            tqdm (for progress tracking)

        Parameters:
            images_files (list[str]): List of calibration image filenames.
            checker_files (str): Directory path where the chessboard images are stored.
            checkerboard (tuple[int, int]): Dimensions of the chessboard (number of inner corners per row and column).

        Output:
            camera_matrix (numpy.ndarray): The intrinsic camera matrix computed from the calibration process (3 * 3).
    """
    #TODO: Fill this functions
    
    return camera_matrix