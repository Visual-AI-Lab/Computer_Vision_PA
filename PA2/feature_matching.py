import cv2
import glob
import numpy as np
import os
from tqdm import tqdm

def matching_two_image(image1_path, image2_path, threshold_knn=0.75):
    """
    TODO:
    Step1: Accepts two image file paths and performs SIFT-based matching between them.
    It detects keypoints and computes descriptors using the SIFT algorithm,
    then matches descriptors using BFMatcher with k-NN matching.
    Finally, it applies Lowe's ratio test to filter out unreliable matches.
    
    Allow functions:
        cv2.cvtColor()
        cv2.SIFT_create()
        cv2.SIFT_create().*
        cv2.BFMatcher()
        cv2.BFMatcher().*
        cv2.drawMatchesKnn()
        
    Parameters:
        image1_path (str): File path for the first image.
        image2_path (str): File path for the second image.
        threshold_knn (float): Lowe's ratio test threshold (default is 0.75).
        
    Output:
        img1, img2 (numpy.ndarray): The original images.
        kp1, kp2 (list[cv2.KeyPoint]): Lists of keypoints detected in each image.
        des1, des2 (numpy.ndarray): SIFT descriptors for each image.
        matches (list[cv2.DMatch]): The matching results after applying Lowe's ratio test.
    """
    #TODO: Fill this functions
    
    return img1, img2, kp1, kp2, des1, des2, matches
