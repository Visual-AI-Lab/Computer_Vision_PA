from utils.arguments import arg_as_list
from utils.keypoints import keypoints_to_dict, dict_to_keypoints, dmatches_to_dict, dict_to_dmatches
from utils.pointcloud import write_ply
from utils.calculate_color import calculate_colors

from camera_calibration import camera_calibaration
from feature_matching import matching_two_image
from E_estimation import essential_matrix_estimation
from E_decomposition import essential_matrix_decomposition
from triangulation import triangulate_points

import argparse
import os
import natsort
import pickle
import matlab.engine
import cv2
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def args_parser():
    parser = argparse.ArgumentParser(description="GIST computer vision programming assignment 2")
    parser.add_argument('-s', '--step', type=str, default='all')
    parser.add_argument('-d', '--dataset_path', type=str, default='./input')
    parser.add_argument('-o', '--output_path', type=str, default='./output')
    
    parser.add_argument('--object', type=str, default='moai')
    parser.add_argument('--initial_image_num', type=int, default=0)
    parser.add_argument('--second_image_num', type=int, default=1)
    
    parser.add_argument('--matching_threshold_knn', type=int, default=0.75)
    parser.add_argument('--ransac_iter', type=int, default=1000)
    parser.add_argument('--em_threshold', type=int, default=1e-4)

    parser.add_argument('--visualize_camera_pose', type=bool, default=False)
    
    args = parser.parse_args()
    
    if args.step == 'all':
        args.step = [1, 2, 3, 4, 5, 6, 7]
    else:
        args.step = arg_as_list(args.step)

    args.image_mode = 'jpg'
    
    return args

def main():
    #############################################################################
    ############################## Step0: Settings ##############################
    #############################################################################
    # Parse command-line arguments
    args = args_parser()
    output_path = os.path.join(args.output_path, args.object)
    os.makedirs(output_path, exist_ok=True)
    CHECKER_BOARD = (6, 8)
    
    # Start MATLAB engine if required in step 3 or 6
    if 2 in args.step or 5 in args.step:
        eng = matlab.engine.start_matlab()
        eng.addpath(r'./Step2/', nargout=0)
        
    #############################################################################
    ######################### Step7: Camera Calibration #########################
    #############################################################################
    calib_file_path = os.path.join(args.output_path, 'camera_intrinsic.pkl')
    if 7 in args.step and not os.path.isfile(calib_file_path):
        # Define checkerboard image directory path
        checker_file_path = os.path.join(args.dataset_path, "checker_board")
        checker_files = os.listdir(checker_file_path)
        checker_files = natsort.natsorted(checker_files)
        
        # TODO: Camera calibration using checkerboard images
        camera_intrinsic = camera_calibaration(checker_files, checker_file_path, CHECKER_BOARD)
        with open(calib_file_path, 'wb') as f:
            pickle.dump(camera_intrinsic, f)
    else:
        with open(calib_file_path, 'rb') as f: 
            camera_intrinsic = pickle.load(f)
            
    print("Camera intrinsic matrix: ")
    print(camera_intrinsic)
    
    #############################################################################
    ################## Step1: Feature Extraction and Matching ###################
    #############################################################################
    matching_result_img_path = os.path.join(output_path, 'matching_results_init_images.jpg')
    matching_result_path = os.path.join(output_path, 'matching_results_init_images.pkl')
    if 1 in args.step and not os.path.isfile(matching_result_path):
        # Define the paths for the two consecutive images
        init_image_path = os.path.join(args.dataset_path, args.object, str(args.initial_image_num) + "." + args.image_mode)
        matching_image_path = os.path.join(args.dataset_path, args.object, str((args.second_image_num) % 10) + "." + args.image_mode)
        
        # TODO: Feature matching between two images
        img1, img2, kp1, kp2, des1, des2, matches = matching_two_image(init_image_path, matching_image_path, threshold_knn=args.matching_threshold_knn)
        
        matches_for_draw = [[m] for m in matches]
        matching_result_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches_for_draw, None, flags=2)
        matching_result_img = cv2.resize(matching_result_img, (3200, 1200), interpolation=cv2.INTER_AREA)
        cv2.imwrite(matching_result_img_path, matching_result_img)
        matching_data = {
            "image1": str(args.initial_image_num) + "." + args.image_mode, 
            "image2": str((args.second_image_num) % 10) + "." + args.image_mode,
            "kp1": keypoints_to_dict(kp1), "kp2": keypoints_to_dict(kp2), "des1": des1, "des2": des2, "matches": dmatches_to_dict(matches)
        }
        with open(matching_result_path, 'wb') as f:
            pickle.dump(matching_data, f)
    else:
        with open(matching_result_path, 'rb') as f: 
            matching_data = pickle.load(f)
            
        assert (matching_data["image1"]) == (str(args.initial_image_num) + "." + args.image_mode) and \
            (matching_data["image2"]) == (str(args.second_image_num) + "." + args.image_mode), "Different matching data in image num"
        kp1 = dict_to_keypoints(matching_data["kp1"])
        kp2 = dict_to_keypoints(matching_data["kp2"])
        matches = dict_to_dmatches(matching_data["matches"])
    
    print(f"\nNumber of keypoints and matches (image{args.initial_image_num}, image{args.second_image_num}): ")
    print(f"kp1 : {len(kp1)}, kp2 : {len(kp2)}, matches : {len(matches)}")
    
    #############################################################################
    ################### Step2: Essential Matrix Estimation ######################
    #############################################################################
    E_result_path = os.path.join(output_path, 'E_estimation.pkl')
    if 2 in args.step and not os.path.isfile(E_result_path):
        # TODO: Estimate the essential matrix using matched keypoints
        E, inlier_p1, inlier_p2, inlier_matches_idx = essential_matrix_estimation(kp1, kp2, matches, camera_intrinsic, eng, max_iter=args.ransac_iter, threshold=args.em_threshold)
        
        data = {
            'E': E,
            'inlier_p1': inlier_p1,
            'inlier_p2': inlier_p2,
            'inlier_matches_idx': inlier_matches_idx
        }
        with open(E_result_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(E_result_path, 'rb') as f:
            data = pickle.load(f)
        E = data['E']
        inlier_p1 = data['inlier_p1']
        inlier_p2 = data['inlier_p2']
        
    print(E)
    print(f"Number of inlier points: {inlier_p1.shape[0]}")
    
    #############################################################################
    ################## Step3: Essential Matrix Decomposition ####################
    #############################################################################
    P1_result_path = os.path.join(output_path, 'camera_pose.pkl')
    if 3 in args.step and not os.path.isfile(P1_result_path):
        # TODO: Estimate the essential decompostion for camera pose
        P0, P1 = essential_matrix_decomposition(E, inlier_p1, inlier_p2, camera_intrinsic)

        data_pose = {'P0': P0, 'P1': P1}
        with open(P1_result_path, 'wb') as f:
            pickle.dump(data_pose, f)
    else:
        with open(P1_result_path, 'rb') as f:
            data_pose = pickle.load(f)
        P0 = data_pose['P0']
        P1 = data_pose['P1']

    print("Camera 1 Pose (P0):")
    print(P0)
    print("Camera 2 Pose (P1):")
    print(P1)
    
    #############################################################################
    ########################### Step4: Triangulation ############################
    #############################################################################
    triangulation_result_path = os.path.join(output_path, 'triangulation_results.pkl')
    pcl_result_path = os.path.join(output_path, 'two_view_results.ply')
    if 4 in args.step and not os.path.isfile(triangulation_result_path):
        points_3d, inlier_idx = triangulate_points(P0, P1, inlier_p1, inlier_p2, camera_intrinsic)
        
        data_tri = {'points_3d': points_3d, "inlier_idx": inlier_idx}
        with open(triangulation_result_path, 'wb') as f:
            pickle.dump(data_tri, f)
    else:
        with open(triangulation_result_path, 'rb') as f:
            data_tri = pickle.load(f)
        points_3d, inlier_idx = data_tri['points_3d'], data_tri['inlier_idx']

    print("Number of triangulated 3D points:", points_3d.shape[0])
    
    init_image_path = os.path.join(args.dataset_path, args.object, str(args.initial_image_num) + "." + args.image_mode)
    matching_image_path = os.path.join(args.dataset_path, args.object, str((args.second_image_num) % 10) + "." + args.image_mode)
    colors = calculate_colors(init_image_path, matching_image_path, inlier_p1, inlier_p2, inlier_idx)
    write_ply(pcl_result_path, points_3d, colors, [P0, P1], show_camera=args.visualize_camera_pose)

if __name__ == '__main__':
    main()
