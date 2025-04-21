from utils.arguments import arg_as_list
from utils.keypoints import keypoints_to_dict, dict_to_keypoints, dmatches_to_dict, dict_to_dmatches
from utils.pointcloud import write_ply
from utils.calculate_color import calculate_colors

from camera_calibration import camera_calibaration
from feature_matching import matching_two_image
from E_estimation import essential_matrix_estimation
from E_decomposition import essential_matrix_decomposition
from triangulation import triangulate_points

from three_point_algorithm import three_point_algorithm, calculate_inlier_points
from bundle import bundle

import argparse
import os
import natsort
import pickle
import matlab.engine
import cv2
import numpy as np
from tqdm import tqdm

def args_parser():
    parser = argparse.ArgumentParser(description="GIST computer vision programming assignment 2")
    parser.add_argument('-s', '--step', type=str, default='all')
    parser.add_argument('-d', '--dataset_path', type=str, default='./input')
    parser.add_argument('-o', '--output_path', type=str, default='./output_multi')
    
    parser.add_argument('--object', type=str, default='moai')
    parser.add_argument('--image_order', type=str, default='[4,3,2,1,0,19,18,17,16]')
    
    parser.add_argument('--matching_threshold_knn', type=int, default=0.60)
    parser.add_argument('--ransac_iter', type=int, default=1000)
    parser.add_argument('--em_threshold', type=int, default=1e-4)
    
    parser.add_argument('--three_point_ransac_iter', type=int, default=2000)
    parser.add_argument('--three_point_threshold', type=int, default=1e-6)
    parser.add_argument('--three_point_inlier_threshold', type=int, default=1e-4)
    
    parser.add_argument('--apply_bundle', type=bool, default=False)

    parser.add_argument('--visualize_camera_pose', type=bool, default=False)
    
    args = parser.parse_args()
    
    if args.step == 'all':
        args.step = [1, 2, 3, 4, 5, 6, 7]
    else:
        args.step = arg_as_list(args.step)
    args.image_order = arg_as_list(args.image_order)
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
        eng.addpath(r'./Step5/', nargout=0)
        
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
    ################ Initial Steps: Initial from two view images ################
    #############################################################################
    #############################################################################
    ################## Step1: Feature Extraction and Matching ###################
    #############################################################################
    matching_result_folder = os.path.join(output_path, "matching_results")
    os.makedirs(matching_result_folder, exist_ok=True)
    matching_result_path = os.path.join(matching_result_folder, 'matching_results_init_images.pkl')
    if 1 in args.step and not os.path.isfile(matching_result_path):
        image_names = []
        image_features = []
        image_matches = []
        pbar = tqdm(range(len(args.image_order) - 1))
        for img_idx in pbar:
            image1 = str(args.image_order[img_idx]) + "." + args.image_mode
            image1_path = os.path.join(args.dataset_path, args.object, image1)
            image2 = str(args.image_order[img_idx+1]) + "." + args.image_mode
            image2_path = os.path.join(args.dataset_path, args.object, image2)
            pbar.set_description(f"Image matching({image1}, {image2})")
            
            # TODO: Feature matching between two images
            img1, img2, kp1, kp2, des1, des2, matches = matching_two_image(image1_path, image2_path, threshold_knn=args.matching_threshold_knn)
            
            if img_idx == 0:
                image_names.append(image1)
                image_features.append(kp1)
            image_names.append(image2)
            image_features.append(kp2)
            image_matches.append(matches)
            
            # Visualize matching results
            matches_for_draw = [[m] for m in matches]
            matching_result_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches_for_draw, None, flags=2)
            matching_result_img = cv2.resize(matching_result_img, (3200, 1200), interpolation=cv2.INTER_AREA)
            matching_result_img_path = os.path.join(matching_result_folder, f"matching_results_{str(args.image_order[img_idx])}_{str(args.image_order[img_idx+1])}.jpg")
            cv2.imwrite(matching_result_img_path, matching_result_img)
        kps = [keypoints_to_dict(k) for k in image_features]
        matches = [dmatches_to_dict(match) for match in image_matches]
        
        matching_data = {
            "images": image_names, 
            "kps": kps,
            "matches": matches
        }
        with open(matching_result_path, 'wb') as f:
            pickle.dump(matching_data, f)
    else:
        with open(matching_result_path, 'rb') as f: 
            matching_data = pickle.load(f)
        # Ensure that the loaded matching data corresponds to the expected image numbers
        assert (len(matching_data["images"])) == (len(args.image_order)), "Different matching data in image num"
            
        image_names = matching_data["images"]
        keypoints = matching_data["kps"]
        image_features = [dict_to_keypoints(k) for k in keypoints]
        image_matches = [dict_to_dmatches(matches) for matches in matching_data["matches"]]
        
    print(f"\nImage lists: {image_names}")
    print(f"Number of keypoints")
    for img_idx in range(len(args.image_order)):
        print(f"{image_names[img_idx]}: {len(image_features[img_idx])}")
    print(f"\nNumber of matches")
    for img_idx in range(len(args.image_order) - 1):
         print(f"{image_names[img_idx]}-{image_names[img_idx+1]}: {len(image_matches[img_idx])}")
    
    #############################################################################
    ################### Step2: Essential Matrix Estimation ######################
    #############################################################################
    initial_result_folder = os.path.join(output_path, "initial_results")
    os.makedirs(initial_result_folder, exist_ok=True)
    E_result_path = os.path.join(initial_result_folder, 'E_estimation.pkl')
    if 2 in args.step and not os.path.isfile(E_result_path):
        # TODO: Estimate the essential matrix using matched keypoints
        E, inlier_p1, inlier_p2, inlier_matches_idx = essential_matrix_estimation(image_features[0], image_features[1], image_matches[0], camera_intrinsic, eng, max_iter=args.ransac_iter, threshold=args.em_threshold)
        
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
        inlier_matches_idx = data['inlier_matches_idx']
    print("Essential Matrix:")
    print(E)
    
    print(f"Number of inlier points: {inlier_p1.shape[0]}")
    
    #############################################################################
    ################## Step3: Essential Matrix Decomposition ####################
    #############################################################################
    P1_result_path = os.path.join(initial_result_folder, 'camera_pose.pkl')
    if 3 in args.step and not os.path.isfile(P1_result_path):
        # Decompose the essential matrix to obtain the camera poses
        P0, P1 = essential_matrix_decomposition(E, inlier_p1, inlier_p2, camera_intrinsic)

        # Save the recovered camera poses to a file
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
    triangulation_result_path = os.path.join(initial_result_folder, 'triangulation_results.pkl')
    pcl_result_path = os.path.join(initial_result_folder, 'two_view_results.ply')
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
    
    img1_path = os.path.join(args.dataset_path, args.object, image_names[0])
    img2_path = os.path.join(args.dataset_path, args.object, image_names[1])
    colors = calculate_colors(img1_path, img2_path, inlier_p1, inlier_p2, inlier_idx)
    write_ply(pcl_result_path, points_3d, colors, [P0, P1], show_camera=args.visualize_camera_pose)
    #############################################################################
    ############################# End Initial Steps #############################
    #############################################################################
    all_points = []
    all_colors = []
    
    
    points_list = []
    colors_list = []
    points2matches_idx_list = [] 
    all_point3d_idx = []
    camera_matrix_list = []
    
    
    points_list.append(points_3d)
    colors_list.append(colors)
    points2matches_idx_list.append(inlier_matches_idx[inlier_idx])
    all_point3d_idx.append(inlier_idx)
    
    camera_matrix_list.append(P0)
    camera_matrix_list.append(P1)
    
    all_points = np.array(points_3d)
    all_colors = np.array(colors)
    
    #############################################################################
    ############################### Growing Step ################################
    #############################################################################
    growing_result_folder = os.path.join(output_path, "growing_step_results")
    os.makedirs(growing_result_folder, exist_ok=True)
    for img_idx in range(len(image_names)-2):
        img_name0, img_name1, img_name2 = image_names[img_idx].split(".")[0], image_names[img_idx+1].split(".")[0], image_names[img_idx+2].split(".")[0]
        step_name = f"{img_name0}_{img_name1}_{img_name2}"
        step_result_folder = os.path.join(growing_result_folder, step_name)
        os.makedirs(step_result_folder, exist_ok=True)
        
        #############################################################################
        ################### Step 5: Three Point Algorithm for PnP. ##################
        #############################################################################
        if args.apply_bundle:
            before_bundle_results_path = os.path.join(step_result_folder, 'before_bundle_results.pkl')
            before_bundle_pcl_path = os.path.join(step_result_folder, 'before_bundle_results.ply')
        else:
            before_bundle_results_path = os.path.join(step_result_folder, 'growing_step.pkl')
            before_bundle_pcl_path = os.path.join(step_result_folder, 'growing_step.ply')
            
        print(f"{step_name} Processing...")
        if 5 in args.step and not os.path.isfile(before_bundle_results_path):
            # TODO: Fill this function
            add_camera_matrix = three_point_algorithm(image_matches[img_idx],
                                                      image_matches[img_idx+1], 
                                                      points2matches_idx_list[img_idx], 
                                                      points_list[img_idx], 
                                                      image_features[img_idx+2], 
                                                      camera_intrinsic, 
                                                      eng, 
                                                      threepoint_threshold=args.three_point_threshold,
                                                      threepoint_max_iter=args.three_point_ransac_iter)
            # TODO: Fill this function
            inlier_p1, inlier_p2, inlier_matches_idx = calculate_inlier_points(camera_matrix_list[img_idx+1],
                                                                               add_camera_matrix, 
                                                                               image_features[img_idx+1], 
                                                                               image_features[img_idx+2], 
                                                                               image_matches[img_idx+1], 
                                                                               camera_intrinsic,
                                                                               three_point_inlier_threshold=args.three_point_inlier_threshold)
            points_3d, inlier_idx = triangulate_points(camera_matrix_list[img_idx+1], 
                                                       add_camera_matrix, 
                                                       inlier_p1, 
                                                       inlier_p2, 
                                                       camera_intrinsic)
            
            print(f"Camera {img_name2} matrix:")
            print(add_camera_matrix)
            print("Number of triangulated 3D points:", points_3d.shape[0])
            
            img1_path = os.path.join(args.dataset_path, args.object, image_names[img_idx+1])
            img2_path = os.path.join(args.dataset_path, args.object, image_names[img_idx+2])
            colors = calculate_colors(img1_path, img2_path, inlier_p1, inlier_p2, inlier_idx)
            
            camera_matrix_list.append(add_camera_matrix)
            points_list.append(points_3d)
            colors_list.append(colors)
            points2matches_idx_list.append(inlier_matches_idx[inlier_idx])
            all_point3d_idx.append(inlier_idx)
            all_points = np.concatenate(points_list, axis=0)
            all_colors = np.concatenate(colors_list, axis=0)
            
            write_ply(before_bundle_pcl_path, all_points, all_colors, camera_matrix_list, show_camera=args.visualize_camera_pose)
            with open(before_bundle_results_path, 'wb') as f:
                pickle.dump({'points': all_points, 
                             'colors': all_colors, 
                             'camera_matrix': camera_matrix_list, 
                             'points_list': points_list, 
                             'colors_list': colors_list, 
                             'points2matches_idx_list': points2matches_idx_list, 
                             'all_point3d_idx':all_point3d_idx}, f)
        else:
            with open(before_bundle_results_path, 'rb') as f:
                data = pickle.load(f)
            all_points = data['points']
            all_colors = data['colors']
            camera_matrix_list = data['camera_matrix']
            points_list = data['points_list']
            colors_list = data['colors_list']
            points2matches_idx_list = data['points2matches_idx_list']
            all_point3d_idx = data['all_point3d_idx']

        #############################################################################
        ######################### Step 6: Bundle Adjustments ########################
        #############################################################################
        if args.apply_bundle:
            bundle_result_path = os.path.join(step_result_folder, 'after_bundle_results.pkl')
            bundle_pcl_path = os.path.join(step_result_folder, 'after_bundle_results.ply')

            if 7 in args.step and not os.path.isfile(bundle_result_path):
                print("Performing Bundle Adjustment...")
                points_refined, poses_refined = bundle(
                    camera_matrix_list,
                    points_list,
                    points2matches_idx_list,
                    image_features,
                    image_matches,
                    camera_intrinsic,
                    eng
                )
                
                points_list = points_refined
                camera_matrix_list = poses_refined
                all_points = np.concatenate(points_list, axis=0)
                
                with open(bundle_result_path, 'wb') as f:
                    pickle.dump({'points': points_refined, 'poses': poses_refined}, f)
            else:
                with open(bundle_result_path, 'rb') as f:
                    data_bundle = pickle.load(f)
                points_refined, poses_refined = data_bundle['points'], data_bundle['poses']

            all_colors = np.vstack(colors_list)
            write_ply(bundle_pcl_path, all_points, all_colors, poses_refined, show_camera=args.visualize_camera_pose)
        print(f"Finished {step_name} processing.")
        print("Number of points in this step:", all_points.shape[0])
        print("Number of colors in this step:", all_colors.shape[0])
    print("Growing step finished.")
    #############################################################################
    ############################# End Growing Steps #############################
    #############################################################################
    result_folder = os.path.join(output_path, "final_results")
    os.makedirs(result_folder, exist_ok=True)
    final_pcl_path = os.path.join(result_folder, 'final_results.ply')
    write_ply(final_pcl_path, all_points, all_colors, camera_matrix_list, show_camera=args.visualize_camera_pose)
    print("Final point cloud saved to:", final_pcl_path)
    print("MATLAB engine closed.")
    print("All steps completed successfully.")
    
if __name__ == '__main__':
    main()