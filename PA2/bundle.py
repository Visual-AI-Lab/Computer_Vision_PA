import numpy as np
import cv2
import matlab.engine

def bundle(camera_matrix_list, points_list, points2matches_idx_list, image_features,image_matches, camera_intrinsic, eng):
    """
    Perform bundle adjustment across multiple images using MATLAB's LM2_iter_dof.

    Parameters:
        camera_matrix_list (list[np.ndarray]): Initial camera matrices (3x4 each).
        points_list (list[np.ndarray]): 3D points for each image.
        points2matches_idx_list (list[np.ndarray]): Indices mapping 3D points to feature matches per image.
        image_features (list[list[cv2.KeyPoint]]): Keypoints detected in each image.
        image_matches (list[list[cv2.DMatch]]): Pairwise matches between consecutive images.
        camera_intrinsic (np.ndarray): Intrinsic camera matrix (3x3).
        eng (matlab.engine): MATLAB engine instance.

    Returns:
        new_points (list[np.ndarray]): Refined 3D points per image.
        new_camera_matrices (list[np.ndarray]): Refined camera matrices.
    """
    all_keypoint1 = []
    all_keypoint2 = []
    for idx in range(len(points_list)):
        matches = image_matches[idx]
        query_idx = [match.queryIdx for match in matches]
        train_idx = [match.trainIdx for match in matches]
        good_kp1 = np.float32([image_features[idx][ind].pt for ind in query_idx])
        good_kp2 = np.float32([image_features[idx+1][ind].pt for ind in train_idx])
        all_keypoint1.append(np.array(good_kp1).reshape(-1, 2))
        all_keypoint2.append(np.array(good_kp2).reshape(-1, 2))
    
    all_identical_points = []
    for idx in range(len(points_list)-1):
        inlinear = points2matches_idx_list[idx]
        matches = image_matches[idx]
        next_matches = image_matches[idx+1]
        query_idx = [match.queryIdx for match in matches]
        train_idx = [match.trainIdx for match in matches]
        
        next_query_idx = [match.queryIdx for match in next_matches] 
        double_matched_points = []

        for i in range(len(next_query_idx)):
            if next_query_idx[i] in np.array(train_idx)[inlinear].tolist():
                double_matched_points.append([np.array(train_idx)[inlinear].tolist().index(next_query_idx[i]), train_idx.index(next_query_idx[i]), i]) 
    
        double_matched_points = np.array(double_matched_points)
        all_identical_points.append(double_matched_points[:,-2:])

    all_xidx = []
    samegroup = []
    number_idx = 0
    for i in range(len(all_keypoint1)):
        x_idx = [-1 for _ in range(len(all_keypoint1[i]))]
        if samegroup != []:
            flat_samegroup = [item for sublist in samegroup for item in sublist]
        else:
            flat_samegroup = []
        for j, index in enumerate(points2matches_idx_list[i]):
            if i>0: 
                if index in all_identical_points[i-1][:,1].tolist():
                    point_index = all_identical_points[i-1][:,1].tolist().index(index)
                    matched_point = all_xidx[-1][all_identical_points[i-1][point_index][0]]
                    if matched_point != -1:
                        if not matched_point in flat_samegroup:
                            samegroup.append([matched_point, np.int64(j + number_idx + 1)])
                        else :
                            for row in samegroup:
                                if matched_point in row:
                                    row.append(np.int64(j + number_idx + 1))
            x_idx[index] = j + number_idx + 1
        all_xidx.append(x_idx)
        number_idx += len(points2matches_idx_list[i])
        
    if samegroup != []:
        flat_samegroup = [item for sublist in samegroup for item in sublist]
        flat_samegroup = np.array(flat_samegroup)
    
    
    total_3dpoint_num = 0 #전체 3d points 수
    for i in range(len(points2matches_idx_list)):
        total_3dpoint_num += len(points2matches_idx_list[i])

    change_3d = [np.int64(j+1) for j in range(total_3dpoint_num)]
    remake_3d = [np.int64(j+1) for j in range(total_3dpoint_num)]

    for i in range(total_3dpoint_num):
        flag = np.int64(i+1)
        if flag in flat_samegroup:
            for row in samegroup:
                if flag in row:
                    if min(row) != flag:
                        change_3d.remove(flag)
                        remake_3d[i] = min(row)

    #### 초기 설정
    x = np.array([])
    for i in range(len(camera_matrix_list)):
        R = camera_matrix_list[i][:, :3]
        t = camera_matrix_list[i][:, 3]
        rvec, _ = cv2.Rodrigues(R)
        rvec = rvec.flatten()
        t = t.flatten()
        camera_vec = np.concatenate((rvec, t))
        x = np.concatenate((x, camera_vec))

    number_idx = 0
    for i in range(len(points_list)):
        for j, point in enumerate(points_list[i]):
            flag = np.int64(number_idx + j + 1)
            if flag in flat_samegroup:
                for row in samegroup:
                    if flag in row:
                        if min(row) == flag:
                            point3d_vec = point.flatten()
                            x = np.concatenate((x, point3d_vec))
            else:
                point3d_vec = point.flatten()
                x = np.concatenate((x, point3d_vec))
        number_idx += len(points_list[i])

    keypoint1_vecs = []
    number_idx = 0
    for i in range(len(all_keypoint1)):
        x_idx = [-1 for _ in range(len(all_keypoint1[i]))]
        for j, index in enumerate(points2matches_idx_list[i]):
            flag = j + number_idx + 1
            if flag in flat_samegroup and i!=0:
                for row in samegroup:
                    if flag in row:
                        x_idx[index] = int(change_3d.index(min(row))) + 1
            else:
                x_idx[index] = int(change_3d.index(j + number_idx + 1)) + 1
        all_keypoint1_T = all_keypoint1[i].T
        x_idx = np.array(x_idx).reshape(1, -1)
        point2d_vec = np.vstack((all_keypoint1_T, x_idx))
        point2d_vec_filtered = point2d_vec[:, point2d_vec[2, :] != -1]
        keypoint1_vecs.append(point2d_vec_filtered)
        number_idx += len(points2matches_idx_list[i])
    
    keypoint2_vecs = []
    number_idx = 0
    for i in range(len(all_keypoint2)):
        x_idx = [-1 for _ in range(len(all_keypoint2[i]))]
        for j, index in enumerate(points2matches_idx_list[i]):
            flag = j + number_idx + 1
            if flag in flat_samegroup and i!=0:
                for row in samegroup:
                    if flag in row:
                        x_idx[index] = int(change_3d.index(min(row))) + 1
            else:
                x_idx[index] = int(change_3d.index(j + number_idx + 1)) + 1 
        all_keypoint2_T = all_keypoint2[i].T
        x_idx = np.array(x_idx).reshape(1, -1)
        point2d_vec = np.vstack((all_keypoint2_T, x_idx))
        point2d_vec_filtered = point2d_vec[:, point2d_vec[2, :] != -1]
        keypoint2_vecs.append(point2d_vec_filtered)
        number_idx += len(points2matches_idx_list[i])

    uv = []
    for i in range(len(camera_matrix_list)):
        if i==0:
            uv.append(keypoint1_vecs[i])
        elif i==len(camera_matrix_list)-1:
            uv.append(keypoint2_vecs[i-1])
        else:
            uv.append(np.concatenate((keypoint2_vecs[i-1], keypoint1_vecs[i]), axis=1))
            
    x_matlab = matlab.double(x)
    uv_matlab = uv
    K_matlab = matlab.double(camera_intrinsic)
    param = eng.struct({'uv': uv_matlab, 'K': K_matlab})
    param['nX'] = len(x) - 6 * len(camera_matrix_list)
    param['key1'] = 2 
    param['key2']  = 0 
    param['optimization'] = 1
    param['dof_remove'] = 0

    x_BA = eng.LM2_iter_dof(x_matlab, param)

    x_BA = np.array(x_BA[-1])
    x_BApoint = x_BA[6*len(camera_matrix_list):]
    x_BApoint = x_BApoint.reshape(-1,3)

    new_points = []
    number_idx = 0
    for i in range(len(points_list)):
        new_point = np.empty((0, 3))
        for j in range(len(points_list[i])):
            point3d = x_BApoint[change_3d.index(remake_3d[number_idx + j])]
            new_point = np.vstack((new_point, point3d))
        new_points.append(new_point)
        number_idx += len(points_list[i])

    new_camera_matrices = []
    for i in range(len(camera_matrix_list)):
        rvec = x_BA[(6*i):(6*i)+3]
        tvec = x_BA[(6*i)+3:(6*i)+6]
        R, _ = cv2.Rodrigues(rvec)
        new_camera_matrix = np.hstack((R, tvec.reshape(-1, 1)))
        new_camera_matrices.append(new_camera_matrix)

    return new_points, new_camera_matrices