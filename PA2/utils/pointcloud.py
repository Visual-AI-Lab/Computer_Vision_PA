import numpy as np

def write_ply(filename, points, colors, camera_poses=None, show_camera=True):
    vertices = []
    faces = []
    
    for i in range(points.shape[0]):
        x, y, z = points[i]
        r, g, b = colors[i].astype(int)
        vertices.append((x, y, z, r, g, b))
    
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_range = bbox_max - bbox_min
    scale_val = np.mean(bbox_range) / 10.0  
    pyramid_scale = scale_val
    pyramid_depth = scale_val

    if show_camera:
        for pose in camera_poses:
            R = pose[:, :3]
            t = pose[:, 3]
            apex = t
            base_center = t + pyramid_depth * R[:, 2]
            right = pyramid_scale * R[:, 0]
            up    = pyramid_scale * R[:, 1]
            v1 = base_center + right + up
            v2 = base_center - right + up
            v3 = base_center - right - up
            v4 = base_center + right - up
            
            cam_vertices = [apex, v1, v2, v3, v4]
            cam_indices = []
            for v in cam_vertices:
                vertices.append((v[0], v[1], v[2], 255, 0, 0))
                cam_indices.append(len(vertices) - 1)
            
            faces.append((cam_indices[0], cam_indices[1], cam_indices[2]))
            faces.append((cam_indices[0], cam_indices[2], cam_indices[3]))
            faces.append((cam_indices[0], cam_indices[3], cam_indices[4]))
            faces.append((cam_indices[0], cam_indices[4], cam_indices[1]))
            faces.append((cam_indices[1], cam_indices[2], cam_indices[3]))
            faces.append((cam_indices[1], cam_indices[3], cam_indices[4]))
            
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex %d\n" % len(vertices))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("element face %d\n" % len(faces))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v in vertices:
            f.write("%f %f %f %d %d %d\n" % v)

        for face in faces:
            f.write("3 %d %d %d\n" % face)