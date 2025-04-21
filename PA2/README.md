# PA2: Structure from Motion
This repository contains the code for the second programming assignment on Structure from Motion. The goal of the assignment is to recover 3D structure from two images by performing feature matching, essential matrix estimation and decomposition, and triangulation.

- **Due Date:** 10th May 2025  
- **TA Sessions:** 1st May 2025 and 8th May 2025  
- **TA:** Dongmin Shin (newdm2000@gm.gist.ac.kr)

**NO PLAGIARISM, NO DELAY, DON'T USE AI SUPPORTER (If you do not comply, you will get F!)**

## Overview

### Before you start, make sure you understand the code flow by reading main_two_view.py

The provided code (`main_two_view.py`) is structured in multiple stages:  
1. **Feature Extraction and Matching:**  
   Two consecutive images are processed to extract keypoints and match features between them.  
   *Fill the #todo blank in:* `matching_two_image`

2. **Essential Matrix Estimation:**  
   An essential matrix is estimated using RANSAC based on the inlier matches from the feature extraction.  
   *Fill the #todo blank in:* `essential_matrix_estimation`

3. **Essential Matrix Decomposition:**  
   The essential matrix is decomposed to recover the relative camera pose (rotation and translation).  
   *Fill the #todo blank in:* `essential_matrix_decomposition`

4. **Triangulation:**  
   Using the recovered camera poses and inlier matches, the 3D points are triangulated and the point cloud is saved in PLY format.  
   *Fill the #todo blank in:* `triangulate_points`

### Additional credit
5. **Three Point Algorithm:**  
   In growing step, calculate additional camera pose using three point algorithm with RANSAC. Additionally, calculate inlier points.  
   *Fill the #todo blank in:* `three_point_algorithm`, and `calculate_inlier_points`

6. **Camera Calibration:**  
   Using the checker board, the intrinsic camera matrix are calculated by the function. Additionally, make your own dataset and test the SfM.  
   *Fill the #todo blank in:* `camera_calibaration`

## Directory Structure
```
.  
├── input/                    # Input dataset directory 
│ ├── checker_board/          # 19 checker board images for camera calibration (optional)
│ ├── Checkerboard8x6.pdf     # Checker boerd image for custom dataset (optional)
│ ├── custom/                 # Make you dataset (optional)
│ ├── choonsik/               # 19 Object images
│ ├── toothless/              # 21 Object images
│ ├── nike/                   # 17 Object images
│ └── moai/                   # 19 Object images
│
├── output/                   # Output directory to save results
│ └── camera_intrinsic.pkl    # Camera intrinsic parameter
├── output_multi/             # Output directory to save results for multi view (optional)
│ └── camera_intrinsic.pkl    # Camera intrinsic parameter (optional)
│
├── Step2/                    # MATLAB scripts for essential matrix estimation
├── Step5/                    # MATLAB scripts for essential matrix estimation (optional)
├── utils/                    # Utility modules for keypoint conversion, point cloud writing, etc.
│
├── E_decomposition.py        # Module to implement essential_matrix_decomposition 
├── E_estimation.py           # Module to implement essential_matrix_estimation 
├── feature_matching.py       # Module to implement matching_two_image 
├── triangulation.py          # Module to implement triangulate_points
│
├── three_point_algorithm.py  # Module to implement three_point_algorithm and calculate_inlier_points  (optional)
├── bundle.py                 # Already implemented. Only use the bundle for multi view.  (optional)
│
├── main_two_view.py          # Main Python script to run the assignment 
├── main_mulit_view.py        # Main Python script to run the assignment (optional)
└── README.md                 # This README file
```

## Requirements

### Python Requirements

- **Python Version:** 3.6 or above
- **Libraries:**  
  - OpenCV (`cv2`)
  - NumPy
  - argparse
  - natsort
  - pickle
  - tqdm
  - MATLAB Engine API for Python (see below)

You can install the required Python libraries (except the MATLAB Engine) using pip:

```
pip install opencv-python numpy natsort tqdm
```

### MATLAB Engine API for Python
The MATLAB Engine API for Python allows your Python code to invoke MATLAB functions. This is essential for running certain parts of the assignment (e.g., essential matrix estimation and triangulation).

**Installation Instructions:**

1. **Verify MATLAB Installation:**  
    Confirm that MATLAB is installed on your system.

2. **Locate the MATLAB Engine Directory:**  
    The MATLAB Engine API resides in the MATLAB installation folder:

   - Windows:  
    `C:\Program Files\MATLAB\{R2022b}\extern\engines\python`

   - macOS/Linux:  
    `{MATLAB_ROOT}/extern/engines/python`  
    Replace `R2022b` (or `MATLAB_ROOT`) with your MATLAB version.

3. **Install the MATLAB Engine API:**  
    Open a terminal (or command prompt) and navigate to the MATLAB engine directory. Then run:

    ```
    cd {path_to_matlab_installation}/extern/engines/python
    python setup.py install
    ```

    On macOS/Linux, you might need to use sudo:

    ```
    sudo python setup.py install
    ```
4. **Test the Installation:**  
    Launch a Python shell and try importing the MATLAB engine module:

    ```
    import matlab.engine
    ```
    If there are no errors, the installation was successful.

## Usage

### Arguments
Run the main script using the command line. The script accepts several arguments to control various processing steps:

- `-s` or `--step`: Steps to execute (default: all steps). For example, `-s 1,2,3` runs only the camera calibration, feature matching, and essential matrix estimation steps. If `-s all`, runs all step of two view SfM.

- `-d` or `--dataset_path`: Path to the dataset folder (default: ./input).

- `-o` or `--output_path`: Folder where outputs will be saved (default: ./output).

- `--object`: Specifies the object name to process. This should correspond to a subfolder in the dataset (e.g., "nike", "moai", "toothless"). (default: moai)

- `--initial_image_num`: The starting image number for processing.

- `--second_image_num`: The starting image number for processing.

- `--matching_threshold_knn`: The threshold for k-NN feature matching. Adjust as necessary to suit your dataset. (default: moai)
  
- `--ransac_iter`: The number of iterations for the RANSAC algorithm during essential matrix estimation. (default: moai)

- `--em_threshold`: Threshold value for inlier selection during essential matrix estimation. (default: moai)
  
- `--visualize_camera_pose`: When set to True, the camera poses recovered during essential matrix decomposition will be visualized. (default: moai)

### Example Command

Run the main script from the command line with the desired arguments. For instance:

```
python main.py -s all -d ./input -o ./output --object moai
```

## Implementation Details
1. **Step 0: Settings and MATLAB Engine Initialization**
   - Initializes the MATLAB engine if required for MATLAB-dependent steps

2. **Step 1: Feature Extraction and Matching**
   - Extracts features using SIFT algorithm from a pair of images 
   - Finds matching points between the extracted features using the KNN algorithm.

3. **Step 2: Essential Matrix Estimation**
   - Nomalise the image pixels using intrinsic camera matrix.
   - Uses 5 point algorithm along with a RANSAC strategy to compute the essential matrix.
   - Calculate error and estimate essential matrix that has lowest error.

4. **Step 3: Essential Matrix Decomposition**
   - Decomposes the essential matrix to obtain the relative camera pose and saves the computed poses.

5. **Step 4: Triangulation**
   - Triangulates 3D points based on the established camera poses and inlier matches.
   - Generates a PLY file containing a colored point cloud.

## Implementation Details for Additional Credit
6. **Step 5: Three Point Algorithm for PnP.**  
   - Estimate the additional camera pose using three point algorithm with RANSAC
   - And, calculate the addtional inlier points to 3d points

7. **Step 6: Bundle Adjustments.(Implemented)**  
   - Already implemented the bundle. Only use the function. (use the option `--apply_bundle True`)

8. **Step 7: Camera Calibration**  
   - Detect corner of checker board and calculate intrinsic matrix using opencv.
   - Print the checker board and take a picture.
   - Make your own dataset. Note that, fix the manual focus.
   - Run SfM. (make a folder "custom" in input and use the arguments `--object custom`)

## Output  
The script generates the following outputs for each object:
- Matching result: Saved `matching_results_init_images.jpg` and `matching_results_init_images.pkl`
- Essential matrix results: Saved `E_estimation.pkl`
- Camera pose: `camera_pose.pkl`
- Triangulation results: Saved `triangulation_results.pkl`
- 3D point cloud results: Saved `two_view_results.ply`

## Troubleshooting
- Edit the line 71 in `main_two_view.py` for folder structure of camera_calibration.
