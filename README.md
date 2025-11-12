# Drone_swarm_algos_robofest


THE FOLLOWING DIRECTORIES IN THIS REPO ARE AS FOLLOWS:

PIPELINE_A: MAIN DRONE STITCHING + DETECTION + PATH PLANNING 
1) image capture + stitching
2) running yolo + sahi on stitched image for detection
3) get path planning [as a list of coords in image pixels] [so some mapping between drone x,y,z and the actual coords in form of pixels in stitched image]

PIPELINE_B: ADVANCED SPEECH TO TEXT WITH WAKE WORD + USER COMMAND IDENTIFICATION + CONFIRMATION + EXECUTION SERVICE/PARAM CALLS + TEXT TO SPEECH 
1) porcupine wake word detection
2) silero-vad Voice activity detection
3) vosk speech to text
4) intent analysis
5) text to speech for declaration of user command, pending confirmation
6) silero-vad + vosk: to get user response[yes/no]
7) user response is passed to intent analysis to finally get confirmation/rejection of command
8) command execution + command message via text to speech

[NOTE: THE FOLLOWING YOLO_PREP CAN BE SKIPPED ITS FOR INFO ONLY, I HAVE ALREADY PROVIDED A TRAINED CUSTOM YOLO MODEL IN REPO, SO USE IT DIRECTLY IN THE ABOVE PIPELINE_A] :-

YOLO_PREP: THE CUSTOM SYNTHETIC IMAGE DATASET GENERATION SCRIPT AND THE YOLO TRAINING SCRIPT 
1) synthetic image generation script
2) training yolo script
3) testing and running inference script


-----

## Documentation: Pipeline A (Drone Stitching, Detection & Path Planning)

### Overview

This pipeline provides a complete, end-to-end solution for processing aerial drone footage to identify and navigate around obstacles. It operates in three main stages:

1.  **Stitching:** It ingests multiple drone video files, extracts relevant frames, and stitches them into a single, high-resolution orthomosaic (top-down map).
2.  **Detection:** It uses a YOLO model, enhanced by SAHI (Slicing Aided Hyper Inference), to run high-accuracy object detection over the entire stitched map, even on very small objects.
3.  **Path Planning:** It uses the A\* (A-star) algorithm to find the shortest, safest path from a defined start point to an end point, treating all detected objects as obstacles with a user-defined safety margin.

### Prerequisites & Setup

Before running the pipeline, ensure the following are in place:

1.  **Python & Libraries:** A Python environment (e.g., 3.8+) with the required libraries installed.
    ```bash
    pip install numpy
    pip install opencv-python
    pip install opencv-contrib-python  # Required for the Stitcher module
    pip install sahi
    pip install ultralytics
    pip install torch
    ```
2.  **YOLO Model:** A trained YOLO model file (e.g., `best.pt`) must be available. Its path is set in the `MODEL_PATH` parameter.
3.  **Video Data:** The drone video files (`.mp4` or similar) must be available. Their paths are set in the `DRONE_..._VIDEO_PATH` parameters.

### 🔧 Key Parameters (Configuration)

All key parameters are located at the top of the script for easy customization:

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `DRONE_..._VIDEO_PATH` | `str` | File paths to the input drone videos. |
| `STITCHING_FRAME_RATE`| `int` | The interval for frame extraction (e.g., `10` means every 10th frame). |
| `MODEL_PATH` | `str` | Path to the trained YOLO model (`best.pt`). |
| `MODEL_CONFIDENCE_THRESHOLD` | `float` | The confidence score (0.0 to 1.0) required to register a detection. |
| `GRID_RESOLUTION` | `int` | The size (in pixels) of one grid cell for the A\* algorithm (e.g., `10` = 10x10 pixel cells). |
| `SAFETY_MARGIN` | `int` | The buffer (in pixels) to add around each detected obstacle's bounding box. |
| `START_POINT` | `tuple` | The `(x, y)` pixel coordinate for the path's start. |
| `END_POINT` | `tuple` | The `(x, y)` pixel coordinate for the path's end. |

### How to Run

1.  Place your drone video files and your `best.pt` model file in their respective locations.
2.  Open the Python script and update the **Key Parameters** section to match your file paths and desired settings (especially `START_POINT` and `END_POINT`).
3.  Execute the script 
4.  The script will run all three stages in sequence and print progress to the console.

**Expected Outputs:**

  * `stitched_drone_image.tiff`: The final high-resolution stitched map.
  * `final.jpg`: The stitched map with all detections drawn as red bounding boxes.
  * `Safe_Path.jpg`: The stitched map showing detections and the calculated safe path as a green line.

-----

### Step 1: Image Capture & Stitching

This stage creates the base map for all subsequent operations.

  * **Key Functions:** `stitching_process()`, `stitch_drone_videos()`, `capture_frames()`
  * **Process:**
    1.  **Frame Extraction:** The `capture_frames()` function iterates through each video specified in `drone_videos`. It intelligently extracts the very first frame and then every `STITCHING_FRAME_RATE`-th frame. Crucially, it also appends the *very last frame* of the video to ensure no data at the end of a flight path is missed.
    2.  **Stitcher Initialization:** The `stitch_drone_videos()` function uses OpenCV's stitcher, specifically requesting `cv2.Stitcher_SCANS` mode. This mode is optimized for aerial or planar scenes and is generally more robust than the default "panorama" mode for this use case.
    3.  **Stitching:** The stitcher is fed all extracted frames from all videos. It finds common keypoints, calculates homographies, and warps/blends the images into a single mosaic.
    4.  **Post-Processing:**
          * **Cropping:** `simple_crop_black_borders()` is called to find the largest non-black contour in the stitched image. This effectively crops out the black, warped border areas, leaving a clean rectangular image.
          * **Enhancement:** `enhance_brightness_contrast()` is applied to the final image, increasing brightness, contrast, and applying a sharpening filter to make features and potential obstacles more visually distinct for the detection model.
  * **Output:** A single, high-resolution `stitched_drone_image.tiff` file is saved to disk.

### Step 2: YOLO + SAHI Object Detection

This stage finds all objects of interest on the large, stitched map. Using SAHI is critical because the stitched image is far too large to be processed by YOLO at once without significant downscaling, which would cause small objects to be missed.

  * **Key Functions:** `detect_with_sahi()`, `extract_detection_data()`, `plot_detections_simple()`
  * **Process:**
    1.  **Model Loading:** `AutoDetectionModel` from SAHI is used to load the YOLO model specified by `MODEL_PATH`.
    2.  **Sliced Inference:** `get_sliced_prediction()` is the core of this step. It divides the large `stitched_drone_image.tiff` into 640x640 slices with a 25% overlap.
    3.  **Detection & Merging:** The YOLO model runs inference on each individual slice. SAHI then merges all detections from the slices back into the original image's coordinate space, automatically handling duplicate detections in the overlapping regions using Non-Maximum Suppression (NMS).
    4.  **Data Extraction:** `extract_detection_data()` converts SAHI's prediction objects into a clean `List[Dict]` containing essential data for each detection: bounding box (`x1`, `y1`, `x2`, `y2`), center point, confidence, and class name.
  * **Output:**
      * **In-Memory:** A `detections` list is passed to the next stage.
      * **On-Disk:** A `final.jpg` image is saved, visualizing all the detections found.

### Step 3: A\* Path Planning

This final stage uses the detection data to find a safe and efficient route across the map.

  * **Key Functions:** `begin_a_star()`, `PathFinder` (class), `map_detections_to_pathfinder()`
  * **Process:**
    1.  **Map Initialization:** The `PathFinder` class is initialized. It creates a 2D boolean grid (`self.obstacle_grid`) that represents the entire image, discretized into cells of `GRID_RESOLUTION`x`GRID_RESOLUTION` pixels.
    2.  **Obstacle Mapping:**
          * `map_detections_to_pathfinder()` filters the full detection list to only include classes relevant for pathfinding (e.g., 'landmine', 'mine').
          * `add_landmine_obstacles()` iterates through each "landmine" detection. It applies the `SAFETY_MARGIN` (in pixels) to the detection's bounding box, effectively enlarging it.
          * It then marks all grid cells that fall within this enlarged "danger zone" as `True` (an obstacle) in the `self.obstacle_grid`.
    3.  **A\* Algorithm:**
          * `find_path()` implements the A\* search algorithm. It converts the pixel-space `START_POINT` and `END_POINT` into grid-space coordinates.
          * Using a priority queue and a Manhattan distance `heuristic`, it efficiently explores the grid from the start node, moving in 8 directions (diagonals included) and avoiding all cells marked as obstacles.
    4.  **Path Reconstruction:** Once the algorithm reaches the end node, it backtracks to reconstruct the optimal path. This path is converted from grid coordinates back into pixel coordinates (using the center of each grid cell).
  * **Output:**
      * **On-Disk:** A `Safe_Path.jpg` image is saved, showing the final path (green line) superimposed on the map with the detected obstacles (red boxes).
      * **In-Console:** The script prints whether a path was found and, if so, the number of waypoints. The `safe_path` variable (as returned by `find_path`) contains the **list of `(x, y)` pixel coordinate tuples** that define the path.

-----


