import numpy as np
import cv2
import time
from typing import List, Dict


# Set memory limit for large image processing
cv2.setNumThreads(4)  # for limit threads to manage memory usage

# PARAMETERS TO CUSTOMIZE:
DRONE_1_VIDEO_PATH = "./drone_1.mp4"
DRONE_2_VIDEO_PATH = "./drone_2.mp4"
DRONE_3_VIDEO_PATH = "./drone_3.mp4"
DRONE_4_VIDEO_PATH = "./drone_4.mp4"
STITCHING_FRAME_RATE = 10  # every nth frame will be considered for stitching
BORDER_SIZE = 5 # for border thinning of the stitched image

MODEL_PATH = r"E:\dataset_img\yolo_dataset\runs\detect\train10\weights\best.pt"
MODEL_CONFIDENCE_THRESHOLD = 0.3  # for yolo detection, allow only objects above 30% confidence

GRID_RESOLUTION = 10 # for pathfinding resolution
SAFETY_MARGIN = 30 # for pathfinding safety margin
"""
example of start and end points
START_POINT = (50, 100)    # 50 pixels from left, 100 pixels from top
END_POINT = (1800, 800)    # 1800 pixels from left, 800 pixels from top
PS: filhaal ms paint se pixels dekh lena
"""
START_POINT = (184, 210)  # start point for pathfinding
END_POINT = (1721, 643) # end point for pathfinding


#STICHING FUNCTIONS
def capture_frames(video_path, frame_interval):

    if not video_path or video_path == "path/to/drone1.mp4":
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return []

    images = []
    frame_count = 0
    last_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Always keep track of the last valid frame
        last_frame = frame.copy()

        # Take first frame and every nth frame
        if frame_count == 0 or frame_count % frame_interval == 0:
            images.append(frame)

        frame_count += 1

    cap.release()

    # Ensure last frame is included (if it's not already in the list)
    if last_frame is not None and len(images) > 0:
        # Check if last frame is different from the last captured frame
        if not np.array_equal(images[-1], last_frame):
            images.append(last_frame)
            print(f"Added last frame to ensure complete coverage")

    print(f"Extracted {len(images)} frames from {video_path}")
    return images
def enhance_brightness_contrast(image, brightness, contrast):
    """
    Simple image enhancement: brightness, contrast, and sharpening

    Args:
        image: Input image
        brightness: Brightness adjustment (0-100, typically 15-30)
        contrast: Contrast multiplier (1.0-2.0, typically 1.1-1.3)
    """
    # Step 1: Adjust brightness and contrast
    enhanced = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    # Step 2: Apply simple sharpening kernel
    sharpening_kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])

    # Apply sharpening filter
    sharpened = cv2.filter2D(enhanced, -1, sharpening_kernel)

    # Blend original and sharpened (to avoid over-sharpening)
    result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)

    return result
def resize_if_too_large(images, max_dimension=3000):
    # Resize images if they're too large to prevent memory issues

    if not images:
        return images, 1.0

    # Check the size of first image
    h, w = images[0].shape[:2]
    max_dim = max(h, w)

    if max_dim > max_dimension:
        scale_factor = max_dimension / max_dim
        print(f"Resizing images by {scale_factor:.3f} to prevent memory issues")

        resized_images = []
        for img in images:
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            resized_images.append(resized)

        return resized_images, scale_factor

    return images, 1.0
def simple_crop_black_borders(image):
    # Perform simple cropping to remove black borders without being too aggressive

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # get all non-black pixels
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (main image area)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add small padding to avoid cutting content
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        # Crop the image
        cropped = image[y:y + h, x:x + w]
        print(f"Cropped from {image.shape[1]}x{image.shape[0]} to {cropped.shape[1]}x{cropped.shape[0]}")
        return cropped

    return image
def stitch_drone_videos(drone_paths, frame_interval, enhance_image=True):
    """
    agruments to the function:-
        drone_paths: list of paths to drone videos
        frame_interval: extract every nth frame
        enhance_image: whether to enhance brightness/contrast
    """

    # Step 1:extract frames from all videos
    all_images = []

    for i, video_path in enumerate(drone_paths):
        frames = capture_frames(video_path, frame_interval)

        if frames:
            all_images.extend(frames)

    if not all_images:
        print("No images found! Check your video paths.")
        return None

    print(f"\nTotal images for stitching: {len(all_images)}")
    print(f"Original image size: {all_images[0].shape[1]}x{all_images[0].shape[0]}")

    # Step 2: Resize if needed to prevent memory issues
    all_images, scale_factor = resize_if_too_large(all_images, max_dimension=5000)

    if scale_factor < 1.0:
        print(f"Images resized by factor {scale_factor:.3f}")

    # Step 3: Create and configure stitcher
    print(f"\nStarting stitching process...")

    try:
        # Try SCANS mode first (better for aerial footage)
        stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
        print("Using SCANS mode for aerial footage")
    except:
        # Fallback to default mode
        stitcher = cv2.Stitcher_create()
        print("Using default stitching mode")

    # Configure stitcher for better quality
    try:
        stitcher.setRegistrationResol(0.6)
        stitcher.setSeamEstimationResol(0.1)
        stitcher.setCompositingResol(-1)  # Full resolution for final result
        stitcher.setPanoConfidenceThresh(0.8)
    except:
        print("Warning: Could not set advanced stitcher parameters")

    # Step 4: Perform stitching
    status, stitched = stitcher.stitch(all_images)

    if status == cv2.Stitcher_OK:
        print(f"Raw stitched size: {stitched.shape[1]}x{stitched.shape[0]}")

        # Step 5: Post-processing
        print("\n Post-processing...")

        # Step 5.1 Remove black borders
        stitched = simple_crop_black_borders(stitched)

        # Step 6: Enhance brightness and contrast
        if enhance_image:
            print("Enhancing brightness and contrast...")

            # Enhance the image
            stitched = enhance_brightness_contrast(
                stitched,
                brightness=20,  # Increase brightness
                contrast=1.15  # Increase contrast
            )

        print(f"Final image size: {stitched.shape[1]}x{stitched.shape[0]}")
        return stitched

    else:
        # error messages
        error_messages = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed - check image overlap",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed"
        }

        error_msg = error_messages.get(status, f"Unknown error (code: {status})")
        print(f"Stitching failed: {error_msg}")
        return None
def stitching_process(drone_1_video_path, drone_2_video_path, drone_3_video_path, drone_4_video_path, stitching_frame_rate):

    # Configure your video paths here
    drone_videos = [
        drone_1_video_path,
        drone_2_video_path,
        drone_3_video_path,
        drone_4_video_path
    ]

    # Perform stitching
    result = stitch_drone_videos(
        drone_videos,
        frame_interval=stitching_frame_rate,
        enhance_image=True  # Enable brightness enhancement
    )

    if result is not None:
        # Save the result
        output_path = "stitched_drone_image.tiff"
        success = cv2.imwrite(output_path, result)
        if success:
            print(f"Stitching Success")



# SAHI FUNCTIONS

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
def detect_with_sahi(image_path):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL_PATH,
        confidence_threshold=MODEL_CONFIDENCE_THRESHOLD,
        device="cpu",  # or 'cuda:0'
    )
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
    )
    object_prediction_list = result.object_prediction_list
    return object_prediction_list
def extract_detection_data(object_prediction_list) -> List[Dict]:

    detections = []

    for i, prediction in enumerate(object_prediction_list):
        # bounding box object
        bbox = prediction.bbox

        x1 = bbox.minx  # Left
        y1 = bbox.miny  # Top
        x2 = bbox.maxx  # Right
        y2 = bbox.maxy  # Bottom

        # get center coords
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # get width n height
        width = x2 - x1
        height = y2 - y1

        # get confidence score and class
        confidence = prediction.score.value
        class_id = prediction.category.id
        class_name = prediction.category.name

        detection = {
            'id': i + 1,
            'bbox': {
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2)
            },
            'center': {
                'x': float(center_x),
                'y': float(center_y)
            },
            'dimensions': {
                'width': float(width),
                'height': float(height),
                'area': float(width * height)
            },
            'confidence': float(confidence),
            'class': {
                'id': int(class_id),
                'name': str(class_name)
            }
        }

        detections.append(detection)

    return detections
def print_detection_summary(detections: List[Dict]):

    print(f"Total detections: {len(detections)}")

    if not detections:
        print("No landmines detected!")
        return

    # stats to get pred quality
    confidences = [d['confidence'] for d in detections]
    areas = [d['dimensions']['area'] for d in detections]

    print(f"avg confidence: {np.mean(confidences):.3f}")
    print(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
    print(f"avg detection area: {np.mean(areas):.1f} pixels²")
def plot_detections_simple(image_path, detections, output_path="final.jpg"):
    image = cv2.imread(image_path)
    print(f"plotting {len(detections)} red bounding boxes")

    # get coords and draw boxes
    for detection in detections:
        x1 = int(detection['bbox']['x1'])
        y1 = int(detection['bbox']['y1'])
        x2 = int(detection['bbox']['x2'])
        y2 = int(detection['bbox']['y2'])

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # save img
    cv2.imwrite(output_path, image)
    print(f"finished saving image to: {output_path}")



# PATH FINDING FUNCTIONS

from typing import List, Tuple
import heapq
class PathFinder:
    def __init__(self, image_width: int, image_height: int, grid_resolution: int = 10):
        self.width = image_width
        self.height = image_height
        self.grid_resolution = grid_resolution  # pixels per grid cell
        self.grid_width = image_width // grid_resolution
        self.grid_height = image_height // grid_resolution
        self.obstacle_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)

    def add_landmine_obstacles(self, detections: List[dict], safety_margin: int = 50):
        """
        detections: list of dicts with keys like {'x1', 'y1', 'x2', 'y2', 'confidence'}
        """
        for detection in detections:
            # Convert bounding box to grid coordinates with safety margin
            x1 = max(0, (detection['x1'] - safety_margin) // self.grid_resolution)
            y1 = max(0, (detection['y1'] - safety_margin) // self.grid_resolution)
            x2 = min(self.grid_width, (detection['x2'] + safety_margin) // self.grid_resolution)
            y2 = min(self.grid_height, (detection['y2'] + safety_margin) // self.grid_resolution)

            # Mark area as obstacle
            self.obstacle_grid[y1:y2 + 1, x1:x2 + 1] = True

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (8-directional movement)"""
        x, y = node
        neighbors = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                if (0 <= nx < self.grid_width and
                        0 <= ny < self.grid_height and
                        not self.obstacle_grid[ny, nx]):
                    neighbors.append((nx, ny))

        return neighbors

    def find_path(self, start_pixel: Tuple[int, int], end_pixel: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm"""
        # Convert pixel coordinates to grid coordinates
        start_grid = (start_pixel[0] // self.grid_resolution, start_pixel[1] // self.grid_resolution)
        end_grid = (end_pixel[0] // self.grid_resolution, end_pixel[1] // self.grid_resolution)

        if self.obstacle_grid[start_grid[1], start_grid[0]] or self.obstacle_grid[end_grid[1], end_grid[0]]:
            return []  # Start or end point is blocked

        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, end_grid)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == end_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    # Convert back to pixel coordinates
                    pixel_coord = (current[0] * self.grid_resolution + self.grid_resolution // 2,
                                   current[1] * self.grid_resolution + self.grid_resolution // 2)
                    path.append(pixel_coord)
                    current = came_from[current]

                # Add start point
                start_pixel_center = (start_grid[0] * self.grid_resolution + self.grid_resolution // 2,
                                      start_grid[1] * self.grid_resolution + self.grid_resolution // 2)
                path.append(start_pixel_center)

                return path[::-1]  # Reverse to get start->end order

            for neighbor in self.get_neighbors(current):
                # Cost of moving to neighbor (diagonal moves cost more)
                dx, dy = abs(neighbor[0] - current[0]), abs(neighbor[1] - current[1])
                move_cost = 1.414 if dx + dy == 2 else 1  # √2 for diagonal

                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found


    def visualize_path(self, image: np.ndarray, path: List[Tuple[int, int]],
                       detections: List[dict]) -> np.ndarray:
        """Draw path and obstacles on the image"""
        result_img = image.copy()

        # Draw landmine detections
        for det in detections:
            cv2.rectangle(result_img, (det['x1'], det['y1']),
                          (det['x2'], det['y2']), (0, 0, 255), 2)
            cv2.putText(result_img, f"Mine: {det['confidence']:.2f}",
                        (det['x1'], det['y1'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw path
        if len(path) > 1:
            for i in range(len(path) - 1):
                cv2.line(result_img, path[i], path[i + 1], (0, 255, 0), 3)

            # Mark start and end points
            cv2.circle(result_img, path[0], 8, (255, 0, 0), -1)  # Blue start
            cv2.circle(result_img, path[-1], 8, (0, 255, 255), -1)  # Yellow end

        return result_img
def begin_a_star(detections, img_path, grid_resolution, safety_margin, start_point, end_point):
    img = cv2.imread(img_path)
    img_width = img.shape[1]
    img_height = img.shape[0]
    # Initialize pathfinder
    pathfinder = PathFinder(image_width=img_width, image_height=img_height, grid_resolution=grid_resolution)
    landmine_detections = map_detections_to_pathfinder(detections)
    pathfinder.add_landmine_obstacles(landmine_detections, safety_margin=safety_margin)

    # Find path from start to end point
    start_point = start_point  # Click coordinates on image
    end_point = end_point  # Click coordinates on image

    safe_path = pathfinder.find_path(start_point, end_point)

    if safe_path:
        print(f"Safe path found with {len(safe_path)} waypoints")
        # Visualize result
        result_image = pathfinder.visualize_path(img, safe_path, landmine_detections)
        cv2.imwrite("Safe_Path.jpg", result_image)
    else:
        print("No safe path found!")
def map_detections_to_pathfinder(your_detections: List[dict]) -> List[dict]:
    """
    Convert your detection format to the format expected by pathfinder
    """
    landmine_detections = []

    for detection in your_detections:
        # Only include landmine detections (assuming 'landmine' is the class name)
        if detection['class']['name'].lower() in ['landmine', 'mine']:  # adjust class names as needed
            mapped_detection = {
                'x1': int(detection['bbox']['x1']),
                'y1': int(detection['bbox']['y1']),
                'x2': int(detection['bbox']['x2']),
                'y2': int(detection['bbox']['y2']),
                'confidence': detection['confidence'],
                'center_x': int(detection['center']['x']),
                'center_y': int(detection['center']['y']),
                'area': detection['dimensions']['area'],
                'class_name': detection['class']['name'],
                'detection_id': detection['id']
            }
            landmine_detections.append(mapped_detection)

    return landmine_detections



if __name__ == "__main__":
    # stitching:-
    print("Starting stitching")
    stitching_start_time = time.time()
    stitching_process(DRONE_1_VIDEO_PATH, DRONE_2_VIDEO_PATH, DRONE_3_VIDEO_PATH, DRONE_4_VIDEO_PATH, STITCHING_FRAME_RATE)
    stitching_time = time.time() - stitching_start_time
    print(f"Stitching completed in: {stitching_time:.1f} seconds")

    time.sleep(2)

    # detection with SAHI(slicing Aided Hyper Inference):-
    IMAGE_PATH = "./stitched_drone_image.tiff"
    print("starting detection")
    start_time = time.time()
    object_prediction_list = detect_with_sahi(IMAGE_PATH)
    processing_time = time.time() - start_time
    print(f"SAHI detection completed in: {processing_time:.1f} seconds")
    detections = extract_detection_data(object_prediction_list)
    print_detection_summary(detections)
    plot_detections_simple(IMAGE_PATH, detections, output_path="final.jpg")
    time.sleep(2)

    # path finder:-
    print("starting path finder")
    start_time = time.time()
    begin_a_star(detections, IMAGE_PATH, GRID_RESOLUTION, SAFETY_MARGIN, START_POINT, END_POINT)
    processing_time = time.time() - start_time
    print(f"Path finder completed in: {processing_time:.1f} seconds")






