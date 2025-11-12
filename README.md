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


---

## 📂 Project Files Overview

### 1. `stitcher_yolo_path_finding.py`

**Description:**
The **initial implementation script** that integrates drone video stitching, YOLO + SAHI detection, and path planning in a single workflow.

---

### 2. `drone_landmine_detector.py`

**Description:**
A **modular, importable Python library** version of the original implementation.
This script organizes the system into **five main classes**, each handling a specific functional block of the drone landmine detection and path planning pipeline.

#### 🧩 Class Overview

---

#### **1. `DroneImageStitcher`**

Handles **drone video stitching and enhancement**.

**Main Methods:**

* `capture_frames()` — Extracts frames from input drone videos.
* `stitch()` — Stitches multiple video frames into a large panoramic image.
* `enhance_brightness_contrast()` — Enhances image brightness and contrast for better visibility.
* `crop_black_borders()` — Removes black borders from the stitched output.

---

#### **2. `LandmineDetector`**

Handles **YOLO + SAHI-based detection** of landmines or obstacles.

**Main Methods:**

* `load_model()` — Loads the custom YOLO model for inference.
* `detect()` — Performs object detection using SAHI for large stitched images.
* `print_summary()` — Displays detection statistics (e.g., count per class).
* `visualize_detections()` — Draws bounding boxes and labels on the image.

---

#### **3. `PathFinder`**

Implements an **A*** (A-star) **pathfinding algorithm** for safe route planning.

**Main Methods:**

* `add_obstacles()` — Marks detected landmines as obstacles in the grid.
* `find_path()` — Computes a safe path from a given start to end coordinate.
* `visualize_path()` — Draws the computed path over the stitched image.

---

#### **4. `DroneLandmineSystem`**

Represents the **complete end-to-end system** integrating all components.

**Main Methods:**

* `stitch_videos()` — Handles video stitching using `DroneImageStitcher`.
* `detect_landmines()` — Performs landmine detection using `LandmineDetector`.
* `find_safe_path()` — Executes safe pathfinding using `PathFinder`.
* `run_complete_pipeline()` — Runs the full workflow automatically (stitch → detect → plan).

---

#### **5. (Optional Utility Classes / Helpers)**

> *(If applicable in your implementation — for logging, visualization, or configuration.)*

---

### 3. `example.py`

**Description:**
Demonstrates **example usage** of the `DroneLandmineSystem` library with **five test cases** showing different pipeline configurations or datasets.



-----

# PIPELINE\_B: Advanced Speech-to-Command

This project implements a complete, hands-free voice command pipeline. It is designed to run locally, starting from wake-word detection and ending with confirmed command execution.

The full flow is:
**Wake Word -\> Voice Activity Detection (VAD) -\> Speech-to-Text (STT) -\> Intent Analysis -\> User Confirmation (TTS -\> VAD -\> STT) -\> Execution**

## Core Technologies

  * **Wake Word:** [Picovoice Porcupine](https://picovoice.ai/platform/porcupine/) for highly-accurate, low-power "drone swarm" detection.
  * **VAD:** [Silero VAD](https://github.com/snakers4/silero-vad) to detect when a user starts and stops speaking.
  * **Speech-to-Text (STT):** [Vosk](https://alphacephei.com/vosk/) for fast, offline transcription.
  * **Intent Analysis:** A custom Python function (`intent_analyser`) to map transcribed text to specific service calls and parameters.
  * **Audio I/O:** [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) for capturing microphone data.

## Workflow

The `speech_to_command.py` script follows this logical loop:

1.  **Listen for Wake Word:** Porcupine continuously processes the audio stream, waiting for the "drone swarm" keyword.
2.  **Command Recording:**
      * Once the wake word is detected, the system plays an alert (simulated by a `print` statement) and begins recording.
      * `record_until_silence()` uses Silero VAD to capture audio until the user stops speaking for a set duration (1.5s).
3.  **Command Transcription:**
      * The recorded audio is saved to `spoken_command.wav`.
      * `transcribe_vosk()` transcribes the audio.
      * **Crucially**, it passes the *entire list of valid commands* (from `intent_analyser(get_commands=True)`) to the Vosk recognizer as a grammar list. This significantly improves accuracy by biasing the STT model to pick a valid command over a similar-sounding, invalid one.
4.  **Intent Analysis:**
      * `intent_analyser()` parses the transcribed text (e.g., "start one").
      * It returns a `service` (e.g., `/initialise`) and `params` (e.g., `1`).
      * If the command is not recognized, it returns `/unknown`.
5.  **Confirmation Loop:**
      * **Invalid Command:** If the service is `/unknown`, it announces "invalid command" (via `TEXT_4_TEXT_TO_SPEECH`) and loops back to step 1.
      * **Valid Command:**
        1.  The system announces the intended action for confirmation: `f"do i execute: {text_transcribed}, with service: {service}"`.
        2.  It immediately calls `record_until_silence()` again (with a longer silence limit of 2.3s) to listen for "yes" or "no".
        3.  The response is saved to `response_to_command.wav` and transcribed.
6.  **Execution / Cancellation:**
      * If the user says **"yes"**, the `TO_EXECUTE` flag is set to `True`, and an "Executing" message is prepared.
      * If the user says **"no"**, the `TO_EXECUTE` flag is set to `False`, and a "Denied" message is prepared.
      * If the response is not "yes" or "no", it's treated as an invalid response, and the loop restarts.
7.  **Loop:** The process restarts from step 1, waiting for the wake word.

> **Note on TTS:** The script generates messages in the `TEXT_4_TEXT_TO_SPEECH` variable. These are currently printed to the console. A full implementation would pipe this text to a Text-to-Speech engine (e.g., Silero TTS, pyttsx3) to be spoken aloud.

## Project Structure

For the script to run correctly, your project must be organized as follows:

```
speech-to-command/
├── models/
│   ├── vosk-model-small-en-us-0.15/
│   │   ├── am/
│   │   ├── conf/
│   │   └── ... (all other vosk model files)
│   ├── Drone-Swarm_en_windows_v3_0_0.ppn
│   └── start-swarm_en_raspberry-pi_v3_0_0.ppn
│
├── .env
├── requirements.txt
├── voicecommandcontroller.py
├── usage_example.py
├── tts.py
└── speech_to_command.py
```

  * **`models/`**: This folder contains all necessary models.
  * **`models/vosk-model-small-en-us-0.15/`**: The *unzipped* directory containing the Vosk model files.
  * **`models/*.ppn`**: The Porcupine keyword files, *unzipped* and placed here.

## Setup & Installation

### 1\. Prerequisites

  * Python 3.8+
  * A Picovoice Access Key (get one for free from [Picovoice Console](https://console.picovoice.ai/))
  * A working microphone.

### 2\. Create Project Directory

Set up the folder structure as shown above.

  * Download and unzip the [Vosk model](https://alphacephei.com/vosk/models) (`vosk-model-small-en-us-0.15`) into the `models/` folder.
  * Download and unzip your Porcupine wake word files (`.ppn`) into the `models/` folder.

### 3\. Python Environment & Dependencies

It is highly recommended to use a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 4\. Install Requirements

Create a file named `requirements.txt` with the following content:

```ini
# Core deep learning stack (CPU-only, Windows compatible)
torch==2.8.0+cpu
torchaudio==2.8.0+cpu
--extra-index-url https://download.pytorch.org/whl/cpu

# Audio and numeric dependencies
soundfile==0.13.1
numpy==2.3.4
packaging==24.1

# Vosk STT
vosk==0.3.45

# Porcupine Wake Word
pvporcupine==3.0.2

# PyAudio for mic access
pyaudio==0.2.14

# .env file support
python-dotenv==1.0.1
```

Install the requirements:

```bash
pip install -r requirements.txt
```

### 5\. Configuration

1.  **Create the `.env` file** in the root of your project:

    ```
    PORCUPINE_ACCESS_KEY="YOUR_ACCESS_KEY_HERE"
    ```

    Replace `"YOUR_ACCESS_KEY_HERE"` with your actual key from the Picovoice Console.

2.  **Set Audio Device (if needed):**
    The script defaults to `input_device_index=2`. This may not be correct for your system. To find the right index, run this small script:

    ```python
    import pyaudio
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"Index: {i} - Name: {info['name']}")
    p.terminate()
    ```

    Find the index of your microphone and change this line in `speech_to_command.py`:

    ```python
    audio_stream = pa.open(
        ...
        input_device_index=2,  # <-- CHANGE THIS
        ...
    )
    ```

3.  **Set Wake Word:**
    The script is set to use the Windows wake word.
    WINDOWS NATIVE WAKE PHRASE: "DRONE SWARM"
    ```python
    porcupine = pvporcupine.create(
        access_key=ACCESS_KEY,
        keyword_paths=["models/Drone-Swarm_en_windows_v3_0_0.ppn"] # <-- CHANGE THIS
    )
    ```

    To use the Raspberry Pi model, change the path to `"models/start-swarm_en_raspberry-pi_v3_0_0.ppn"`.
    RPI NATIVE WAKE PHRASE: "START SWARM"
## Usage

Once all setup and configuration steps are complete, run the script from your activated virtual environment:

```bash
python speech_to_command.py
```

The script will initialize all models (this may take a few seconds) and then print:

`🚀 Listening for wake word... (Ctrl+C to stop)`

You can now say "drone swarm" to activate the command pipeline.

## Command Reference

The `intent_analyser` function supports the following commands.

| Voice Command | Service Call | Parameters |
| :--- | :--- | :--- |
| "start one" / "start two" ... | `/initialise` | `1`, `2`, `3`, or `4` |
| "start" | `/initialise` | `"ALL_DRONES"` |
| "scan one" / "scan two" ... | `/generate_scan_waypoints` | `1`, `2`, `3.`, or `4` |
| "scan" | `/generate_scan_waypoints` | `"ALL_DRONES"` |
| "pause scan one" ... | `/pause_drone` | `1`, `2`, `3`, or `4` |
| "pause scan" | `/pause_drone` | `"ALL_DRONES"` |
| "resume scan one" ... | `/resume_drone` | `1`, `2`, `3`, or `4` |
| "resume scan" | `/resume_drone` | `"ALL_DRONES"` |
| "restart scan one" ... | `/restart_scan` | `1`, `2`, `3`, or `4` |
| "restart scan" | `/restart_scan` | `"ALL_DRONES"` |
| "mark one" / "mark two" ... | `/mark_mines` | `1`, `2`, `3`, or `4` |
| "mark" | `/mark_mines` | `"ALL_DRONES"` |
| "pause mark one" ... | `/mark_mines_pause` | `1`, `2`, `3`, or `4` |
| "pause mark" | `/mark_mines_pause` | `"ALL_DRONES"` |
| "resume mark one" ... | `/mark_mines_resume` | `1`, `2`, `3`, or `4` |
| "resume mark" | `/mark_mines_resume` | `"ALL_DRONES"` |
| "generate path" | `/generate_path` | `{}` |
| "start guidance" | `/start_guidance` | `{}` |
| "pause guidance" | `/pause_guidance` | `{}` |
| **Confirmation** | | |
| "yes" | (Confirms execution) | `N/A` |
| "no" | (Cancels execution) | `N/A` |
| **Fallback** | | |
| *(any other phrase)* | `/unknown` | `{}` |

---

## 1\. Documentation: `voicecommandcontroller.py`

This module defines the `VoiceCommandController`, a high-level Python class that encapsulates the entire speech-to-command pipeline. It is designed to be a reusable, "plug-and-play" library for adding voice control to other applications.

The class handles:

  * Loading and managing all necessary models.
  * Opening and controlling the audio stream.
  * Detecting a wake word.
  * Recording a command using Voice Activity Detection (VAD).
  * Transcribing the command using Speech-to-Text (STT).
  * Parsing the command into a service call.
  * Handling the "yes/no" confirmation loop.
  * Providing a callback to the parent application with the final, confirmed command.

### Core Technologies

| Technology | Purpose | Library |
| :--- | :--- | :--- |
| **Wake Word** | Detects the "drone swarm" hotword. | [Picovoice Porcupine](https://picovoice.ai/platform/porcupine/) |
| **VAD** | Detects when a user starts/stops speaking. | [Silero VAD](https://github.com/snakers4/silero-vad) |
| **STT** | Transcribes spoken audio to text. | [Vosk](https://alphacephei.com/vosk/) |
| **Audio I/O** | Captures microphone data. | PyAudio |

-----

### Class: `VoiceCommandController`

`class VoiceCommandController(porcupine_access_key, wake_word_path, vosk_model_path, ...)`

A library class for voice-controlled drone swarm commands. Handles wake word detection, speech recording, transcription, and command parsing.

#### Initialization (`__init__`)

When you create an instance of this class, it loads all models (VAD, Vosk, Porcupine) and opens the audio stream.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `porcupine_access_key`| `Optional[str]` | `None` | Your Picovoice AccessKey. If `None`, it loads from `.env`. |
| `wake_word_path` | `str` | `"models/..."` | Path to the `.ppn` wake word file. |
| `vosk_model_path` | `str` | `"models/..."` | Path to the Vosk model directory. |
| `audio_device_index`| `Optional[int]`| `2` | The index of your microphone. Use `None` for default. |
| `sample_rate` | `int` | `16000` | Audio sample rate (must be 16000 for Vosk/Silero). |

#### Main Public Methods

These are the primary methods you will use to interact with the controller.

  * `run(blocking: bool = True)`

      * Starts the main listening loop.
      * If `blocking=True` (default), this function will run forever, listening for the wake word and handling commands until you stop the script (Ctrl+C).
      * If `blocking=False`, it will process *one* wake word/command cycle and then return.

  * `set_command_callback(callback: Callable)`

      * This is the **most important method** for getting data out.
      * You pass it a function (the "callback") that will be called *after* a command is fully processed (including "yes/no" confirmation).
      * Your callback function **must** accept three arguments:
        1.  `service` (str): The parsed service name (e.g., `"/initialise"`).
        2.  `params` (any): The parameters for the service (e.g., `1` or `"ALL_DRONES"`).
        3.  `execute` (bool): `True` if the user said "yes," `False` if they said "no".

  * `process_wake_word_detection() -> bool`

      * Processes a single frame of audio. Returns `True` if the wake word was detected *in that frame*, otherwise `False`.
      * This is used for **non-blocking** loops (see Example 3 in `usage_example.py`).

  * `handle_voice_command() -> Optional[Tuple[str, any, bool]]`

      * Runs the full command-handling flow:
        1.  Plays a "beep" (prints to console).
        2.  Records command with VAD.
        3.  Transcribes command.
        4.  Parses command.
        5.  Asks for confirmation.
        6.  Records "yes/no" response.
        7.  Transcribes response.
      * Returns a tuple `(service, params, execute_bool)` if successful, or `None` if the user gave an invalid command or response.

  * `cleanup()`

      * Closes the audio stream and releases all resources from Porcupine and PyAudio. This is critical to call when your program exits.

#### Context Manager Support

The class can be used as a context manager to ensure `cleanup()` is called automatically. This is the recommended way to use the class.

```python
with VoiceCommandController() as controller:
    # controller is initialized and audio stream is open
    controller.run()
# __exit__ is called, and controller.cleanup() runs automatically
```

#### Utility & Static Methods

These methods are public and can be used for convenience or testing.

  * `parse_command(input_text: str) -> Tuple[str, any]`

      * The core intent parser. Maps text like `"start one"` to `("/initialise", 1)`.

  * `get_supported_commands() -> List[str]`

      * A **`staticmethod`** that returns the complete list of all valid command phrases (including "yes" and "no"). This list is fed to Vosk to improve transcription accuracy.

  * `transcribe_vosk(audio_path: str) -> str`

      * Transcribes a given `.wav` file using the loaded Vosk model and the built-in command list as a grammar.

  * `record_until_silence(...) -> Optional[np.ndarray]`

      * Records audio from the mic until VAD detects silence.

  * `save_wav(audio_data: np.ndarray, filename: str)`

      * Saves a NumPy array of audio data to a `.wav` file.

#### Command Reference

The `parse_command` method supports the following commands:

| Voice Command | Service Call | Parameters |
| :--- | :--- | :--- |
| "start one" / "start two" ... | `/initialise` | `1`, `2`, `3`, or `4` |
| "start" | `/initialise` | `"ALL_DRONES"` |
| "scan one" / "scan two" ... | `/generate_scan_waypoints` | `1`, `2`, `3`, or `4` |
| "scan" | `/generate_scan_waypoints` | `"ALL_DRONES"` |
| "pause scan one" ... | `/pause_drone` | `1`, `2`, `3`, or `4` |
| "pause scan" | `/pause_drone` | `"ALL_DRONES"` |
| ... (*all other commands*) ... | ... | ... |
| "generate path" | `/generate_path` | `{}` |
| "start guidance" | `/start_guidance` | `{}` |
| "pause guidance" | `/pause_guidance` | `{}` |

-----

## 2\. Documentation: `usage_example.py`

This file provides several examples of how to import and use the `VoiceCommandController` library.

### Example 1: Simple Usage with Callback

This is the most common and straightforward way to use the controller. It uses the `blocking=True` mode of `run()` and handles all commands through a callback.

```python
def command_handler(service, params, execute):
    """Handle executed commands"""
    if execute:
        print(f"✅ Executing: {service} with params {params}")
        # Add your ROS service call or other logic here
    else:
        print(f"❌ Command cancelled: {service}")

# Create controller
controller = VoiceCommandController(audio_device_index=2)
# Register the callback
controller.set_command_callback(command_handler)
# Run forever
controller.run()
```

### Example 2: Using Context Manager

This is the recommended, safest pattern. It's identical to Example 1 but uses a `with` statement to ensure `controller.cleanup()` is *always* called, even if the program crashes.

```python
with VoiceCommandController() as controller:
    controller.set_command_callback(my_command_callback)
    controller.run()
```

### Example 3: Manual Control (Non-blocking)

This advanced pattern is for integrating the controller into an existing application's main loop (e.g., a GUI, a game, or a ROS node) without blocking it.

```python
controller = VoiceCommandController()
try:
    while True:
        # Your other application logic can go here
        # e.g., update_gui(), check_robot_status()

        # Check for a wake word
        if controller.process_wake_word_detection():
            print("Wake word detected!")

            # Handle the full command interaction
            result = controller.handle_voice_command()

            if result:
                service, params, execute = result
                if execute:
                    print(f"Execute: {service}({params})")
                    # Your custom logic here

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    controller.cleanup()
```

### Example 4: Get List of Supported Commands

This shows how to access the static list of all valid command phrases.

```python
commands = VoiceCommandController.get_supported_commands()
print("Supported commands:", commands)
```

### Example 5: Parse Command

This shows how to use the controller's parsing logic by itself for testing or other purposes.

```python
controller = VoiceCommandController()
service, params = controller.parse_command("start scan one")
print(f"Command parsed: {service} with {params}")
controller.cleanup()
```

-----

## 3\. Documentation: `tts.py`: Local Text-to-Speech (TTS) Module

## Overview

This module provides simple Python wrappers for two local, offline Text-to-Speech (TTS) engines: **eSpeak NG** and **Mimic 1**. It is designed to be cross-platform, with specific playback logic for Windows and Linux (e.g., Raspberry Pi).

The functions work by calling the external command-line tools for each engine, generating speech, and then playing the resulting audio.

## Prerequisites & Installation

This module **requires external command-line tools to be installed** on your system.

### 1\. For `speak_espeak_ng` (Windows & Linux)

This is a versatile, cross-platform engine.

  * **Windows:**

    1.  Download the **`...msi` installer** for eSpeak NG. The latest version can be found on the [eSpeak NG GitHub Releases page](https://github.com/espeak-ng/espeak-ng/releases).
    2.  Run the installer. This will add `espeak-ng.exe` to your system's `PATH`, allowing the script to find it.

  * **Linux (Debian/Raspberry Pi):**

    1.  Install the `espeak-ng` package:
        ```bash
        sudo apt update
        sudo apt install espeak-ng
        ```
    2.  Install `aplay` (part of `alsa-utils`) for audio playback:
        ```bash
        sudo apt install alsa-utils
        ```

### 2\. For `speak_mimic` (Linux / Raspberry Pi)

This engine provides a more natural-sounding voice and is primarily intended for Linux systems.

1.  Install the `mimic` package:
    ```bash
    sudo apt update
    sudo apt install mimic
    ```
2.  Install `aplay` (if not already installed) for audio playback when saving to a file:
    ```bash
    sudo apt install alsa-utils
    ```

## Function Reference

-----

### `speak_espeak_ng`

```python
speak_espeak_ng(text, output_file="output.wav", speed=150, pitch=50, voice="en-us")
```

Generates speech using `eSpeak NG`, saves it to a `.wav` file, and then plays it. This function is cross-platform (Windows/Linux).

**Parameters:**

  * **`text`** (`str`): The text you want to speak.
  * **`output_file`** (`str`, optional): The path to save the temporary `.wav` file. Defaults to `"output.wav"`.
  * **`speed`** (`int`, optional): The speaking speed in words per minute. Defaults to `150`.
  * **`pitch`** (`int`, optional): The voice pitch (0-99). Defaults to `50`.
  * **`voice`** (`str`, optional): The voice/language code to use (e.g., `"en-us"`, `"fr"`, `"es"`). Defaults to `"en-us"`.

**Playback Logic:**

  * The function automatically detects the operating system.
  * On **Windows**, it uses a PowerShell `SoundPlayer` command to play the audio *synchronously* (the script will wait for playback to finish).
  * On **Linux**, it uses the `aplay` command to play the audio.

-----

### `speak_mimic`

```python
speak_mimic(text: str, voice: str = "ap", save_to_file: bool = False)
```

Generates speech using Mycroft's `Mimic 1`. This is primarily intended for Linux/Raspberry Pi.

**Parameters:**

  * **`text`** (`str`): The text you want to speak.
  * **`voice`** (`str`, optional): The Mimic 1 voice to use. `"ap"` (Alan Pope) is a common, high-quality voice. Defaults to `"ap"`.
  * **`save_to_file`** (`bool`, optional):
      * If `False` (default), `mimic` streams the audio directly to the speaker.
      * If `True`, the audio is first saved to `"speech.wav"` and then played using `aplay`.

**Error Handling:**

  * This function first checks if `mimic` is available in the system's `PATH` using `shutil.which()`.
  * If the `mimic` executable is not found, it will print an error message and return without speaking.

## Example Usage

The main block of the script provides clear examples for both functions.

```python
if __name__ == "__main__":
    
    # --- Example for Windows ---
    # This function works on both Windows and Linux,
    # but is the recommended one for Windows.
    print("Testing eSpeak NG...")
    speak_espeak_ng(
        text="Hello! This is eSpeak NG speaking.",
        output_file="speech.wav",
        speed=160,
        pitch=55,
        voice="en-us"
    )

    # --- Example for Raspberry Pi (Linux) ---
    # This function is recommended for Linux/RPi for a more natural voice.
    # Make sure 'mimic' is installed first!
    print("\nTesting Mimic 1...")
    # The following line is commented out to prevent errors
    # if you are not on a Raspberry Pi. Uncomment to test.
    
    # speak_mimic("Hello! This is mimic one speaking.")
```
----
