# Blob Tracker

This repository contains Python scripts for tracking colored objects in videos. It provides both a simple, command-line based script for a specific color and a more advanced GUI-based tool for tracking various colors with different visual effects.

## Features

  * **Color-Based Object Tracking**: Detects and tracks objects based on their color in the HSV color space.
  * **Motion Detection**: Uses background subtraction to focus on moving objects and ignore stationary ones.
  * **Distance Measurement**: Calculates and displays the distance in pixels between detected objects.
  * **GUI Interface (`handvol.py`)**:
      * Easy-to-use interface for selecting video files, colors, and blend modes.
      * Supports tracking multiple colors: orange, red, green, blue, yellow, and purple.
      * Offers various blend modes for creative visual effects: difference, add, subtract, multiply, and screen.
      * Live preview of the video processing.
      * Progress bar and the ability to cancel the processing.
  * **Simple Script (`blobtracking.py`)**:
      * A straightforward script pre-configured to track yellow objects.
      * Demonstrates the core object tracking logic.

## Requirements

  * Python 3
  * OpenCV (`opencv-python`)
  * NumPy
  * Tkinter (usually included with Python)

You can install the required libraries using pip:

```bash
pip install opencv-python numpy
```

## Usage

### GUI-Based Tracking (`handvol.py`)

This is the recommended way to use the blob tracker.

1.  **Run the script:**
    ```bash
    python handvol.py
    ```
2.  **Select a video file:** Click the "Browse" button to choose an MP4 video file.
3.  **Select a color:** Choose the color of the objects you want to track from the dropdown menu.
4.  **Select a blending mode:** Choose a visual effect to apply to the output video.
5.  **Start processing:** Click the "Start Processing" button. A live preview will be shown, and the progress will be displayed in the progress bar.
6.  **Output:** The processed video will be saved in the same directory with a filename indicating the chosen color and blend mode (e.g., `output_orange_difference.mp4`).

### Command-Line Script (`blobtracking.py`)

This script is pre-configured to track yellow objects in a video named `carousel_animation.mp4`.

1.  **Configure the script:**
      * Open `blobtracking.py` in a text editor.
      * Change the `INPUT_VIDEO_PATH` and `OUTPUT_VIDEO_PATH` variables to your desired input and output files.
      * You can also modify the `COLOR_CONFIG` to track a different color by changing the `lower` and `upper` HSV values.
2.  **Run the script:**
    ```bash
    python blobtracking.py
    ```
3.  **Output:** The processed video will be saved to the path specified in `OUTPUT_VIDEO_PATH`.

## How It Works

The object tracking process involves the following steps:

1.  **Frame Capture:** The script reads the input video frame by frame.
2.  **Color Space Conversion:** Each frame is converted from BGR to the HSV (Hue, Saturation, Value) color space, which is more effective for color-based filtering.
3.  **Motion Detection:** A background subtractor (MOG2) is used to create a motion mask, identifying areas with movement.
4.  **Color Masking:** A color mask is created by thresholding the HSV image for the selected color range.
5.  **Combined Mask:** The motion mask and color mask are combined using a bitwise AND operation. This ensures that only moving objects of the specified color are detected.
6.  **Contour Detection:** The `cv2.findContours` function is used to find the outlines of the detected objects in the combined mask.
7.  **Object Analysis:**
      * Contours smaller than a minimum area are ignored.
      * For each valid contour, the center and radius of the enclosing circle are calculated.
      * The brightness of the object is used to adjust the size of the circle drawn on the output.
8.  **Distance Calculation:** The script calculates the Euclidean distance between the centers of all detected objects.
9.  **Visualization:**
      * Circles are drawn around the detected objects.
      * Lines are drawn to connect the objects, with the distance displayed.
10. **Blend Mode Application:** The final frame is created by blending the original frame with the overlay of detected objects using the selected blend mode.
11. **Output:** The processed frame is written to the output video file.
