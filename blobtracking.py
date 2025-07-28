import cv2
import numpy as np
import math
import itertools

# --- Configuration ---
# Specify the path to your input video file
INPUT_VIDEO_PATH = 'carousel_animation.mp4'
# Specify the path for the output video file
OUTPUT_VIDEO_PATH = 'output_orange_difference.mp4'

# Minimum size for an object to be considered (in pixels).
MIN_CONTOUR_AREA = 100

# Define the properties for only the orange color
COLOR_CONFIG = [
    {
        "name": "yellow",
        "lower": np.array([20, 100, 100]),  # Lower bound for yellow in HSV
        "upper": np.array([30, 255, 255]),  # Upper bound for yellow in HSV
        "bgr": (0, 255, 255)  # BGR color for drawing (bright yellow)
    }
]

# --- Initialization ---
# Start video capture from the specified file
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file at {INPUT_VIDEO_PATH}")
    exit()

# Initialize the background subtractor for motion detection
backSub = cv2.createBackgroundSubtractorMOG2()

# Get video properties for the output file
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

print("Processing video... Press 'q' on the video window to stop early.")

# --- Main Processing Loop ---
while cap.isOpened():
    # 1. Capture a frame
    ret, frame = cap.read()
    if not ret:
        print("Reached end of video.")
        break

    # Create a black overlay. We will draw our shapes on this.
    # For difference blend, a black overlay results in no change where there are no shapes.
    overlay = np.zeros_like(frame, dtype=np.uint8)

    # 2. Create a single motion mask for the frame
    motion_mask = backSub.apply(frame)
    motion_mask = cv2.erode(motion_mask, None, iterations=1)
    motion_mask = cv2.dilate(motion_mask, None, iterations=5)

    # Convert the frame to HSV color space once for all color checks
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 3. Loop through each color configuration to detect objects (now only orange)
    for color in COLOR_CONFIG:
        # 3a. Create a mask for the current color
        color_mask = cv2.inRange(hsv, color["lower"], color["upper"])

        color_mask = cv2.erode(color_mask, None, iterations=2)
        color_mask = cv2.dilate(color_mask, None, iterations=2)

        # 3b. Combine color and motion masks
        combined_mask = cv2.bitwise_and(color_mask, motion_mask)

        # 3c. Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_centers = []

        # 3d. Process each contour found for this color
        for c in contours:
            if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                continue

            # Get the base circle properties
            ((x, y), base_radius) = cv2.minEnclosingCircle(c)

            # Calculate the average brightness (Value in HSV) within the contour
            mask_for_mean = np.zeros(hsv.shape[:2], dtype="uint8")
            cv2.drawContours(mask_for_mean, [c], -1, 255, -1)
            mean_hsv = cv2.mean(hsv, mask=mask_for_mean)
            brightness = mean_hsv[2] / 255.0  # Normalize brightness to 0-1

            # Scale the radius based on brightness
            final_radius = base_radius * (0.25 + brightness/2)

            # Calculate centroid
            M = cv2.moments(c)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                detected_centers.append(center)

                # Draw the filled circle ON THE OVERLAY with anti-aliasing for smooth edges
                cv2.circle(overlay, (int(x), int(y)), int(final_radius), color["bgr"], -1, cv2.LINE_AA)

        # 3e. Calculate and draw distances between objects
        if len(detected_centers) >= 2:
            for point1, point2 in itertools.combinations(detected_centers, 2):
                distance = math.hypot(point2[0] - point1[0], point2[1] - point1[1])

                # Draw the connecting line ON THE OVERLAY with anti-aliasing for smoothness
                cv2.line(overlay, point1, point2, color["bgr"], 2, cv2.LINE_AA)

                midpoint = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
                distance_text = f"{distance:.2f} px"
                # Draw the text ON THE OVERLAY. Use a smaller font scale and anti-aliasing.
                cv2.putText(overlay, distance_text, (midpoint[0], midpoint[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Apply the difference blend mode
    # cv2.absdiff calculates the per-element absolute difference between the two images.
    final_frame = cv2.absdiff(frame, overlay)

    # 5. Write the processed frame to the output video
    video_writer.write(final_frame)

    # 6. Display the result
    cv2.imshow("Object Tracking", final_frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Processing stopped by user.")
        break

# --- Cleanup ---
print(f"Processing complete. Output video saved to {OUTPUT_VIDEO_PATH}")
cap.release()
video_writer.release()
cv2.destroyAllWindows()
