# Pixelmess: The Glitch Art Workstation <img src="logo.png" height="40" align="bottom"/>

**Pixelmess** is a real-time visual performance tool that turns your webcam or video files into reactive, high-energy glitch art. Designed for creative coders, VJs, and content creators, it breaks free from static filters with a "drag-and-drop" interface for instant cyberpunk visuals.

## Key Features

*   **Live & Offline**: Jam live with your webcam or batch-process video clips for high-quality production.
*   **Dynamic Tracking**: Automatically detects motion to apply effects only where it matters—on the action.
*   **Retro-Future Aesthetic**: Built-in "Win2K", "CRT", "Thermal", and "Dither" effects.
*   **Smart Export**: Render 60FPS MP4s with a single click, ready for social media.
*   **Shape-Aware Labels**: Intelligent labeling system that adapts to both circular and rectangular tracking regions.

## Installation

### For Users (Windows)
1.  Go to the **[Releases](../../releases)** page.
2.  Download `Pixelmess.exe`.
3.  Run it. No installation required.

### For Developers
1.  Clone the repository:
    ```bash
    git clone https://github.com/Jalpan04/blob-tracker.git
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the app:
    ```bash
    python app.py
    ```

## Controls

| Area | Function |
| :--- | :--- |
| **Shape** | Toggle between Square, Rect, and Circle tracking regions. |
| **Region Style** | Choose the visual style (e.g., Scope, Dash, L-Frame). |
| **Filter Effects** | Apply pixel shaders like CRT, Thermal, or Edge detection. |
| **Export** | Toggle Webcam recording or Export video files. |

## Credits

Created by **Jalpan Vyas**.
Powered by **Python**, **OpenCV**, and **Dear PyGui**.

---
*License: MIT*
