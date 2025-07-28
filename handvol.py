import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import threading

# --- Color Ranges in HSV ---
COLOR_RANGES = {
    "orange":  ((5, 100, 100), (25, 255, 255), (0, 140, 255)),
    "red":     ((0, 100, 100), (10, 255, 255), (0, 0, 255)),
    "green":   ((40, 70, 70), (80, 255, 255), (0, 255, 0)),
    "blue":    ((100, 150, 0), (140, 255, 255), (255, 0, 0)),
    "yellow":  ((20, 100, 100), (30, 255, 255), (0, 255, 255)),
    "purple":  ((130, 50, 50), (160, 255, 255), (255, 0, 255)),
}

BLEND_MODES = ["difference", "add", "subtract", "multiply", "screen"]
MIN_CONTOUR_AREA = 100
cancel_flag = False

def apply_blend_mode(base, overlay, mode):
    if mode == "difference":
        return cv2.absdiff(base, overlay)
    elif mode == "add":
        return cv2.add(base, overlay)
    elif mode == "subtract":
        return cv2.subtract(base, overlay)
    elif mode == "multiply":
        return cv2.multiply(base, overlay, scale=1.0 / 255)
    elif mode == "screen":
        base_inv = 255 - base
        overlay_inv = 255 - overlay
        screen = 255 - cv2.multiply(base_inv, overlay_inv, scale=1.0 / 255)
        return screen
    else:
        return base  # fallback

def process_video(input_path, color_name, blend_mode, progress_bar, root, run_btn, cancel_btn):
    global cancel_flag
    cancel_flag = False

    if color_name not in COLOR_RANGES:
        messagebox.showerror("Error", f"Color {color_name} not defined.")
        return

    lower, upper, bgr = COLOR_RANGES[color_name]
    output_path = f"output_{color_name}_{blend_mode}.mp4"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open {input_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    backSub = cv2.createBackgroundSubtractorMOG2()

    frame_count = 0

    while True:
        if cancel_flag:
            break

        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        overlay = np.zeros_like(frame)
        motion = backSub.apply(frame)
        motion = cv2.erode(motion, None, iterations=1)
        motion = cv2.dilate(motion, None, iterations=5)

        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        combined = cv2.bitwise_and(mask, motion)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []

        for c in contours:
            if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                continue
            ((x, y), r) = cv2.minEnclosingCircle(c)
            mask_for_mean = np.zeros(hsv.shape[:2], dtype="uint8")
            cv2.drawContours(mask_for_mean, [c], -1, 255, -1)
            brightness = cv2.mean(hsv, mask=mask_for_mean)[2] / 255.0
            final_radius = r * (0.25 + brightness / 2)
            M = cv2.moments(c)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                centers.append(center)
                cv2.circle(overlay, center, int(final_radius), bgr, -1, cv2.LINE_AA)

        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = int(np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]))
                mid = ((centers[i][0] + centers[j][0]) // 2, (centers[i][1] + centers[j][1]) // 2)
                cv2.line(overlay, centers[i], centers[j], bgr, 2, cv2.LINE_AA)
                cv2.putText(overlay, f"{d}px", (mid[0], mid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        final = apply_blend_mode(frame, overlay, blend_mode)
        out.write(final)

        # --- Live Preview ---
        preview_frame = cv2.resize(final, (640, 360))
        cv2.imshow("Live Preview", preview_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cancel_flag = True
            break

        # --- Update Progress ---
        frame_count += 1
        percent = int((frame_count / total_frames) * 100)
        progress_bar["value"] = percent
        root.update_idletasks()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    progress_bar["value"] = 0
    run_btn.config(state="normal")
    cancel_btn.config(state="disabled")

    if cancel_flag:
        messagebox.showinfo("Cancelled", "Processing was cancelled.")
    else:
        messagebox.showinfo("Done", f"Output saved: {output_path}")
        os.startfile(output_path)

# --- GUI Setup ---
def start_gui():
    root = tk.Tk()
    root.title("Color Blob Tracker")
    root.geometry("420x360")
    root.resizable(False, False)

    selected_file = tk.StringVar()
    selected_color = tk.StringVar(value="orange")
    selected_blend_mode = tk.StringVar(value="difference")

    def browse_file():
        file = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4")])
        if file:
            selected_file.set(file)

    def start_processing():
        if not selected_file.get():
            messagebox.showwarning("Missing", "Please select a video file.")
            return
        run_btn.config(state="disabled")
        cancel_btn.config(state="normal")
        threading.Thread(target=process_video, args=(
            selected_file.get(),
            selected_color.get(),
            selected_blend_mode.get(),
            progress_bar, root, run_btn, cancel_btn
        ), daemon=True).start()

    def cancel_processing():
        global cancel_flag
        cancel_flag = True
        cancel_btn.config(state="disabled")

    # --- GUI Layout ---
    tk.Label(root, text="Select Video File:").pack(pady=10)
    tk.Button(root, text="Browse", command=browse_file).pack()
    tk.Label(root, textvariable=selected_file, wraplength=380).pack(pady=5)

    tk.Label(root, text="Select Color:").pack(pady=5)
    ttk.Combobox(root, values=list(COLOR_RANGES.keys()), textvariable=selected_color, state="readonly").pack()

    tk.Label(root, text="Blending Mode:").pack(pady=5)
    ttk.Combobox(root, values=BLEND_MODES, textvariable=selected_blend_mode, state="readonly").pack()

    progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress_bar.pack(pady=15)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    run_btn = tk.Button(button_frame, text="Start Processing", command=start_processing, bg="green", fg="white")
    run_btn.grid(row=0, column=0, padx=10)

    cancel_btn = tk.Button(button_frame, text="Cancel", command=cancel_processing, bg="red", fg="white", state="disabled")
    cancel_btn.grid(row=0, column=1, padx=10)

    root.mainloop()

if __name__ == "__main__":
    start_gui()
