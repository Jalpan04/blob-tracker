import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import time
import threading
import math
import itertools

class BlobTrackerApp:
    def __init__(self):
        self.width = 1600
        self.height = 900
        self.sidebar_width = 380
        
        # State
        self.video_source = None
        self.cap = None
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.is_exporting = False
        
        # Texture dimensions (fixed for now or dynamic)
        self.tex_w = 1280
        self.tex_h = 720
        
        # --- Parameters ---
        # Recording
        self.recording = False
        self.writer = None
        self.output_file = "output.mp4"

        # Shapes
        self.shape_type = "Square" 
        
        # Region Style (Effects)
        self.effect_style = "None" 
        
        # Filters
        self.filter_type = "None"
        
        # Connection
        self.connection_enabled = True
        self.line_style = "Solid" # Solid, Dashed
        self.stroke_width = 1
        self.connection_rate = 0.25 
        
        # Blob
        self.blob_size_min = 50
        
        # Color
        self.blob_color = (0, 0, 0, 255) 
        
        # Tracking (Motion Only)
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        
        # Init DPG
        dpg.create_context()
        dpg.create_viewport(title='Pixelmess', width=self.width, height=self.height, small_icon="logo.png", large_icon="logo.png")
        dpg.setup_dearpygui()
        
        self.setup_theme()
        self.init_cv()
        self.setup_ui()

    def setup_theme(self):
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (20, 20, 20))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (30, 30, 30))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (60, 60, 60))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 50, 50))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (70, 70, 70))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 165, 0)) # Orange accent
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 8)
        dpg.bind_theme(global_theme)

    def init_cv(self):
        # Init texture
        data = np.zeros((self.tex_h, self.tex_w, 4), dtype=np.float32)
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(width=self.tex_w, height=self.tex_h, default_value=data.flatten(), tag="video_texture")

    # --- Callbacks ---
    def load_video_callback(self, sender, app_data):
        fpath = app_data.get('file_path_name', '')
        
        # Clean path: dpg sometimes returns wildcard pattern
        import os
        if not os.path.exists(fpath):
            if fpath.endswith(".*"):
                fpath = fpath[:-2]
            
            if not os.path.exists(fpath):
                print(f"Error: File not found: {fpath}")
                return

        print(f"Loading video: {fpath}")
        self.video_source = fpath
        with self.lock:
            if self.cap: self.cap.release()
            self.cap = cv2.VideoCapture(fpath)
        dpg.configure_item("btn_record", label="Export Video")
        
    def export_save_callback(self, sender, app_data):
        # Callback for the SAVE dialog
        fpath = app_data.get('file_path_name', '')
        if not fpath: return
        
        # Ensure extension
        if not fpath.lower().endswith('.mp4'):
            fpath += ".mp4"
            
        threading.Thread(target=self.run_batch_export, args=(fpath,), daemon=True).start()

    def toggle_recording(self, s, a, u):
        # Branch based on source
        if self.video_source == "Webcam" or self.video_source is None:
            # LIVE RECORDING (Webcam)
            if not self.recording:
                self.recording = True
                dpg.configure_item("btn_record", label="Stop Recording")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(self.output_file, fourcc, 30.0, (self.tex_w, self.tex_h))
                print("Started Live Recording...")
            else:
                self.recording = False
                dpg.configure_item("btn_record", label="Start Recording")
                if self.writer:
                    self.writer.release()
                    self.writer = None
                print(f"Saved to {self.output_file}")
        else:
            # OFFLINE EXPORT (Video File)
            # Open Save Dialog instead of auto-starting
            dpg.show_item("save_dialog")

    def run_batch_export(self, output_path="export_video.mp4"):
        print(f"Starting Batch Export to {output_path}...")
        dpg.configure_item("btn_record", label="Exporting...", enabled=False)
        self.is_exporting = True
        
        # Setup Export
        cap = cv2.VideoCapture(self.video_source)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Get FPS from source if possible, else 30
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.tex_w, self.tex_h))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if not ret: break
            
            final = self.process_frame(frame)
            out.write(final)
            
            count += 1
            if count % 100 == 0: print(f"Exported {count}/{total_frames}")

        cap.release()
        out.release()
        
        print("Export Complete!")
        self.is_exporting = False
        dpg.configure_item("btn_record", label="Export Video", enabled=True)
    def set_shape(self, s, a, user_data): self.shape_type = user_data
    def set_effect(self, s, a, user_data): self.effect_style = user_data
    def set_filter(self, s, a, user_data): self.filter_type = user_data
    def set_stroke(self, s, a): self.stroke_width = int(a)
    def set_blob_min(self, s, a, user_data): self.blob_size_min = user_data
    def set_color(self, s, a): 
        if isinstance(a, (list, tuple)):
            # DPG returns 0.0-1.0 floats mostly, check and scale if needed
            if max(a) <= 1.0:
                self.blob_color = [int(x * 255) for x in a]
            else:
                self.blob_color = [int(x) for x in a]

    def set_line_style(self, s, a, user_data): self.line_style = user_data
    
    
    def load_webcam_callback(self, s, a, u):
        print("Loading Webcam...")
        self.video_source = "Webcam"
        with self.lock:
            if self.cap: self.cap.release()
            self.cap = cv2.VideoCapture(0)
        dpg.configure_item("btn_record", label="Start Recording")

    def reset_settings(self):
        # Stop recording if active
        if self.recording:
            self.toggle_recording(None, None, None)
            
        self.shape_type = "Square"
        self.effect_style = "None"
        self.filter_type = "None"
        self.connection_enabled = True
        self.line_style = "Solid"
        self.stroke_width = 1
        self.blob_size_min = 50
        self.blob_color = (0, 0, 0, 255)
        
        # Update UI to reflect state
        dpg.set_value("cb_connection", True)
        dpg.set_value("slider_stroke", 1)
        dpg.set_value("color_picker", (0, 0, 0, 255))
        # We can't easily update all button highlights without complex logic, 
        # but the functional state is reset.

    def build_grid_buttons(self, items, cols, callback):
        # Grid layout
        rows = [items[i:i + cols] for i in range(0, len(items), cols)]
        for row in rows:
            with dpg.group(horizontal=True):
                for item in row:
                     # Calculate accurate width to fill sidebar
                     # sidebar_width = 380, padding ~8*cols
                     btn_w = (self.sidebar_width - (10 * cols)) // cols
                     dpg.add_button(label=item, width=btn_w, callback=callback, user_data=item)

    def setup_ui(self):
        # File Dialog
        # User requested ONLY mp4 and mov
        with dpg.file_dialog(directory_selector=False, show=False, callback=self.load_video_callback, tag="file_dialog", width=700, height=400):
            dpg.add_file_extension(".mp4", color=(0, 255, 0, 255))
            dpg.add_file_extension(".mov", color=(0, 255, 0, 255))
            
        # Save Dialog
        with dpg.file_dialog(directory_selector=False, show=False, callback=self.export_save_callback, tag="save_dialog", width=700, height=400, default_filename="my_export"):
            dpg.add_file_extension(".mp4", color=(255, 255, 0, 255))

        # Main Layout
        with dpg.window(tag="Primary Window"):
            
            with dpg.group(horizontal=True):
                
                # --- SIDEBAR ---
                with dpg.child_window(width=self.sidebar_width, border=False):
                    dpg.add_text("PIXELMESS", color=(255, 255, 255)) # Styled later
                    dpg.add_text("Made by Jalpan Vyas", color=(150, 150, 150))
                    dpg.add_separator()
                    
                    # Actions
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Upload Video", width=110, callback=lambda: dpg.show_item("file_dialog"))
                        dpg.add_button(label="Webcam", width=110, callback=self.load_webcam_callback)
                        dpg.add_button(label="Reset", width=110, callback=self.reset_settings)

                    dpg.add_spacer(height=20)
                    
                    # Export Video
                    dpg.add_text("Export")
                    dpg.add_button(label="Start Recording", width=150, callback=self.toggle_recording, tag="btn_record")
                    
                    dpg.add_spacer(height=20)

                    dpg.add_spacer(height=20)

                    # Shape
                    dpg.add_text("Shape")
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Rect", width=110, callback=self.set_shape, user_data="Rect")
                        dpg.add_button(label="Square", width=110, callback=self.set_shape, user_data="Square")
                        dpg.add_button(label="Circle", width=110, callback=self.set_shape, user_data="Circle")
                        
                    dpg.add_spacer(height=20)
                    
                    # Region Style (Basic Effects)
                    dpg.add_text("Region Style")
                    region_items = ["None", "Label", "Frame", 
                                    "L-Frame", "X-Frame", "Grid",
                                    "Dash", "Scope",
                                    "Win2K", "Label 2"]
                    self.build_grid_buttons(region_items, 3, self.set_effect)
                    
                    dpg.add_spacer(height=20)
                    
                    # Filter Effects
                    dpg.add_text("Filter Effects")
                    filter_items = ["None", "Edge", "Invert", "CRT",
                                    "X-Ray", "Thermal", "Dither", "Pixel",
                                    "Blur"]
                    self.build_grid_buttons(filter_items, 3, self.set_filter)
                    
                    dpg.add_spacer(height=20)

                    # Connection
                    dpg.add_text("Connection")
                    dpg.add_checkbox(label="Connect Blobs", default_value=True, tag="cb_connection")
                    dpg.add_slider_float(label="Connect Rate", default_value=0.25, max_value=1.0, tag="slider_rate")
                    with dpg.group(horizontal=True):
                         dpg.add_text("Style:")
                         dpg.add_button(label="Solid", width=50, callback=self.set_line_style, user_data="Solid")
                         dpg.add_button(label="Dashed", width=60, callback=self.set_line_style, user_data="Dashed")
                    
                    dpg.add_text("Stroke Width")
                    dpg.add_slider_int(tag="slider_stroke", default_value=1, max_value=10, callback=self.set_stroke)

                    dpg.add_spacer(height=20)
                    
                    # Blob Size
                    dpg.add_text("Blob Min Size")
                    with dpg.group(horizontal=True):
                         dpg.add_button(label="5", width=60, callback=self.set_blob_min, user_data=5)
                         dpg.add_button(label="20", width=60, callback=self.set_blob_min, user_data=20)
                         dpg.add_button(label="50", width=60, callback=self.set_blob_min, user_data=50)
                         dpg.add_button(label="100", width=60, callback=self.set_blob_min, user_data=100)
                    
                    dpg.add_spacer(height=20)
                    
                    # Color
                    dpg.add_text("Color")
                    dpg.add_color_picker(display_hex=True, default_value=self.blob_color, callback=self.set_color, width=300, height=150, tag="color_picker")
                    
                    dpg.add_spacer(height=20)
                    

                    
                    # Removed Tracking Calibration (Motion is default)
                    dpg.add_text("Tracking: Motion Only (Auto)")

                # --- VIEWPORT ---
                with dpg.child_window(border=False):
                    dpg.add_image("video_texture")

        dpg.set_primary_window("Primary Window", True)

    # --- Processing Logic ---
    def process_frame(self, frame):
        # Resize inputs to tex size ??? Or resize tex to fit?
        # For simplicity, resize frame to 1280x720
        frame = cv2.resize(frame, (self.tex_w, self.tex_h))
        


        # 2. Tracking (Motion Only)
        mask = self.back_sub.apply(frame)
        # Remove shadows (gray) -> binary
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        
        # Clean noise
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for c in contours:
            if cv2.contourArea(c) < self.blob_size_min * 10: # Scale slider value
                continue
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
                (x,y,w,h) = cv2.boundingRect(c)
                
                # Force Square Aspect Ratio
                if self.shape_type == "Square" or self.shape_type == "Circle":
                    side = max(w, h)
                    cx = x + w // 2
                    cy = y + h // 2
                    x = cx - side // 2
                    y = cy - side // 2
                    w = side
                    h = side

                # --- ROI Filter Logic ---
                # Clamp ROI to frame bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                
                if w > 0 and h > 0:
                    try:
                        roi = frame[y:y+h, x:x+w]
                        
                        if self.filter_type == "Edge":
                            # Sobel/Canny: Interior Black, Edges White
                            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            edges = cv2.Canny(gray, 100, 200)
                            roi = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                            
                        elif self.filter_type == "Invert":
                            roi = cv2.bitwise_not(roi)
                            
                        elif self.filter_type == "CRT":
                            # Scanlines
                            rh, rw = roi.shape[:2]
                            scanlines = np.zeros((rh, rw, 3), dtype=np.uint8)
                            scanlines[::3] = 50 
                            roi = cv2.subtract(roi, scanlines)
                            
                        elif self.filter_type == "X-Ray":
                            # Invert + Cyan Tint
                            roi = cv2.bitwise_not(roi)
                            # Tint: Blue/Green boost
                            b, g, r = cv2.split(roi)
                            b = cv2.add(b, 50)
                            g = cv2.add(g, 30)
                            r = cv2.multiply(r, 0.5)
                            roi = cv2.merge([b, g, np.array(r, dtype=np.uint8)])
                            
                        elif self.filter_type == "Thermal":
                             gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                             roi = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                             
                        elif self.filter_type == "Dither":
                             # 1-Bit threshold
                             gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                             _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                             roi = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                             
                        elif self.filter_type == "Pixel":
                             rh, rw = roi.shape[:2]
                             if rh > 10 and rw > 10:
                                 temp = cv2.resize(roi, (rw//10, rh//10), interpolation=cv2.INTER_LINEAR)
                                 roi = cv2.resize(temp, (rw, rh), interpolation=cv2.INTER_NEAREST)
                                 
                        elif self.filter_type == "Blur":
                             roi = cv2.GaussianBlur(roi, (21, 21), 0)

                        # Paste back
                        # Paste back
                        if self.shape_type == "Circle":
                            # Circular Mask Logic
                            mask_img = np.zeros((h, w), dtype=np.uint8)
                            center = (w // 2, h // 2)
                            radius = min(w, h) // 2
                            cv2.circle(mask_img, center, radius, 255, -1)
                            
                            # Copy filtered ROI only where mask is white
                            # Original frame pixels remain where mask is black
                            cv2.copyTo(roi, mask_img, frame[y:y+h, x:x+w])
                        else:
                            # Standard Rectangular Paste
                            frame[y:y+h, x:x+w] = roi
                    except Exception as e:
                        pass # Safety skip if ROI fails
                
                # 3. Region Style & Shape Drawing
                color_bgr = (self.blob_color[2], self.blob_color[1], self.blob_color[0])
                t = max(1, self.stroke_width)
                
                # Special Case: Win2K is always a Window Box
                if self.effect_style == "Win2K":
                    # Retro 3D border + Blue Header + X button
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (192, 192, 192), 2)
                    header_h = 20
                    cv2.rectangle(frame, (x, y-header_h), (x+w, y), (128, 0, 0), -1)
                    cv2.putText(frame, "Error", (x+2, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                    btn_size = 14
                    bx, by = x+w-btn_size-2, y-header_h+3
                    cv2.rectangle(frame, (bx, by), (bx+btn_size, by+btn_size), (192,192,192), -1)
                    cv2.line(frame, (bx+3, by+3), (bx+btn_size-3, by+btn_size-3), (0,0,0), 1)
                    cv2.line(frame, (bx+3, by+btn_size-3), (bx+btn_size-3, by+3), (0,0,0), 1)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), color_bgr, 1)

                elif self.shape_type == "Circle":
                    # --- CIRCULAR STYLES ---
                    center = (x + w // 2, y + h // 2)
                    radius = w // 2
                    
                    if self.effect_style == "Grid":
                        # Radar Style (Concentric Rings)
                        cv2.circle(frame, center, radius, color_bgr, t)
                        cv2.circle(frame, center, radius // 2, color_bgr, 1)
                        cv2.circle(frame, center, radius // 4, color_bgr, 1)
                        cv2.line(frame, (center[0]-radius, center[1]), (center[0]+radius, center[1]), color_bgr, 1)
                        cv2.line(frame, (center[0], center[1]-radius), (center[0], center[1]+radius), color_bgr, 1)
                        
                    elif self.effect_style == "Scope":
                        # Circle + Compass Ticks
                        cv2.circle(frame, center, radius, color_bgr, t)
                        tl = 10 
                        cv2.line(frame, (center[0], y), (center[0], y+tl), color_bgr, t) # N
                        cv2.line(frame, (center[0], y+h), (center[0], y+h-tl), color_bgr, t) # S
                        cv2.line(frame, (x, center[1]), (x+tl, center[1]), color_bgr, t) # W
                        cv2.line(frame, (x+w, center[1]), (x+w-tl, center[1]), color_bgr, t) # E
                        
                    elif self.effect_style == "Dash":
                        # Dashed Circle (Arcs)
                        n_dashes = 12
                        for i in range(n_dashes):
                            start_angle = (i * 360 / n_dashes)
                            end_angle = start_angle + (360 / n_dashes * 0.5)
                            cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, color_bgr, t)
                            
                    elif self.effect_style == "Frame":
                         # Circle + Nodes
                         cv2.circle(frame, center, radius, color_bgr, 1)
                         r_node = 4
                         cv2.circle(frame, (center[0], y), r_node, color_bgr, -1) # N
                         cv2.circle(frame, (center[0], y+h), r_node, color_bgr, -1) # S
                         cv2.circle(frame, (x, center[1]), r_node, color_bgr, -1) # W
                         cv2.circle(frame, (x+w, center[1]), r_node, color_bgr, -1) # E

                    elif self.effect_style == "X-Frame":
                         # Circle + X inside
                         cv2.circle(frame, center, radius, color_bgr, 1)
                         p = int(radius * 0.707) # 45 deg point
                         cv2.line(frame, (center[0]-p, center[1]-p), (center[0]+p, center[1]+p), color_bgr, 1)
                         cv2.line(frame, (center[0]+p, center[1]-p), (center[0]-p, center[1]+p), color_bgr, 1)
                         
                    elif self.effect_style == "L-Frame":
                         # Arc Corners
                         radius = w // 2
                         angles = [45, 135, 225, 315]
                         for ang in angles:
                             start = ang - 20
                             end = ang + 20
                             cv2.ellipse(frame, center, (radius, radius), 0, start, end, color_bgr, t)

                    else:
                        # Default Solid
                        cv2.circle(frame, center, radius, color_bgr, t)

                else:
                    # --- RECT / SQUARE STYLES (Original Logic) ---
                    if self.effect_style == "Grid":
                        steps = 8
                        dy = h // steps
                        for i in range(1, steps):
                            cv2.line(frame, (x, y + i*dy), (x+w, y + i*dy), color_bgr, 1)
                        cv2.rectangle(frame, (x,y), (x+w, y+h), color_bgr, t)

                    elif self.effect_style == "Scope":
                        cv2.rectangle(frame, (x,y), (x+w, y+h), color_bgr, t)
                        tick_len = 10
                        cv2.line(frame, (x+w//2, y), (x+w//2, y+tick_len), color_bgr, t)
                        cv2.line(frame, (x+w//2, y+h), (x+w//2, y+h-tick_len), color_bgr, t)
                        cv2.line(frame, (x, y+h//2), (x+tick_len, y+h//2), color_bgr, t)
                        cv2.line(frame, (x+w, y+h//2), (x+w-tick_len, y+h//2), color_bgr, t)

                    elif self.effect_style == "Dash":
                         # Solid Corners
                         cl = 5 
                         cv2.line(frame, (x, y), (x+cl, y), color_bgr, t)
                         cv2.line(frame, (x, y), (x, y+cl), color_bgr, t)
                         cv2.line(frame, (x+w-cl, y), (x+w, y), color_bgr, t)
                         cv2.line(frame, (x+w, y), (x+w, y+cl), color_bgr, t)
                         cv2.line(frame, (x, y+h-cl), (x, y+h), color_bgr, t)
                         cv2.line(frame, (x, y+h), (x+cl, y+h), color_bgr, t)
                         cv2.line(frame, (x+w-cl, y+h), (x+w, y+h), color_bgr, t)
                         cv2.line(frame, (x+w, y+h-cl), (x+w, y+h), color_bgr, t)
                         # Dashes
                         dash_len = 10
                         gap_len = 5
                         for i in range(x, x+w, dash_len+gap_len):
                             cv2.line(frame, (i, y), (min(i+dash_len, x+w), y), color_bgr, t)
                             cv2.line(frame, (i, y+h), (min(i+dash_len, x+w), y+h), color_bgr, t)
                         for i in range(y, y+h, dash_len+gap_len):
                             cv2.line(frame, (x, i), (x, min(i+dash_len, y+h)), color_bgr, t)
                             cv2.line(frame, (x+w, i), (x+w, min(i+dash_len, y+h)), color_bgr, t)

                    elif self.effect_style == "X-Frame":
                         cv2.rectangle(frame, (x,y), (x+w, y+h), color_bgr, 1)
                         cv2.line(frame, (x, y), (x+w, y+h), color_bgr, 1)
                         cv2.line(frame, (x+w, y), (x, y+h), color_bgr, 1)

                    elif self.effect_style == "Frame":
                         cv2.rectangle(frame, (x,y), (x+w, y+h), color_bgr, 1)
                         r = 4
                         cv2.circle(frame, (x, y), r, color_bgr, -1)
                         cv2.circle(frame, (x+w, y), r, color_bgr, -1)
                         cv2.circle(frame, (x, y+h), r, color_bgr, -1)
                         cv2.circle(frame, (x+w, y+h), r, color_bgr, -1)

                    elif self.effect_style == "L-Frame":
                         l = min(w, h) // 3
                         cv2.line(frame, (x,y), (x+l, y), color_bgr, t)
                         cv2.line(frame, (x,y), (x, y+l), color_bgr, t)
                         cv2.line(frame, (x+w-l, y), (x+w, y), color_bgr, t)
                         cv2.line(frame, (x+w, y), (x+w, y+l), color_bgr, t)
                         cv2.line(frame, (x, y+h-l), (x, y+h), color_bgr, t)
                         cv2.line(frame, (x, y+h), (x+l, y+h), color_bgr, t)
                         cv2.line(frame, (x+w-l, y+h), (x+w, y+h), color_bgr, t)
                         cv2.line(frame, (x+w, y+h-l), (x+w, y+h), color_bgr, t)

                    else:
                        # None -> Solid
                        cv2.rectangle(frame, (x,y), (x+w, y+h), color_bgr, t)

                # Labels
                if "Label" in self.effect_style:
                     label_text = f"Object {len(centers)}" # Mimic "Object 1" style
                     font = cv2.FONT_HERSHEY_DUPLEX
                     font_scale = 0.5
                     (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, 1)
                     
                     if self.effect_style == "Label":
                        # Style 1: External Tag (Top-Left, Outside)
                        # Box (Only for Rect/Square)
                        if self.shape_type != "Circle":
                             cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 1) 
                        
                        # Tag Background (White)
                        cv2.rectangle(frame, (x, y-th-8), (x+tw+6, y), (255,255,255), -1)
                        # Text (Black)
                        cv2.putText(frame, label_text, (x+3, y-6), font, font_scale, (0,0,0), 1)
                        
                     elif self.effect_style == "Label 2":
                        # Style 2: Internal Tag
                        if self.shape_type == "Circle":
                             # For Circle: Place truly INSIDE (Centered)
                             # No box needed, just text on background
                             cx, cy = x + w // 2, y + h // 2
                             # Center text: cx - half_width, cy + half_height
                             tx = cx - tw // 2
                             ty = cy + th // 2
                             
                             # Tag Background (White) centered
                             cv2.rectangle(frame, (tx-4, ty-th-4), (tx+tw+4, ty+4), (255,255,255), -1)
                             # Text (Black)
                             cv2.putText(frame, label_text, (tx, ty), font, font_scale, (0,0,0), 1)
                        else:
                             # For Rect: Inside Top-Left Corner
                             cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 1)
                             cv2.rectangle(frame, (x, y), (x+tw+6, y+th+8), (255,255,255), -1)
                             cv2.putText(frame, label_text, (x+3, y+th+4), font, font_scale, (0,0,0), 1)

        # 4. Connections
        if dpg.get_value("cb_connection") and len(centers) >= 2:
            color_bgr = (self.blob_color[2], self.blob_color[1], self.blob_color[0])
            
            # Dynamic Threshold (Rate 0.0-1.0 => 0-1000px)
            rate = dpg.get_value("slider_rate")
            if rate is None: rate = 0.25
            thresh = rate * 1000.0
            
            t = max(1, self.stroke_width)

            for p1, p2 in itertools.combinations(centers, 2):
                dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                if dist < thresh: 
                    if self.line_style == "Solid":
                        cv2.line(frame, p1, p2, color_bgr, t)
                    elif self.line_style == "Dashed":
                        # Draw dashed line manually
                        # Vector p1->p2
                        x1, y1 = p1
                        x2, y2 = p2
                        total_len = dist
                        dash_len = 10
                        gap_len = 10
                        
                        dx = (x2 - x1) / total_len
                        dy = (y2 - y1) / total_len
                        
                        curr = 0
                        while curr < total_len:
                            start_x = int(x1 + dx * curr)
                            start_y = int(y1 + dy * curr)
                            end_x = int(x1 + dx * min(curr + dash_len, total_len))
                            end_y = int(y1 + dy * min(curr + dash_len, total_len))
                            cv2.line(frame, (start_x, start_y), (end_x, end_y), color_bgr, t)
                            curr += dash_len + gap_len

        return frame

    def loop(self):
        while self.running:
            if self.is_exporting:
                time.sleep(0.1)
                continue
                
            with self.lock:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    # Process
                    final = self.process_frame(frame)
                    
                    if self.recording and self.writer is not None:
                         self.writer.write(final)

                    final_rgba = cv2.cvtColor(final, cv2.COLOR_BGR2RGBA)
                    data = final_rgba.astype(np.float32) / 255.0
                    
                    dpg.set_value("video_texture", data.flatten())
            
            # Standard 60 FPS cap
            time.sleep(1/60)

    def run(self):
        dpg.show_viewport()
        t = threading.Thread(target=self.loop, daemon=True)
        t.start()
        dpg.start_dearpygui()
        self.running = False
        t.join(timeout=1.0)
        dpg.destroy_context()

if __name__ == "__main__":
    app = BlobTrackerApp()
    app.run()
