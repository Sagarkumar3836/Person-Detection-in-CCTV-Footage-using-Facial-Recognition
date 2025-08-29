import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
import cv2
import face_recognition
import numpy as np
import os

# Global variables for file paths
video_path = None
image_path = None
detected_faces = []

def select_video():
    global video_path
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        # filetypes=[("MP4 files", "*.mp4")]

        filetypes = [
    ("Video files", "*.mp4 *.mkv *.mov"),
    ("MP4 files", "*.mp4"),
    ("MKV files", "*.mkv"),
    ("MOV files", "*.mov"),
    ("All files", "*.*")
]

    )
    if video_path:
        video_label.config(text=os.path.basename(video_path))

def select_image():
    global image_path
    image_path = filedialog.askopenfilename(
        title="Select Reference Image",
        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )
    if image_path:
        # Display the selected image
        img = Image.open(image_path)
        img.thumbnail((200, 200))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img  # Keep reference

def start_detection():
    if not video_path or not image_path:
        messagebox.showwarning("Warning", "Please select both video and reference image.")
        return

    # Clear previous results
    result_box.config(state=tk.NORMAL)
    result_box.delete(1.0, tk.END)
    result_box.insert(tk.END, "Processing video...\n")
    for widget in face_frame.winfo_children():
        widget.destroy()

    # Load the reference image and encode the face
    reference_image = face_recognition.load_image_file(image_path)
    reference_encoding = face_recognition.face_encodings(reference_image)[0]

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Get video properties
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # Analyze one frame per second
    frame_number = 0
    timestamps = []

    # Reset and start the progress bar
    progress_bar['value'] = 0
    progress_bar['maximum'] = total_frames

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Update progress bar
        progress_bar['value'] = frame_number
        root.update_idletasks()

        # Only analyze every nth frame for speed
        if frame_number % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces([reference_encoding], face_encoding)
                face_distance = face_recognition.face_distance([reference_encoding], face_encoding)

                if matches[0] and face_distance[0] < 0.6:
                    timestamp = frame_number / fps
                    result_box.insert(tk.END, f"Person found at {timestamp:.2f} seconds.\n")
                    timestamps.append(timestamp)

                    # Extract and display face thumbnail
                    face_image = rgb_frame[top:bottom, left:right]
                    face_pil = Image.fromarray(face_image)
                    face_pil.thumbnail((100, 100))
                    face_thumb = ImageTk.PhotoImage(face_pil)

                    thumb_label = tk.Label(face_frame, image=face_thumb)
                    thumb_label.image = face_thumb  # Keep reference
                    thumb_label.pack()

        frame_number += 1

    video_capture.release()

    if not timestamps:
        result_box.insert(tk.END, "No match found.")
    else:
        result_box.insert(tk.END, "\nTimestamps where the person was detected:\n")
        for ts in timestamps:
            result_box.insert(tk.END, f"{ts:.2f} seconds\n")
        
        # Save timestamps to a file
        with open("detection_timestamps.txt", "w") as f:
            f.write("Timestamps where the person was detected:\n")
            for ts in timestamps:
                f.write(f"{ts:.2f} seconds\n")
        
        result_box.insert(tk.END, "\nTimestamps saved to detection_timestamps.txt")

    result_box.config(state=tk.DISABLED)

# Create Tkinter window
root = tk.Tk()
root.title("Person Detection in Video")

# Video Selection
tk.Label(root, text="Select Video File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
video_button = tk.Button(root, text="Browse", command=select_video)
video_button.grid(row=0, column=1, padx=5, pady=5)
video_label = tk.Label(root, text="No file selected.")
video_label.grid(row=0, column=2, padx=5, pady=5)

# Image Selection
tk.Label(root, text="Select Reference Image:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
image_button = tk.Button(root, text="Browse", command=select_image)
image_button.grid(row=1, column=1, padx=5, pady=5)
image_label = tk.Label(root)
image_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

# Start Detection Button
start_button = tk.Button(root, text="Start Detection", command=start_detection, bg="green", fg="white")
start_button.grid(row=3, column=0, columnspan=3, padx=5, pady=10)

# Progress Bar
progress_bar = ttk.Progressbar(root, orient='horizontal', length=400, mode='determinate')
progress_bar.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

# Result Box
result_box = scrolledtext.ScrolledText(root, width=50, height=15, state=tk.DISABLED)
result_box.grid(row=5, column=0, columnspan=3, padx=5, pady=5)

# Detected Faces Display
tk.Label(root, text="Detected Faces:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
face_frame = tk.Frame(root)
face_frame.grid(row=7, column=0, columnspan=3, padx=5, pady=5)

# Run the GUI
root.mainloop();
