import os
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torch.cuda.amp import autocast
import warnings
import threading
import queue
import librosa
import simpleaudio as sa
import time
import sys

# Thêm đường dẫn cha của rife_1 vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.audio_utils import *
from app.video_utils import *
from app.interpolation import *

warnings.filterwarnings("ignore")

# Hardcoded input parameters
root_video_paths = [
    r"D:\Capstone-project\videos\root\video1.mp4",
    r"D:\Capstone-project\videos\root\video2.mp4"
]
frames = 10
target_height = 720
target_width = 1280
target_fps = 60  # Updated to 60 FPS
sample_rate = 44100  # Default sample rate

# Verify root videos exist
for video_path in root_video_paths:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Root video not found: {video_path}")

# Get frames and metadata from root videos
video1_frames, video1_fps, video1_frame_count, video1_width, video1_height = get_video_frames(root_video_paths[0], target_height, target_width)
video2_frames, video2_fps, video2_frame_count, video2_width, video2_height = get_video_frames(root_video_paths[1], target_height, target_width)

# Check resolution consistency
if video1_width != video2_width or video1_height != video2_height:
    raise ValueError(f"Root video resolutions mismatch: Video 1 ({video1_width}x{video1_height}), Video 2 ({video2_width}x{video2_height})")

# Downscale and preprocess root video frames
original_height, original_width = video1_frames[0].shape[:2]
video1_frames_downscaled = [downscale_frame(frame, target_height, target_width) for frame in video1_frames]
video2_frames_downscaled = [downscale_frame(frame, target_height, target_width) for frame in video2_frames]

# Preprocess root frames to tensors
video1_first_tensor, h1, w1 = preprocess_frame(video1_frames_downscaled[0], target_height, target_width)
video1_last_tensor, _, _ = preprocess_frame(video1_frames_downscaled[1], target_height, target_width)
video2_first_tensor, _, _ = preprocess_frame(video2_frames_downscaled[0], target_height, target_width)
video2_last_tensor, _, _ = preprocess_frame(video2_frames_downscaled[1], target_height, target_width)

# Extract audio for root videos
video1_audio_first = process_audio(root_video_paths[0], 0, 0.2)
video1_audio_last = process_audio(root_video_paths[0], -0.2, 0.2)
video2_audio_first = process_audio(root_video_paths[1], 0, 0.2)
video2_audio_last = process_audio(root_video_paths[1], -0.2, 0.2)

# Calculate frame delays
video_frame_delay = 1000 / target_fps
transition_frame_delay = (0.2 * 1000) / frames  # Adjusted for frames

# Real-time playback loop
cv2.namedWindow("Video Loop", cv2.WINDOW_NORMAL)
current_mode = "root"
current_folder_video = None
folder_transition_tensors = None
folder_transition_dims = None
folder_transition_audio = None
input_mode = False
folder_name = ""
transition_queue = queue.Queue()
current_frame = None
transition_start_frame = None
waiting_for_transition = False

while True:
    if current_mode == "root":
        sa.stop_all()
        while not audio_queue.empty():
            audio_queue.get()
        
        cap = cv2.VideoCapture(root_video_paths[0])
        try:
            audio_data, sr = librosa.load(root_video_paths[0], sr=sample_rate)
            threading.Thread(target=play_audio, args=(audio_data, sr), daemon=True).start()
        except Exception as e:
            print(f"Error loading audio for {root_video_paths[0]}: {str(e)}")
        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = frame
            display_frame = frame.copy()
            cv2.putText(display_frame, "Press 'f' to enter folder name", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if input_mode:
                cv2.putText(display_frame, f"Folder: {folder_name}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if waiting_for_transition and transition_start_frame is not None:
                display_frame = transition_start_frame.copy()
                cv2.putText(display_frame, "Preparing transition...", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Video Loop", display_frame)
            key = cv2.waitKey(int(video_frame_delay)) & 0xFF
            if key == 27:
                if input_mode:
                    input_mode = False
                    folder_name = ""
                elif waiting_for_transition:
                    waiting_for_transition = False
                    transition_start_frame = None
                    transition_queue.queue.clear()
                else:
                    cap.release()
                    sa.stop_all()
                    cv2.destroyAllWindows()
                    exit()
            elif key == ord('f') and not input_mode and not waiting_for_transition:
                input_mode = True
                folder_name = ""
            elif input_mode and key >= 32 and key <= 126:
                folder_name += chr(key)
            elif input_mode and key == 13 and current_frame is not None:
                input_mode = False
                if folder_name:
                    current_folder_video = get_video_from_folder(folder_name)
                    if current_folder_video:
                        sa.stop_all()
                        transition_start_frame = current_frame.copy()
                        waiting_for_transition = True
                        threading.Thread(target=generate_folder_transitions, 
                                        args=(current_folder_video, transition_start_frame, video1_frames[0], frames, transition_queue, target_height, original_height, original_width, transition_frame_delay), 
                                        daemon=True).start()
                folder_name = ""
            if waiting_for_transition and not transition_queue.empty():
                folder_first_tensor, folder_last_tensor, h, w, folder_audio_first, folder_audio_last, error = transition_queue.get()
                if error:
                    print(error)
                    current_folder_video = None
                    waiting_for_transition = False
                    transition_start_frame = None
                else:
                    folder_transition_tensors = (folder_first_tensor, folder_last_tensor)
                    folder_transition_dims = (h, w)
                    folder_transition_audio = (folder_audio_first, folder_audio_last)
                    current_mode = "folder"
                    waiting_for_transition = False
                    transition_start_frame = None
                break
            if current_mode != "root":
                break
        cap.release()

        if current_mode != "root":
            continue

        interpolate_frames(video1_last_tensor, video2_first_tensor, frames, original_height, original_width, transition_frame_delay, h1, w1, video1_audio_last, video2_audio_first)

        sa.stop_all()
        while not audio_queue.empty():
            audio_queue.get()
        
        cap = cv2.VideoCapture(root_video_paths[1])
        try:
            audio_data, sr = librosa.load(root_video_paths[1], sr=sample_rate)
            threading.Thread(target=play_audio, args=(audio_data, sr), daemon=True).start()
        except Exception as e:
            print(f"Error loading audio for {root_video_paths[1]}: {str(e)}")
        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = frame
            display_frame = frame.copy()
            cv2.putText(display_frame, "Press 'f' to enter folder name", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if input_mode:
                cv2.putText(display_frame, f"Folder: {folder_name}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if waiting_for_transition and transition_start_frame is not None:
                display_frame = transition_start_frame.copy()
                cv2.putText(display_frame, "Preparing transition...", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Video Loop", display_frame)
            key = cv2.waitKey(int(video_frame_delay)) & 0xFF
            if key == 27:
                if input_mode:
                    input_mode = False
                    folder_name = ""
                elif waiting_for_transition:
                    waiting_for_transition = False
                    transition_start_frame = None
                    transition_queue.queue.clear()
                else:
                    cap.release()
                    sa.stop_all()
                    cv2.destroyAllWindows()
                    exit()
            elif key == ord('f') and not input_mode and not waiting_for_transition:
                input_mode = True
                folder_name = ""
            elif input_mode and key >= 32 and key <= 126:
                folder_name += chr(key)
            elif input_mode and key == 13 and current_frame is not None:
                input_mode = False
                if folder_name:
                    current_folder_video = get_video_from_folder(folder_name)
                    if current_folder_video:
                        sa.stop_all()
                        transition_start_frame = current_frame.copy()
                        waiting_for_transition = True
                        threading.Thread(target=generate_folder_transitions, 
                                        args=(current_folder_video, transition_start_frame, video1_frames[0], frames, transition_queue, target_height, original_height, original_width, transition_frame_delay), 
                                        daemon=True).start()
                folder_name = ""
            if waiting_for_transition and not transition_queue.empty():
                folder_first_tensor, folder_last_tensor, h, w, folder_audio_first, folder_audio_last, error = transition_queue.get()
                if error:
                    print(error)
                    current_folder_video = None
                    waiting_for_transition = False
                    transition_start_frame = None
                else:
                    folder_transition_tensors = (folder_first_tensor, folder_last_tensor)
                    folder_transition_dims = (h, w)
                    folder_transition_audio = (folder_audio_first, folder_audio_last)
                    current_mode = "folder"
                    waiting_for_transition = False
                    transition_start_frame = None
                break
            if current_mode != "root":
                break
        cap.release()

        if current_mode != "root":
            continue

        interpolate_frames(video2_last_tensor, video1_first_tensor, frames, original_height, original_width, transition_frame_delay, h1, w1, video2_audio_last, video1_audio_first)

    elif current_mode == "folder" and current_folder_video:
        sa.stop_all()
        while not audio_queue.empty():
            audio_queue.get()
        
        transition_start_tensor, _, _ = preprocess_frame(downscale_frame(transition_start_frame or current_frame, target_height, target_width), target_height, target_width)
        
        transition_start_audio = process_audio(root_video_paths[0], -0.2, 0.2)
        interpolate_frames(transition_start_tensor, folder_transition_tensors[0], frames, original_height, original_width, transition_frame_delay, folder_transition_dims[0], folder_transition_dims[1], transition_start_audio, folder_transition_audio[0])

        cap = cv2.VideoCapture(current_folder_video)
        try:
            audio_data, sr = librosa.load(current_folder_video, sr=sample_rate)
            threading.Thread(target=play_audio, args=(audio_data, sr), daemon=True).start()
        except Exception as e:
            print(f"Error loading audio for {current_folder_video}: {str(e)}")
        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            display_frame = frame.copy()
            cv2.putText(display_frame, "Playing folder video", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Video Loop", display_frame)
            if cv2.waitKey(int(video_frame_delay)) & 0xFF == 27:
                cap.release()
                sa.stop_all()
                cv2.destroyAllWindows()
                exit()
        cap.release()

        interpolate_frames(folder_transition_tensors[1], video1_first_tensor, frames, original_height, original_width, transition_frame_delay, folder_transition_dims[0], folder_transition_dims[1], folder_transition_audio[1], video1_audio_first)

        current_mode = "root"
        current_folder_video = None
        folder_transition_tensors = None
        folder_transition_dims = None
        folder_transition_audio = None