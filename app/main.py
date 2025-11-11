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
import pygame # SỬ DỤNG PYGAME
import time
import sys
import speech_recognition as sr
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont # Cần cho font

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from audio_utils import *
from video_utils import *
from interpolation import *
from suggestion import * # SỬ DỤNG suggestion.py

warnings.filterwarnings("ignore")

genai.configure(api_key="AIzaSyBT7G6EfLR45FLyoFVDC8ft2zbfhkx01Oo")

VIDEO_DIR = r"..\Capstone-project\videos"

# --- Tải Font (Sẽ được truyền vào suggestion.py) ---
FONT_NAME = "arial.ttf" 
try:
    font_path = os.path.join(os.path.dirname(__file__), FONT_NAME)
    font_regular = ImageFont.truetype(font_path, 24)
    font_small = ImageFont.truetype(font_path, 18) # Font nhỏ cho các mục hint
    font_title = ImageFont.truetype(font_path, 20) # Font tiêu đề hint
    font_button = ImageFont.truetype(font_path, 30) # Font cho nút 'G'
    print(f"Da tai font: {font_path}")
except IOError:
    print(f"LỖI: Không tìm thấy font: {font_path}. Vui lòng tải 'arial.ttf' vào thư mục 'app'.")
    font_regular = ImageFont.load_default()
    font_small = ImageFont.load_default()
    font_title = ImageFont.load_default()
    font_button = ImageFont.load_default()
# --- KẾT THÚC Tải Font ---

# --- Thêm hàm vẽ PIL (cho main.py và suggestion.py) ---
def draw_text_pil(img, text, position, font, color_bgr):
    try:
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0]) # BGR to RGB
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=font, fill=color_rgb)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR) # RGB to BGR
    except Exception as e:
        print(f"Loi ve van ban: {e}")
        return img
# --- Kết thúc hàm ---

root_video_paths = [
    r"..\Capstone-project\videos\root\video1.mp4",
    r"..\Capstone-project\videos\root\video2.mp4"
]
frames = 5
target_height = 720
target_width = 1280
target_fps = 60
sample_rate = 44100

for video_path in root_video_paths:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Root video not found: {video_path}")

video1_frames, video1_fps, video1_frame_count, video1_width, video1_height = get_video_frames(root_video_paths[0], target_height, target_width)
video2_frames, video2_fps, video2_frame_count, video2_width, video2_height = get_video_frames(root_video_paths[1], target_height, target_width)

if video1_width != video2_width or video1_height != video2_height:
    raise ValueError(f"Root video resolutions mismatch: Video 1 ({video1_width}x{video1_height}), Video 2 ({video2_width}x{video2_height})")

original_height, original_width = video1_frames[0].shape[:2]
video1_frames_downscaled = [downscale_frame(frame, target_height, target_width) for frame in video1_frames]
video2_frames_downscaled = [downscale_frame(frame, target_height, target_width) for frame in video2_frames]

video1_first_tensor, h1, w1 = preprocess_frame(video1_frames_downscaled[0], target_height, target_width)
video1_last_tensor, _, _ = preprocess_frame(video1_frames_downscaled[1], target_height, target_width)
video2_first_tensor, _, _ = preprocess_frame(video2_frames_downscaled[0], target_height, target_width)
video2_last_tensor, _, _ = preprocess_frame(video2_frames_downscaled[1], target_height, target_width)

video1_audio_first = process_audio(root_video_paths[0], 0, 0.2)
video1_audio_last = process_audio(root_video_paths[0], -0.2, 0.2)
video2_audio_first = process_audio(root_video_paths[1], 0, 0.2)
video2_audio_last = process_audio(root_video_paths[1], -0.2, 0.2)

video_frame_delay = 1000 / target_fps
transition_frame_delay = (0.2 * 1000) / frames

cv2.namedWindow("Video Loop", cv2.WINDOW_NORMAL)
current_mode = "root"
current_root_index = 0
current_folder_video = None
folder_transition_tensors = None
folder_transition_dims = None
folder_transition_audio = None
transition_queue = queue.Queue()
current_frame = None
transition_start_frame = None
waiting_for_transition = False

recognizer = sr.Recognizer()
recognizer.pause_threshold = 0.5

mic = sr.Microphone()
with mic as source:
    print("Dang dieu chinh mic...")
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print("✅ Microphone calibrated for ambient noise")

stop_listening = recognizer.listen_in_background(mic, speech_callback)
print("🎤 Background listening started - speak Vietnamese anytime!")

def get_current_mode():
    return current_mode

def get_waiting_for_transition():
    return waiting_for_transition

suggestion_handler = SuggestionHandler(
    target_height, 
    VIDEO_DIR, 
    folder_queue, 
    get_current_mode, 
    get_waiting_for_transition,
    font_title=font_title, 
    font_item=font_small,   
    font_button=font_button 
)
cv2.setMouseCallback("Video Loop", suggestion_handler.mouse_callback)

while True:
    if current_mode == "root":
        pygame.mixer.stop() 
        while not audio_queue.empty():
            audio_queue.get()
        
        current_root_path = root_video_paths[current_root_index]
        cap = cv2.VideoCapture(current_root_path)
        try:    
            audio_data, audio_sr = librosa.load(current_root_path, sr=sample_rate)
            threading.Thread(target=play_audio, args=(audio_data, audio_sr), daemon=True).start()
        except Exception as e:
            print(f"Error loading audio for {current_root_path}: {str(e)}")
        
        while cap.isOpened():
            if waiting_for_transition:
                display_frame = transition_start_frame.copy()
                display_frame = draw_text_pil(display_frame, "Đang chuẩn bị chuyển cảnh...", (10, 90), 
                                              font_regular, (255, 255, 255))
                
                display_frame = suggestion_handler.draw_circular_button(display_frame)
                if suggestion_handler.show_suggestions:
                    display_frame = suggestion_handler.draw_suggestion_overlay(display_frame)
                
                cv2.imshow("Video Loop", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    waiting_for_transition = False
                    transition_queue.queue.clear()
                    continue
                if not transition_queue.empty():
                    folder_first_tensor, folder_last_tensor, h, w, folder_audio_first, folder_audio_last, error = transition_queue.get()
                    if error:
                        print(f"Transition error: {error}")
                        current_folder_video = None
                        waiting_for_transition = False
                        transition_start_frame = None # Dòng này OK
                    else:
                        folder_transition_tensors = (folder_first_tensor, folder_last_tensor)
                        folder_transition_dims = (h, w)
                        folder_transition_audio = (folder_audio_first, folder_audio_last)
                        current_mode = "folder"
                        waiting_for_transition = False
                        # transition_start_frame = None # <<< DÒNG NÀY LÀ LỖI, ĐÃ XÓA
                        suggestion_handler.show_suggestions = False 
                    break
                continue

            ret, frame = cap.read()
            if not ret:
                break
            current_frame = frame
            display_frame = frame.copy()
            display_frame = draw_text_pil(display_frame, "Nói tên thư mục (hoặc nhấn nút 'G')", (10, 30), 
                                          font_regular, (255, 255, 255))
            
            display_frame = suggestion_handler.draw_circular_button(display_frame)
            if suggestion_handler.show_suggestions:
                display_frame = suggestion_handler.draw_suggestion_overlay(display_frame)
            
            cv2.imshow("Video Loop", display_frame)
            key = cv2.waitKey(int(video_frame_delay)) & 0xFF
            if key == 27:
                cap.release()
                pygame.mixer.stop() 
                pygame.quit()
                stop_listening(wait_for_stop=False)
                cv2.destroyAllWindows()
                sys.exit()

            if not folder_queue.empty() and not waiting_for_transition:
                folder_name = folder_queue.get()
                if folder_name and current_frame is not None:
                    current_folder_video = get_video_from_folder(folder_name)
                    if current_folder_video:
                        pygame.mixer.stop() 
                        transition_start_frame = current_frame.copy() # Lưu frame tại đây
                        waiting_for_transition = True
                        threading.Thread(target=generate_folder_transitions, 
                                        args=(current_folder_video, transition_start_frame, video1_frames[0], frames, transition_queue, target_height, target_width, original_height, original_width, transition_frame_delay), 
                                        daemon=True).start()
                    else:
                        print(f"Folder '{folder_name}' not found!")
                else:
                    print("Invalid folder name or no current frame")
            
            if current_mode != "root":
                break
        cap.release()

        if current_mode != "root":
            continue

        next_index = (current_root_index + 1) % len(root_video_paths)
        
        last_frame_of_video = current_frame.copy() if (ret is False and current_frame is not None) else video1_frames[0].copy()
        last_frame_of_video = draw_text_pil(last_frame_of_video, "Đang tạo chuyển cảnh AI (root)...", (10, 90), 
                                            font_regular, (255, 255, 0))
        cv2.imshow("Video Loop", last_frame_of_video)
        cv2.waitKey(1) 

        if current_root_index == 0:
            interpolate_frames(video1_last_tensor, video2_first_tensor, frames, original_height, original_width, transition_frame_delay, h1, w1, video1_audio_last, video2_audio_first)
        else:
            interpolate_frames(video2_last_tensor, video1_first_tensor, frames, original_height, original_width, transition_frame_delay, h1, w1, video2_audio_last, video1_audio_first)
        current_root_index = next_index

    elif current_mode == "folder" and current_folder_video:
        pygame.mixer.stop() 
        while not audio_queue.empty():
            audio_queue.get()
        
        # SỬA LỖI VALUEERROR (đã sửa ở lần trước)
        transition_start_tensor, _, _ = preprocess_frame(downscale_frame(transition_start_frame, target_height, target_width))
        
        transition_start_audio = process_audio(root_video_paths[current_root_index], -0.2, 0.2)
        
        display_message_frame = transition_start_frame.copy()
        display_message_frame = draw_text_pil(display_message_frame, "Đang tạo chuyển cảnh AI...", (10, 90), 
                                              font_regular, (255, 255, 0))
        cv2.imshow("Video Loop", display_message_frame)
        cv2.waitKey(1)

        interpolate_frames(transition_start_tensor, folder_transition_tensors[0], frames, original_height, original_width, transition_frame_delay, folder_transition_dims[0], folder_transition_dims[1], transition_start_audio, folder_transition_audio[0])

        cap = cv2.VideoCapture(current_folder_video)
        try:
            audio_data, audio_sr = librosa.load(current_folder_video, sr=sample_rate)
            threading.Thread(target=play_audio, args=(audio_data, audio_sr), daemon=True).start()
        except Exception as e:
            print(f"Error loading audio for {current_folder_video}: {str(e)}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            display_frame = frame.copy()
            display_frame = draw_text_pil(display_frame, "Đang phát video thư mục (Nhấn ESC để về root)", (10, 30), 
                                          font_regular, (255, 255, 255))
            
            cv2.imshow("Video Loop", display_frame)
            if cv2.waitKey(int(video_frame_delay)) & 0xFF == 27:
                break 
        cap.release()

        last_frame_of_folder = frame.copy() if (ret is False and frame is not None) else video1_frames[0].copy()
        last_frame_of_folder = draw_text_pil(last_frame_of_folder, "Đang tạo chuyển cảnh AI (về root)...", (10, 90), 
                                             font_regular, (255, 255, 0))
        cv2.imshow("Video Loop", last_frame_of_folder)
        cv2.waitKey(1)

        interpolate_frames(folder_transition_tensors[1], video1_first_tensor, frames, original_height, original_width, transition_frame_delay, folder_transition_dims[0], folder_transition_dims[1], folder_transition_audio[1], video1_audio_first)

        current_mode = "root"
        current_folder_video = None
        folder_transition_tensors = None
        folder_transition_dims = None
        folder_transition_audio = None