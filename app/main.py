import os
import cv2
import torch
import numpy as np
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

VIDEO_DIR = r".\videos"

# --- Tải Font (Sẽ được truyền vào suggestion.py) ---
FONT_NAME = "arial.ttf" 
try:
    font_path = os.path.join(os.path.dirname(__file__), FONT_NAME)
    font_regular = ImageFont.truetype(font_path, 24)
    font_small = ImageFont.truetype(font_path, 18) # Font nhỏ cho các mục hint
    font_title = ImageFont.truetype(font_path, 20) # Font tiêu đề hint
    font_button = ImageFont.truetype(font_path, 30) # Font cho nút 'G'
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
    os.path.join(VIDEO_DIR, 'root', 'video1.mp4'),
    os.path.join(VIDEO_DIR, 'root', 'video2.mp4')
]
frames = 5
target_height = 720
target_width = 1280
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

# Thêm device GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

video1_first_tensor, h1, w1 = preprocess_frame(video1_frames_downscaled[0], target_height, target_width)
video1_first_tensor = video1_first_tensor.to(device)  # Chuyển sang GPU
video1_last_tensor, _, _ = preprocess_frame(video1_frames_downscaled[1], target_height, target_width)
video1_last_tensor = video1_last_tensor.to(device)
video2_first_tensor, _, _ = preprocess_frame(video2_frames_downscaled[0], target_height, target_width)
video2_first_tensor = video2_first_tensor.to(device)
video2_last_tensor, _, _ = preprocess_frame(video2_frames_downscaled[1], target_height, target_width)
video2_last_tensor = video2_last_tensor.to(device)

video1_audio_first = process_audio(root_video_paths[0], 0, 0.2)
video1_audio_last = process_audio(root_video_paths[0], -0.2, 0.2)
video2_audio_first = process_audio(root_video_paths[1], 0, 0.2)
video2_audio_last = process_audio(root_video_paths[1], -0.2, 0.2)

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
    recognizer.adjust_for_ambient_noise(source, duration=1)

stop_listening = None  # Khởi tạo stop_listening

def start_listening():
    global stop_listening
    stop_listening = recognizer.listen_in_background(mic, speech_callback)
    print("Listening started in background...")

def stop_listening_func():
    global stop_listening
    if stop_listening:
        stop_listening(wait_for_stop=False)
        print("🎤 Background listening stopped")
        stop_listening = None

# Bắt đầu listening ở root
start_listening()

def get_current_mode():
    return current_mode

def get_waiting_for_transition():
    return waiting_for_transition

suggestion_handler = SuggestionHandler(
    video_dir=VIDEO_DIR, 
    folder_queue=folder_queue, 
    get_current_mode_func=get_current_mode, 
    get_waiting_for_transition_func=get_waiting_for_transition,
    font_title=font_title, 
    font_item=font_small,   
    font_button=font_button,
    font_regular=font_regular,
    target_height=target_height,
    target_width=target_width
)

cv2.setMouseCallback("Video Loop", suggestion_handler.mouse_callback)

audio_start_time = None  # Global để sync audio-video

while True:
    if current_mode == "root":
        # Bật listening nếu chưa
        if not stop_listening:
            start_listening()
        
        pygame.mixer.stop() 
        while not audio_queue.empty():
            audio_queue.get()
        
        current_root_path = root_video_paths[current_root_index]
        cap = cv2.VideoCapture(current_root_path)
        
        # Lấy FPS gốc của video để giữ tốc độ original
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps <= 0:
            actual_fps = 30  # Default nếu không detect được
        video_frame_delay = 1.0 / actual_fps  # Delay in seconds
        
        try:    
            # Load audio với sample_rate fixed để resample nếu cần, như ban đầu
            audio_data, _ = librosa.load(current_root_path, sr=sample_rate)
            
            # Hiển thị frame đầu tiên trước khi start audio
            ret, frame = cap.read()
            if ret:
                current_frame = frame
                display_frame = frame.copy()
                display_frame = draw_text_pil(display_frame, "Nói tên thư mục (hoặc nhấn nút 'G')", (10, 30), 
                                              font_regular, (255, 255, 255))
                display_frame = suggestion_handler.draw_circular_buttons(display_frame)
                if suggestion_handler.show_suggestions:
                    display_frame = suggestion_handler.draw_suggestion_overlay(display_frame)
                if suggestion_handler.show_input_box:
                    display_frame = suggestion_handler.draw_input_overlay(display_frame)
                cv2.imshow("Video Loop", display_frame)
                cv2.waitKey(1)
                
                # Bây giờ start audio với sample_rate
                audio_thread = threading.Thread(target=play_audio, args=(audio_data, sample_rate), daemon=True)
                audio_thread.start()
                audio_start_time = time.time()  # Lưu thời gian start audio
                
                # Reset timer cho video
                video_start_time = time.time()
        except Exception as e:
            print(f"Error loading audio for {current_root_path}: {str(e)}")
        
        while cap.isOpened():
            frame_start_time = time.time()  # Theo dõi thời gian cho mỗi frame
            
            if waiting_for_transition:
                display_frame = transition_start_frame.copy()
                display_frame = suggestion_handler.draw_circular_buttons(display_frame)
                if suggestion_handler.show_suggestions:
                    display_frame = suggestion_handler.draw_suggestion_overlay(display_frame)
                if suggestion_handler.show_input_box:
                    display_frame = suggestion_handler.draw_input_overlay(display_frame)
                
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
                        transition_start_frame = None
                    else:
                        folder_transition_tensors = (folder_first_tensor.to(device), folder_last_tensor.to(device))  # Chuyển sang GPU
                        folder_transition_dims = (h, w)
                        folder_transition_audio = (folder_audio_first, folder_audio_last)
                        current_mode = "folder"
                        waiting_for_transition = False
                        suggestion_handler.show_suggestions = False 
                        suggestion_handler.show_input_box = False
                    break
                continue

            ret, frame = cap.read()
            if not ret:
                break
            current_frame = frame
            display_frame = frame.copy()
            display_frame = draw_text_pil(display_frame, "Nói tên thư mục (hoặc nhấn nút 'G')", (10, 30), 
                                          font_regular, (255, 255, 255))
            
            display_frame = suggestion_handler.draw_circular_buttons(display_frame)
            if suggestion_handler.show_suggestions:
                display_frame = suggestion_handler.draw_suggestion_overlay(display_frame)
            if suggestion_handler.show_input_box:
                display_frame = suggestion_handler.draw_input_overlay(display_frame)
            
            # Sync video với audio
            if audio_start_time is not None:
                current_audio_pos = time.time() - audio_start_time  # seconds
                expected_frame_pos = current_audio_pos * actual_fps
                current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                
                if current_frame_pos < expected_frame_pos - 1:  # Skip frames nếu video chậm
                    cap.set(cv2.CAP_PROP_POS_FRAMES, expected_frame_pos)
                elif current_frame_pos > expected_frame_pos + 1:  # Delay nếu video nhanh
                    time.sleep((current_frame_pos - expected_frame_pos) / actual_fps)
            
            cv2.imshow("Video Loop", display_frame)
            key = cv2.waitKey(1)  # Không cần & 0xFF ở đây, sẽ xử lý sau
            if key == 27:  # ESC to exit program
                cap.release()
                pygame.mixer.stop() 
                pygame.quit()
                stop_listening_func()
                cv2.destroyAllWindows()
                sys.exit()

            if suggestion_handler.show_input_box:
                if key == 27:  # ESC to cancel input
                    suggestion_handler.show_input_box = False
                    suggestion_handler.input_text = ""
                elif key == 13:  # Enter to submit
                    folder_name = suggestion_handler.input_text.strip()
                    if folder_name:
                        available_folders = [d for d in os.listdir(VIDEO_DIR) if os.path.isdir(os.path.join(VIDEO_DIR, d)) and d != "root"]
                        best_match = find_best_folder_match(folder_name, available_folders)
                        if best_match != "None":
                            print(f"Best folder match after text input: {best_match}")
                            folder_queue.put(best_match)
                        else:
                            print(f"No matching folder found for text input: {folder_name}")
                    suggestion_handler.show_input_box = False
                    suggestion_handler.input_text = ""
                elif key == 8:  # Backspace
                    if suggestion_handler.input_text:
                        suggestion_handler.input_text = suggestion_handler.input_text[:-1]
                elif 32 <= key <= 126:  # Printable ASCII characters
                    suggestion_handler.input_text += chr(key)
                # Hỗ trợ tiếng Việt: OpenCV waitKey chỉ hỗ trợ ASCII, tiếng Việt cần xử lý thêm (có thể dùng utf-8 nhưng phức tạp, giả định input ASCII cho đơn giản)
                continue  # Skip other processing while inputting

            if not folder_queue.empty() and not waiting_for_transition:
                folder_name = folder_queue.get()
                if folder_name and current_frame is not None:
                    available_folders = [d for d in os.listdir(VIDEO_DIR) if os.path.isdir(os.path.join(VIDEO_DIR, d)) and d != "root"]
                    
                    best_match = find_best_folder_match(folder_name, available_folders)
                    
                    if best_match != "None":
                        folder_name = best_match
                        print(f"Best folder match after selection: {folder_name}")
                    else:
                        print(f"No matching folder found for: {folder_name}")
                        continue
                    
                    current_folder_video = get_video_from_folder(folder_name)
                    if current_folder_video:
                        pygame.mixer.stop() 
                        transition_start_frame = current_frame.copy()
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
            
            # Điều chỉnh timing để giữ FPS gốc, tránh tụt
            frame_end_time = time.time()
            frame_processing_time = frame_end_time - frame_start_time
            sleep_time = video_frame_delay - frame_processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        cap.release()

        if current_mode != "root":
            continue

        next_index = (current_root_index + 1) % len(root_video_paths)
        
        last_frame_of_video = current_frame.copy() if (current_frame is not None) else video1_frames[0].copy()
        cv2.imshow("Video Loop", last_frame_of_video)
        cv2.waitKey(1) 

        if current_root_index == 0:
            interpolate_frames(video1_last_tensor, video2_first_tensor, frames, original_height, original_width, transition_frame_delay, h1, w1, video1_audio_last, video2_audio_first)
        else:
            interpolate_frames(video2_last_tensor, video1_first_tensor, frames, original_height, original_width, transition_frame_delay, h1, w1, video2_audio_last, video1_audio_first)
        current_root_index = next_index

        # Xóa cache GPU sau interpolation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elif current_mode == "folder" and current_folder_video:
        # Stop listening khi vào folder
        stop_listening_func()
        
        # Disable suggestion handler (không vẽ nút, không mouse callback ở folder)
        suggestion_handler.show_suggestions = False
        suggestion_handler.show_input_box = False
        
        pygame.mixer.stop() 
        while not audio_queue.empty():
            audio_queue.get()
        
        transition_start_tensor, _, _ = preprocess_frame(downscale_frame(transition_start_frame, target_height, target_width))
        transition_start_tensor = transition_start_tensor.to(device)  # Chuyển sang GPU
        
        transition_start_audio = process_audio(root_video_paths[current_root_index], -0.2, 0.2)
        
        display_message_frame = transition_start_frame.copy()
        cv2.imshow("Video Loop", display_message_frame)
        cv2.waitKey(1)

        interpolate_frames(transition_start_tensor, folder_transition_tensors[0], frames, original_height, original_width, transition_frame_delay, folder_transition_dims[0], folder_transition_dims[1], transition_start_audio, folder_transition_audio[0])

        # Xóa cache GPU sau interpolation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cap = cv2.VideoCapture(current_folder_video)
        
        # Lấy FPS gốc cho folder video
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps <= 0:
            actual_fps = 30
        video_frame_delay = 1.0 / actual_fps  # seconds
        
        try:
            # Load audio với sample_rate fixed
            audio_data, _ = librosa.load(current_folder_video, sr=sample_rate)
            
            # Sync tương tự cho folder video
            ret, frame = cap.read()
            if ret:
                display_frame = frame.copy()
                cv2.imshow("Video Loop", display_frame)
                cv2.waitKey(1)
                
                audio_thread = threading.Thread(target=play_audio, args=(audio_data, sample_rate), daemon=True)
                audio_thread.start()
                audio_start_time = time.time()
        except Exception as e:
            print(f"Error loading audio for {current_folder_video}: {str(e)}")
        
        while cap.isOpened():
            frame_start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = frame  # Lưu last frame thực tế
            display_frame = frame.copy()
            
            # Sync video với audio
            if audio_start_time is not None:
                current_audio_pos = time.time() - audio_start_time
                expected_frame_pos = current_audio_pos * actual_fps
                current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                
                if current_frame_pos < expected_frame_pos - 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, expected_frame_pos)
                elif current_frame_pos > expected_frame_pos + 1:
                    time.sleep((current_frame_pos - expected_frame_pos) / actual_fps)
            
            cv2.imshow("Video Loop", display_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break 
            
            # Điều chỉnh timing
            frame_end_time = time.time()
            frame_processing_time = frame_end_time - frame_start_time
            sleep_time = video_frame_delay - frame_processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        cap.release()

        # Sử dụng current_frame làm last_frame_of_folder để tránh nhảy frame
        last_frame_of_folder = current_frame.copy() if current_frame is not None else video1_frames[0].copy()
        cv2.imshow("Video Loop", last_frame_of_folder)
        cv2.waitKey(1)

        # Transition mượt: Không có text overlay, chỉ frame cuối thực tế
        interpolate_frames(folder_transition_tensors[1], video1_first_tensor, frames, original_height, original_width, transition_frame_delay, folder_transition_dims[0], folder_transition_dims[1], folder_transition_audio[1], video1_audio_first)

        # Xóa cache GPU sau interpolation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clean variables và chuyển về root
        current_mode = "root"
        current_folder_video = None
        folder_transition_tensors = None
        folder_transition_dims = None
        folder_transition_audio = None
        transition_queue.queue.clear()  # Clean queue
        folder_queue.queue.clear()  # Clean folder queue
        audio_queue.queue.clear()  # Clean audio queue
        suggestion_handler.show_suggestions = False  # Reset suggestion
        suggestion_handler.show_input_box = False
        waiting_for_transition = False
        transition_start_frame = None
        current_frame = None