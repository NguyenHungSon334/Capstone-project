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

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_default_tensor_type(torch.cuda.HalfTensor)

# Hardcoded input parameters
root_video_paths = [
    r"D:\Python Test\Memory\WebMemory\videos\root\video1.mp4",
    r"D:\Python Test\Memory\WebMemory\videos\root\video2.mp4"
]
videos_base_dir = r"D:\Python Test\Memory\WebMemory\videos"
exp = 3
modelDir = r"rife\train_log"
target_fps = 60
target_height = 720
target_width = 1280
sample_rate = 44100

# Verify root videos exist
for video_path in root_video_paths:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Root video not found: {video_path}")

# Function to extract first and last frames from a video
def get_video_frames(video_path, target_height=720, target_width=1280):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps != target_fps:
        print(f"Warning: {video_path} has FPS {fps}, expected {target_fps}. Using {target_fps} FPS.")
    
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Cannot read first frame from: {video_path}")
    first_frame = downscale_frame(first_frame, target_height, target_width)
    frames.append(first_frame)
    
    if frame_count > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, last_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError(f"Cannot read last frame from: {video_path}")
        last_frame = downscale_frame(last_frame, target_height, target_width)
        frames.append(last_frame)
    
    cap.release()
    return frames, fps, frame_count, width, height

# Function to downscale a frame
def downscale_frame(frame, target_height=720, target_width=1280):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height
    if abs(aspect_ratio - target_aspect_ratio) > 0.01:
        if aspect_ratio > target_aspect_ratio:
            new_width = int(height * target_aspect_ratio)
            start_x = (width - new_width) // 2
            frame = frame[:, start_x:start_x + new_width]
        else:
            new_width = int(height * target_aspect_ratio)
            pad_width = (new_width - width) // 2
            frame = cv2.copyMakeBorder(frame, 0, 0, pad_width, pad_width, cv2.BORDER_REFLECT)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

# Function to upscale a frame
def upscale_frame(frame, original_height, original_width):
    return cv2.resize(frame, (original_width, original_height), interpolation=cv2.INTER_CUBIC)

# Function to get video from folder
def get_video_from_folder(folder_name):
    folder_path = os.path.join(videos_base_dir, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return None
    for file in os.listdir(folder_path):
        if file.endswith((".mp4", ".avi", ".mov")):
            return os.path.join(folder_path, file)
    print(f"No valid video found in folder: {folder_path}")
    return None

# Function to preprocess frame to tensor
def preprocess_frame(frame, target_height=720, target_width=1280):
    frame = downscale_frame(frame, target_height, target_width)
    with torch.no_grad():
        tensor = torch.tensor(frame.transpose(2, 0, 1)).to(device) / 255.0
        n, h, w = tensor.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        tensor = F.pad(tensor.unsqueeze(0), padding)
    return tensor, h, w

# Load RIFE model
from rife_1.train_log.RIFE_HDv3 import Model
model = Model()
model.load_model(modelDir, -1)
print("Loaded v3.x HD model.")
model.eval()
model.device()

# Perform dummy inference to warm up the model
with torch.no_grad():
    dummy_img = torch.zeros(1, 3, 720, 1280, device=device)
    with autocast():
        model.inference(dummy_img, dummy_img)
print("Model warmed up.")

# Function to extract and process audio for transitions
def process_audio(video_path, start_time, duration=0.2):
    try:
        audio_data, sr = librosa.load(video_path, sr=sample_rate, offset=start_time, duration=duration)
        if len(audio_data) == 0:
            print(f"No audio found in video: {video_path}")
            return None
        audio_data = apply_fade(audio_data, sr, int(0.2 * sr))
        return audio_data
    except Exception as e:
        print(f"Error processing audio for {video_path}: {str(e)}")
        return None

# Function to apply fade-in/fade-out to audio data
def apply_fade(audio_data, sr, fade_samples):
    if len(audio_data) < fade_samples:
        fade_samples = len(audio_data) // 2
    fade_in = np.linspace(0, 1, fade_samples)
    audio_data[:fade_samples] *= fade_in
    fade_out = np.linspace(1, 0, fade_samples)
    audio_data[-fade_samples:] *= fade_out
    return audio_data

# Function to play audio using simpleaudio
audio_queue = queue.Queue()
audio_lock = threading.Lock()

def play_audio(audio_data, sr=sample_rate):
    if audio_data is None:
        return
    try:
        with audio_lock:
            sa.stop_all()
            audio_data = (audio_data * 32767).astype(np.int16)
            if audio_data.ndim == 1:
                audio_data = np.stack([audio_data, audio_data], axis=1)
            play_obj = sa.play_buffer(audio_data, num_channels=2, bytes_per_sample=2, sample_rate=sr)
            play_obj.wait_done()
    except Exception as e:
        print(f"Error playing audio: {str(e)}")

# Function to interpolate frames and audio
def interpolate_frames(tensor0, tensor1, exp, original_height, original_width, transition_frame_delay, h, w, audio0=None, audio1=None, window_name="Video Loop"):
    with torch.no_grad():
        if tensor0.shape[2:] != tensor1.shape[2:]:
            raise ValueError(f"Tensor dimension mismatch: tensor0 {tensor0.shape}, tensor1 {tensor1.shape}")
        img_list = [tensor0, tensor1]
        with autocast():
            for i in range(exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    mid = model.inference(img_list[j], img_list[j + 1])
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(img_list[-1])
                img_list = tmp
        
        while not audio_queue.empty():
            audio_queue.get()
        
        total_frames = 2**exp + 1
        transition_duration_ms = total_frames * transition_frame_delay
        transition_duration = transition_duration_ms / 1000
        if audio0 is not None and audio1 is not None:
            crossfade_samples = int(transition_duration * sample_rate)
            fade_samples = min(int(0.2 * sample_rate), len(audio0), len(audio1))
            audio_transition = np.zeros(crossfade_samples)
            t = np.linspace(0, 1, fade_samples)
            audio0 = np.pad(audio0, (0, max(0, crossfade_samples - len(audio0))), mode='constant')
            audio1 = np.pad(audio1, (0, max(0, crossfade_samples - len(audio1))), mode='constant')
            audio_transition[:fade_samples] = audio0[-fade_samples:] * (1 - t) + audio1[:fade_samples] * t
            if crossfade_samples > fade_samples:
                audio_transition[fade_samples:crossfade_samples - fade_samples] = audio1[fade_samples:crossfade_samples - fade_samples]
                audio_transition[crossfade_samples - fade_samples:] = audio1[crossfade_samples - fade_samples:] * (1 - t) + audio0[:fade_samples] * t
            audio_queue.put(audio_transition)
        
        for img in img_list:
            frame = (img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            upscaled_frame = upscale_frame(frame, original_height, original_width)
            display_frame = upscaled_frame.copy()
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(int(transition_frame_delay)) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                sa.stop_all()
                exit()
        
        while not audio_queue.empty():
            audio_transition = audio_queue.get()
            threading.Thread(target=play_audio, args=(audio_transition,), daemon=True).start()
            time.sleep(transition_duration + 0.01)
        
        img_list = None
        torch.cuda.empty_cache()

# Function to generate folder transitions
def generate_folder_transitions(folder_video_path, current_frame, video1_first_frame, exp, result_queue, target_height, original_height, original_width, transition_frame_delay):
    try:
        folder_frames, _, _, folder_width, folder_height = get_video_frames(folder_video_path, target_height, target_width)
        if folder_width != original_width or folder_height != original_height:
            result_queue.put((None, None, None, None, None, None, f"Folder video resolution mismatch: ({folder_width}x{folder_height}) vs Root ({original_width}x{original_height})"))
            return
        folder_frames_downscaled = [downscale_frame(frame, target_height, target_width) for frame in folder_frames]
        folder_first_tensor, h, w = preprocess_frame(folder_frames_downscaled[0], target_height, target_width)
        folder_last_tensor, _, _ = preprocess_frame(folder_frames_downscaled[1], target_height, target_width)
        folder_audio_first = process_audio(folder_video_path, 0, 0.2)
        folder_audio_last = process_audio(folder_video_path, -0.2, 0.2)
        result_queue.put((folder_first_tensor, folder_last_tensor, h, w, folder_audio_first, folder_audio_last, None))
    except Exception as e:
        result_queue.put((None, None, None, None, None, None, str(e)))

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
transition_frame_delay = (0.2 * 1000) / (2**exp + 1)

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
                                        args=(current_folder_video, transition_start_frame, video1_frames[0], exp, transition_queue, target_height, original_height, original_width, transition_frame_delay), 
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

        interpolate_frames(video1_last_tensor, video2_first_tensor, exp, original_height, original_width, transition_frame_delay, h1, w1, video1_audio_last, video2_audio_first)

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
                                        args=(current_folder_video, transition_start_frame, video1_frames[0], exp, transition_queue, target_height, original_height, original_width, transition_frame_delay), 
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

        interpolate_frames(video2_last_tensor, video1_first_tensor, exp, original_height, original_width, transition_frame_delay, h1, w1, video2_audio_last, video1_audio_first)

    elif current_mode == "folder" and current_folder_video:
        sa.stop_all()
        while not audio_queue.empty():
            audio_queue.get()
        
        transition_start_tensor, _, _ = preprocess_frame(downscale_frame(transition_start_frame or current_frame, target_height, target_width), target_height, target_width)
        
        transition_start_audio = process_audio(root_video_paths[0], -0.2, 0.2)
        interpolate_frames(transition_start_tensor, folder_transition_tensors[0], exp, original_height, original_width, transition_frame_delay, folder_transition_dims[0], folder_transition_dims[1], transition_start_audio, folder_transition_audio[0])

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

        interpolate_frames(folder_transition_tensors[1], video1_first_tensor, exp, original_height, original_width, transition_frame_delay, folder_transition_dims[0], folder_transition_dims[1], folder_transition_audio[1], video1_audio_first)

        current_mode = "root"
        current_folder_video = None
        folder_transition_tensors = None
        folder_transition_dims = None
        folder_transition_audio = None