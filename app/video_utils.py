import os
import cv2
import torch
from torch.nn import functional as F

target_fps = 60

videos_base_dir = r"D:\Python\Memory\WebMemory\videos"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_default_tensor_type(torch.cuda.HalfTensor)

    
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
