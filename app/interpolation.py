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
import math
from app.audio_utils import *
from app.video_utils import *
from torch import amp

target_height = 720
target_width = 1280
modelDir = r"rife_1\train_log"


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
    device_type = device.type  # 'cuda' hoặc 'cpu'
    with amp.autocast(device_type=device_type):
        model.inference(dummy_img, dummy_img)
print("Model warmed up.")


def interpolate_frames(tensor0, tensor1, frames, original_height, original_width, transition_frame_delay, h, w, audio0=None, audio1=None, window_name="Video Loop"):
    # Record start time for transition processing
    start_time = time.time()
    
    with torch.no_grad():
        if tensor0.shape[2:] != tensor1.shape[2:]:
            raise ValueError(f"Tensor dimension mismatch: tensor0 {tensor0.shape}, tensor1 {tensor1.shape}")
        
        # Initialize the list with start and end frames
        img_list = [tensor0, tensor1]
        total_frames = frames  # Total desired frames including start and end
        if total_frames < 2:
            raise ValueError("Number of frames must be at least 2")
        
        # Calculate minimum interpolation steps needed
        steps = max(0, int(math.ceil(math.log2(total_frames - 1))))  # Approximate log2 for iterations
        with autocast():
            for _ in range(steps):
                tmp = []
                current_num_frames = len(img_list)
                for i in range(current_num_frames - 1):
                    tmp.append(img_list[i])
                    if len(tmp) < total_frames - 1:  # Only add midpoints if needed
                        mid = model.inference(img_list[i], img_list[i + 1])
                        tmp.append(mid)
                tmp.append(img_list[-1])
                img_list = tmp
            # Trim or pad to exactly total_frames
            if len(img_list) > total_frames:
                step = len(img_list) / total_frames
                img_list = [img_list[int(i * step)] for i in range(total_frames)]
            elif len(img_list) < total_frames:
                # Pad with linear interpolation if needed (rare case)
                last_frame = img_list[-1]
                img_list.extend([last_frame] * (total_frames - len(img_list)))
        
        # Clear audio queue
        while not audio_queue.empty():
            audio_queue.get()
        
        # Audio processing (optimized)
        transition_duration_ms = total_frames * transition_frame_delay
        transition_duration = transition_duration_ms / 1000
        if audio0 is not None and audio1 is not None:
            crossfade_samples = int(transition_duration * sample_rate)
            fade_samples = min(int(0.1 * sample_rate), len(audio0), len(audio1))  # Reduced fade duration
            audio_transition = np.zeros(crossfade_samples, dtype=np.float32)
            t = np.linspace(0, 1, fade_samples, dtype=np.float32)
            # Minimize padding by checking lengths first
            if len(audio0) < fade_samples:
                audio0 = np.pad(audio0, (0, fade_samples - len(audio0)), mode='constant')
            if len(audio1) < fade_samples:
                audio1 = np.pad(audio1, (0, fade_samples - len(audio1)), mode='constant')
            audio_transition[:fade_samples] = audio0[-fade_samples:] * (1 - t) + audio1[:fade_samples] * t
            if crossfade_samples > fade_samples:
                audio_transition[fade_samples:crossfade_samples - fade_samples] = audio1[fade_samples:crossfade_samples - fade_samples]
                audio_transition[crossfade_samples - fade_samples:] = audio1[crossfade_samples - fade_samples:] * (1 - t) + audio0[:fade_samples] * t
            audio_queue.put(audio_transition)
        
        # Batch convert frames to numpy (reduce CPU-GPU transfers)
        frames_np = [(img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w] for img in img_list]
        
        # Display frames
        for frame in frames_np:
            upscaled_frame = upscale_frame(frame, original_height, original_width)
            display_frame = upscaled_frame.copy()
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(int(transition_frame_delay)) & 0xFF
            if key == 27:  # ESC key
                cv2.destroyAllWindows()
                sa.stop_all()
                exit()
        
        # Play audio
        while not audio_queue.empty():
            audio_transition = audio_queue.get()
            threading.Thread(target=play_audio, args=(audio_transition,), daemon=True).start()
            time.sleep(transition_duration + 0.01)
        
        img_list = None  # Clear list to free memory
    
    # Calculate and print total processing time
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Transition processing time at 07:49 PM +07, Sunday, August 03, 2025: {processing_time:.2f} seconds")

# Function to generate folder transitions
def generate_folder_transitions(folder_video_path, current_frame, video1_first_frame, frames, result_queue, target_height, original_height, original_width, transition_frame_delay):
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
