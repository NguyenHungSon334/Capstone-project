import cv2
import numpy as np
import os
import time
import speech_recognition as sr
import google.generativeai as genai
import tkinter as tk
from threading import Thread
import pygame
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import warnings
import ffmpeg
import uuid

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure Gemini API
genai.configure(api_key="AIzaSyBT7G6EfLR45FLyoFVDC8ft2zbfhkx01Oo")

# Device setup for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_default_tensor_type(torch.cuda.HalfTensor)  # Enable FP16 for CUDA

# Video directory and default videos
VIDEO_DIR = "videos"
DEFAULT_VIDEOS = [os.path.join(VIDEO_DIR, "root", "video1.mp4"), os.path.join(VIDEO_DIR, "root", "video2.mp4")]
MODEL_DIR = "rife_1/train_log"
TARGET_FPS = 30  # Can be changed to 60 to match first code

class VideoPlayer:
    def __init__(self):
        self.video_queue = DEFAULT_VIDEOS.copy()
        self.playing = True
        self.current_video = 0
        self.playing_response = False
        self.stop_all = False
        self.transitioning = False
        # self.exp = 4  # Interpolation factor for RIFE (17 frames per transition)
        self.exp = 2  # Use this for weaker GPU memory (3 frames per transition)

        # Load RIFE model
    
        from rife_1.train_log.RIFE_HDv3 import Model
        self.model = Model()
        self.model.load_model(MODEL_DIR, -1)
        print("Loaded v3.x HD model.")
    
        self.model.eval()
        self.model.device()

    def extract_audio(self, video_path):
        """Extract audio from video to a temporary file with a unique name"""
        if not os.path.exists(video_path):
            print(f"Error: Video file does not exist: {video_path}")
            return None
        audio_path = f"temp_audio_{uuid.uuid4().hex}.mp3"
        try:
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream, audio_path, format='mp3', acodec='mp3', loglevel="quiet")
            ffmpeg.run(stream, overwrite_output=True)
            return audio_path
        except AttributeError:
            print("Error: ffmpeg-python library not installed correctly. Please install with: pip install ffmpeg-python")
            return None
        except ffmpeg.Error as e:
            print(f"Cannot extract audio from video: {e}")
            return None
        except FileNotFoundError:
            print("Error: FFmpeg not installed or not found in PATH. Please install FFmpeg and add to PATH.")
            return None
        except Exception as e:
            print(f"Unexpected error while extracting audio: {e}")
            return None

    def get_video_frames(self, video_path, frame_position="last"):
        """Extract a specific frame from a video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        if fps != TARGET_FPS:
            print(f"Warning: {video_path} has FPS {fps}, expected {TARGET_FPS}. Using {TARGET_FPS} FPS.")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if frame_position == "first":
            ret, frame = cap.read()
        else:  # Default to last frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            ret, frame = cap.read()
        
        cap.release()
        if not ret:
            raise ValueError(f"Cannot read {frame_position} frame from: {video_path}")
        
        return frame, fps, frame_count, width, height

    def interpolate_frames(self, img0, img1):
        """Interpolate frames between two images using RIFE"""
        img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        
        n, c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)
        
        img_list = [img0, img1]
        with autocast():
            for i in range(self.exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    mid = self.model.inference(img_list[j], img_list[j + 1])
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(img1)
                img_list = tmp
        
        # Convert back to numpy, preserve original resolution
        result_frames = []
        for img in img_list:
            frame = (img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            result_frames.append(frame)
        return result_frames

    def play_video(self, video_path, transition_start_frame=None):
        if not os.path.exists(video_path):
            print(f"Error: Video file does not exist: {video_path}")
            return 0, 30
        pygame.mixer.init()
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        audio_path = self.extract_audio(video_path)
        audio_loaded = False
        if audio_path and os.path.exists(audio_path):
            try:
                pygame.mixer.music.load(audio_path)
                audio_loaded = True
            except pygame.error as e:
                print(f"Error loading audio into pygame: {e}")

        clock = pygame.time.Clock()
        current_frame = 0
        audio_started = False
        while cap.isOpened():
            if self.stop_all:
                break
            if self.playing_response and video_path != self.video_queue[self.current_video]:
                break
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            if not audio_started and audio_loaded:
                time.sleep(0)
                try:
                    pygame.mixer.music.play(start=0)
                    audio_started = True
                except pygame.error as e:
                    print(f"Error playing audio: {e}")

            cv2.imshow("Video Player", frame)
            elapsed_time = time.time() - start_time
            delay = max(1, int(1000 / fps - elapsed_time * 1000))
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC to exit
                self.stop_all = True
                self.playing = False
                break

            current_frame += 1
            if transition_start_frame and current_frame >= transition_start_frame:
                self.transitioning = True
                if audio_loaded:
                    pygame.mixer.music.stop()
                break

            clock.tick(fps)

        cap.release()
        if audio_loaded:
            pygame.mixer.music.stop()
        pygame.mixer.quit()
        if audio_path and os.path.exists(audio_path):
            for _ in range(3):
                try:
                    os.remove(audio_path)
                    break
                except PermissionError:
                    print(f"Cannot delete {audio_path}: File in use. Retrying...")
                    time.sleep(1)
                except Exception as e:
                    print(f"Cannot delete {audio_path}: {e}")
                    break

        return total_frames, fps

    def transition_videos(self, prev_video, next_video, start_frame):
        """Generate and play transition frames using RIFE"""
        # Get last frame of previous video and first frame of next video
        last_frame, _, _, width1, height1 = self.get_video_frames(prev_video, frame_position="last")
        first_frame, _, _, width2, height2 = self.get_video_frames(next_video, frame_position="first")

        # Check resolution consistency
        if width1 != width2 or height1 != height2:
            raise ValueError(f"Video resolutions mismatch: {prev_video} ({width1}x{height1}), {next_video} ({width2}x{height2})")

        # Generate transition frames
        print(f"Generating transition from {prev_video} to {next_video}...")
        transition_frames = self.interpolate_frames(last_frame, first_frame)

        # Play transition frames
        clock = pygame.time.Clock()
        for frame in transition_frames:
            if self.stop_all:
                break
            cv2.imshow("Video Player", frame)
            if cv2.waitKey(int(1000 / TARGET_FPS)) & 0xFF == 27:
                self.stop_all = True
                break
            clock.tick(TARGET_FPS)

    def loop_videos(self):
        while self.playing:
            if self.stop_all:
                break
            current_video_path = self.video_queue[self.current_video]
            
            cap = cv2.VideoCapture(current_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            cap.release()
            transition_start_frame = max(0, total_frames - fps)
            
            self.play_video(current_video_path, transition_start_frame)
            
            if self.playing and self.transitioning:
                if self.playing_response:
                    response_video = self.video_queue[self.current_video]
                    self.transition_videos(current_video_path, response_video, transition_start_frame)
                    total_frames_response, fps_response = self.play_video(response_video)
                    transition_start_frame_response = max(0, total_frames_response - fps_response)
                    self.video_queue = DEFAULT_VIDEOS.copy()
                    next_video = self.video_queue[0]
                    self.transition_videos(response_video, next_video, transition_start_frame_response)
                    self.current_video = 0
                    self.playing_response = False
                    self.transitioning = False
                else:
                    next_video = self.video_queue[(self.current_video + 1) % len(self.video_queue)]
                    self.transition_videos(current_video_path, next_video, transition_start_frame)
                    self.current_video = (self.current_video + 1) % len(self.video_queue)
                    self.transitioning = False

        cv2.destroyAllWindows()
        for file in os.listdir():
            if file.startswith("temp_audio_") and file.endswith(".mp3"):
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Cannot delete {file}: {e}")

    def play_response_video(self, video_path):
        if not os.path.exists(video_path):
            print(f"Video does not exist: {video_path}")
            return
        self.playing_response = True
        self.video_queue = [video_path]
        self.current_video = 0
        self.transitioning = True

class VoiceAssistant:
    def __init__(self, player):
        self.recognizer = sr.Recognizer()
        self.player = player

    def recognize_speech(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = self.recognizer.listen(source, timeout=5)
        try:
            text = self.recognizer.recognize_google(audio, language="vi-VN")
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Could not recognize speech.")
        except sr.RequestError:
            print("Speech recognition API connection error.")
        return None

    def find_best_match(self, input_text):
        available_questions = [d for d in os.listdir(VIDEO_DIR) if os.path.isdir(os.path.join(VIDEO_DIR, d)) and d != "root"]
        if not available_questions:
            return None

        prompt = f"""
        User input question: "{input_text}".
        List of available topics: {available_questions}.
        Return only the most relevant topic without extra text.
        If no topic matches, return "None".
        """

        print("Prompt sent to Gemini API:\n", prompt)
        
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
        best_match = response.text.strip().strip("`")

        print("Best match returned from Gemini API:", best_match)
        return best_match if best_match in available_questions else None

    def process_query(self):
        query = self.recognize_speech()
        if query:
            best_match = self.find_best_match(query)
            print("Best match after processing:", best_match)
            if best_match:
                question_dir = os.path.join(VIDEO_DIR, best_match)
                video_files = [f for f in os.listdir(question_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]
                if video_files:
                    video_path = os.path.join(question_dir, video_files[0])
                    self.player.play_response_video(video_path)

class GUI:
    def __init__(self, root, player, assistant):
        self.root = root
        self.player = player
        self.assistant = assistant
        self.root.title("AI Video Assistant")
        self.root.geometry("400x200")
        
        self.text_entry = tk.Entry(root, width=50)
        self.text_button = tk.Button(root, text="Search", command=self.process_text)
        self.voice_button = tk.Button(root, text="🎤 Speak", command=self.process_voice)
        
        self.show_interface()
        
        Thread(target=self.player.loop_videos, daemon=True).start()
        self.update_interface()

    def show_interface(self):
        self.text_entry.pack(pady=10)
        self.text_button.pack()
        self.voice_button.pack()

    def hide_interface(self):
        self.text_entry.pack_forget()
        self.text_button.pack_forget()
        self.voice_button.pack_forget()

    def update_interface(self):
        if self.player.playing_response:
            self.hide_interface()
        else:
            self.show_interface()
        self.root.after(100, self.update_interface)

    def process_text(self):
        query = self.text_entry.get()
        best_match = self.assistant.find_best_match(query)
        print("Best match from text input:", best_match)
        if best_match:
            question_dir = os.path.join(VIDEO_DIR, best_match)
            video_files = [f for f in os.listdir(question_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]
            if video_files:
                video_path = os.path.join(question_dir, video_files[0])
                self.player.play_response_video(video_path)
    
    def process_voice(self):
        Thread(target=self.assistant.process_query, daemon=True).start()

if __name__ == "__main__":
    player = VideoPlayer()
    assistant = VoiceAssistant(player)
    
    root = tk.Tk()
    gui = GUI(root, player, assistant)

    def check_exit():
        if player.stop_all:
            root.quit()
        root.after(100, check_exit)

    root.after(100, check_exit)
    root.mainloop()