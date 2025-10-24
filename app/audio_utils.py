
import numpy as np
import threading
import queue
import librosa
import simpleaudio as sa
import speech_recognition as sr  # Added for speech-to-text
import google.generativeai as genai  # Added for Gemini API
import os

sample_rate = 44100

VIDEO_DIR = r"D:\Python\code\videos"
folder_queue = queue.Queue() 


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


def speech_callback(recognizer, audio):
    print("🔴 Processing audio... (detected speech)")
    try:
        # Thay đổi ngôn ngữ thành tiếng Việt
        query = recognizer.recognize_google(audio, language='vi-VN')
        print(f"✅ Recognized (Vietnamese): {query}")
        
        # Get available folders
        available_folders = [d for d in os.listdir(VIDEO_DIR) if os.path.isdir(os.path.join(VIDEO_DIR, d)) and d != "root"]
        
        # Create prompt for Gemini
        prompt = f"""
        User input question: "{query}".
        List of available topics: {available_folders}.
        Return only the most relevant topic without extra text.
        If no topic matches, return "None".
        """
        
        # Call Gemini API
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
        best_match = response.text.strip().strip("`")
        
        print(f"Best match from Gemini: {best_match}")
        
        if best_match != "None":
            folder_queue.put(best_match)
        else:
            print("No matching folder found.")
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
    except sr.RequestError as e:
        print(f"❌ Could not request results; {e}")
    except Exception as e:
        print(f"❌ Gemini API error: {e}")

