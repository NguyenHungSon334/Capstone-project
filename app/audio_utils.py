
import numpy as np
import threading
import queue
import librosa
import simpleaudio as sa

sample_rate = 44100


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
