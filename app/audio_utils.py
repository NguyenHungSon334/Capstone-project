import numpy as np
import threading
import queue
import librosa
import pygame  # SỬ DỤNG PYGAME
import speech_recognition as sr
import google.generativeai as genai
import os


sample_rate = 44100

# --- KHỞI TẠO PYGAME MIXER ---
try:
    pygame.mixer.pre_init(sample_rate, -16, 2, 2048)
    pygame.mixer.init()
    pygame.init()
except Exception as e:
    print(f"❌ KHÔNG THỂ KHỞI TẠO PYGAME: {e}")
    print("Hãy đảm bảo bạn đã cài đặt thư viện 'pygame' (pip install pygame)")
# --- KẾT THÚC KHỞI TẠO ---

VIDEO_DIR = r".\videos"
folder_queue = queue.Queue() 

def process_audio(video_path, start_time, duration=0.2):
    try:
        audio_data, sr = librosa.load(video_path, sr=sample_rate, offset=start_time, duration=duration + 0.1) 
        
        target_samples = int(duration * sample_rate)
        if len(audio_data) > target_samples:
            if start_time < 0:
                 audio_data = audio_data[-target_samples:]
            else:
                audio_data = audio_data[:target_samples]
        
        if len(audio_data) == 0:
            print(f"No audio found in video: {video_path} (for transition snippet)")
            return None
            
        audio_data = apply_fade(audio_data, sr, int(0.2 * sr))
        return audio_data
    except Exception as e:
        if "No audio stream" in str(e) or "Backend" in str(e):
             print(f"No audio stream found in: {video_path}")
             return None
        print(f"Error processing audio for {video_path}: {str(e)}")
        return None

def apply_fade(audio_data, sr, fade_samples):
    if len(audio_data) < fade_samples:
        fade_samples = len(audio_data) // 2
    if fade_samples == 0:
        return audio_data
    fade_in = np.linspace(0, 1, fade_samples)
    audio_data[:fade_samples] *= fade_in
    fade_out = np.linspace(1, 0, fade_samples)
    audio_data[-fade_samples:] *= fade_out
    return audio_data

audio_queue = queue.Queue()
audio_lock = threading.Lock() 

# --- SỬA LỖI DEADLOCK (DÙNG PYGAME) ---
def play_audio(audio_data, sr=sample_rate):
    if audio_data is None or len(audio_data) == 0:
        return
    try:
        with audio_lock: 
            audio_data_int = (audio_data * 32767).astype(np.int16)
            
            if audio_data_int.ndim == 1:
                audio_data_int = np.stack([audio_data_int, audio_data_int], axis=1)
            
            if not audio_data_int.flags['C_CONTIGUOUS']:
                audio_data_int = np.ascontiguousarray(audio_data_int)

            sound = pygame.mixer.Sound(audio_data_int)
            sound.play()
            # Không có wait_done()
            
    except Exception as e:
        print(f"Error playing pygame audio: {e}")
# --- KẾT THÚC SỬA LỖI ---

# --- HÀM RIÊNG CHO GEMINI API ---
genai.configure(api_key="AIzaSyBT7G6EfLR45FLyoFVDC8ft2zbfhkx01Oo")
GEMINI_MODEL = "gemini-2.0-flash"

def find_best_folder_match(user_query: str, available_folders: list) -> str:
    """
    Dùng Gemini để tìm thư mục phù hợp nhất với câu nói của người dùng.
    
    Args:
        user_query (str): Câu người dùng nói 
        available_folders (list): Danh sách thư mục có sẵn 
    
    Returns:
        str: Tên thư mục phù hợp nhất, hoặc "None" nếu không tìm thấy
    """
    if not user_query.strip():
        return "None"

    # Tạo prompt rõ ràng, ngắn gọn
    prompt = f"""
    Người dùng nói: "{user_query}"
    Danh sách thư mục có sẵn: {available_folders}

    Hãy chọn đúng 1 thư mục DUY NHẤT phù hợp nhất với ý người dùng.
    - Chỉ trả về tên thư mục chính xác (không thêm dấu ngoặc, không giải thích).
    - Nếu không có thư mục nào phù hợp, trả về đúng: None
    """

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        best_match = response.text.strip().strip("`").strip('"').strip("'")
        
        print(f"Gemini gợi ý: '{best_match}'")
        
        if best_match == "None" or best_match not in available_folders:
            return "None"
        return best_match

    except Exception as e:
        print(f"Lỗi gọi Gemini API: {e}")
        return "None"
# --- KẾT THÚC HÀM GEMINI ---

def speech_callback(recognizer, audio):
    try:
        query = recognizer.recognize_google(audio, language='vi-VN')
        print(f"✅ Recognized (Vietnamese): {query}")
        
        available_folders = [d for d in os.listdir(VIDEO_DIR) if os.path.isdir(os.path.join(VIDEO_DIR, d)) and d != "root"]
        
        # Sử dụng hàm riêng
        best_match = find_best_folder_match(query, available_folders)
        
        if best_match != "None":
            folder_queue.put(best_match)
        else:
            print(f"No matching folder found or invalid match: {best_match}")
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
    except sr.RequestError as e:
        print(f"❌ Could not request results; {e}")
    except Exception as e:
        print(f"❌ Gemini API error: {e}")