import cv2
import os

# Đường dẫn video
video_path = r"D:\Python\code\videos\Diệm hủy tổng tuyển cử, siết cai trị với Mỹ hậu thuẫn_ miền Nam bắt bớ khiến cách mạng vào bí mật, còn miền Bắc xây kinh tế–quốc phòng làm chỗ dựa\2.MOV"

# Kiểm tra tồn tại
if not os.path.exists(video_path):
    print("❌ Không tìm thấy file video tại:", video_path)
    exit()

# Mở video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Không mở được video. Kiểm tra codec hoặc định dạng file.")
    exit()

print("✅ Đang phát video ở kích thước 1280x720... Nhấn 'q' để thoát.")

# Vòng lặp hiển thị từng frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("🔚 Hết video hoặc lỗi đọc frame.")
        break

    # Resize frame về 1280x720
    frame_resized = cv2.resize(frame, (1280, 720))

    # Hiển thị frame
    cv2.imshow("Video Player - 1280x720", frame_resized)

    # Nhấn 'q' để thoát
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
