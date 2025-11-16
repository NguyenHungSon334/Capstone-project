# AIP491 - Capstone Project: Trình phát Video AI Tương tác về Lịch sử Việt Nam

Đây là một dự án trình phát video tương tác nâng cao được xây dựng bằng Python. Ứng dụng này kết hợp nhận dạng giọng nói, AI tạo sinh (Google Gemini) và AI nội suy khung hình (RIFE) để tạo ra một trải nghiệm xem và tương tác với video liền mạch và điều khiển bằng giọng nói, giúp việc nói với những chiến sĩ, quân nhân dễ dàng biết về lịch sử Việt Nam.

Người dùng có thể xem một vòng lặp video "root" và sau đó yêu cầu hệ thống phát một video chủ đề cụ thể bằng cách sử dụng giọng nói, menu gợi ý hoặc nhập văn bản. Hệ thống sử dụng Gemini để hiểu yêu cầu và tìm video phù hợp, sau đó sử dụng RIFE để tạo hiệu ứng chuyển cảnh mượt mà giữa video gốc và video chủ đề.

## Các Tính năng Chính

  * **Vòng lặp Video Tương tác:** Phát liên tục một vòng lặp gồm hai video gốc (root) khi ở trạng thái chờ.
  * **Nhiều Phương thức Nhập liệu:**
      * **Điều khiển bằng Giọng nói:** Tự động lắng nghe lệnh của người dùng ("Nói tên thư mục...").
      * **Menu Gợi ý (Nút 'G'):** Một menu tương tác cho phép người dùng nhấp chuột để chọn từ các giai đoạn và câu hỏi được xác định trước (từ `hint.json`).
      * **Nhập Văn bản (Nút 'T'):** Một hộp thoại cho phép người dùng gõ tên thư mục mong muốn và gửi bằng phím Enter.
  * **Tìm kiếm bằng AI:** Mọi thông tin đầu vào (giọng nói hoặc văn bản) đều được xử lý bởi mô hình Gemini để tìm ra thư mục video phù hợp nhất với yêu cầu của người dùng.
  * **Chuyển cảnh bằng AI (RIFE):** Sử dụng mô hình AI RIFE (Real-time Intermediate Flow Estimation) để nội suy và tạo ra các khung hình trung gian, mang lại hiệu ứng chuyển cảnh "slow-motion" mượt mà khi chuyển từ video này sang video khác.
  * **Giao diện Động:** Tất cả các menu, nút bấm và hộp văn bản được vẽ trực tiếp lên cửa sổ video `OpenCV` bằng thư viện `PIL` (Pillow).

## Cấu trúc Thư mục Dự án (Bắt buộc)

Để chạy dự án, cấu trúc thư mục của bạn *phải* tuân theo định dạng dưới đây:

```
Capstone-project-master/
│
├── app/                    # Thư mục ứng dụng chính
│   ├── main.py             # File chạy chính
│   ├── audio_utils.py
│   ├── video_utils.py
│   ├── interpolation.py
│   ├── suggestion.py
│   └── arial.ttf         # File font bắt buộc (xem Hướng dẫn Cài đặt)
│
├── rife_1/                 # Thư mục chứa mô hình RIFE
│   └── train_log/
│       ├── RIFE_HDv3.py    # (File này được import trong code)
│       └── ... (Các file trọng số .pth của mô hình RIFE - xem Hướng dẫn Cài đặt)
│
├── videos/                 # Thư mục chứa tất cả các video (BỊ BỎ QUA BỞI .gitignore)
│   ├── root/
│   │   ├── video1.mp4      # Video vòng lặp 1
│   │   └── video2.mp4      # Video vòng lặp 2
│   │
│   └── [tên_thư_mục_1]/
│   │   └── video.mp4       # Video chủ đề 1 (tên file video bên trong không quan trọng)
│   │
│   └── [tên_thư_mục_2]/
│       └── video.mp4       # Video chủ đề 2
│
├── .gitignore
├── hint.json               # File JSON định nghĩa gợi ý cho menu 'G'
├── requirements.txt
└── README.md               # File này
```

## Hướng dẫn Cài đặt Chi tiết

### 1\. Yêu cầu Tiên quyết

  * Python 3.10 trở lên.
  * `pip` (Trình quản lý gói Python).
  * Một trình biên dịch C++ (thường cần thiết để cài đặt `torch`).
  * Một microphone (để sử dụng tính năng nhận dạng giọng nói).

### 2\. Thiết lập Môi trường và Thư viện

1.  **Tải dự án** và giải nén (hoặc `git clone`).
2.  Mở terminal và điều hướng đến thư mục gốc `Capstone-project-master`.
3.  (Khuyến nghị) Tạo một môi trường ảo:
    ```bash
    python -m venv venv
    ```
4.  Kích hoạt môi trường ảo:
      * **Windows:** `venv\Scripts\activate`
      * **macOS/Linux:** `source venv/bin/activate`
5.  Cài đặt tất cả các thư viện bắt buộc từ `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

### 3\. Tải Model AI (RIFE)

Dự án sử dụng mô hình RIFE v3 để nội suy khung hình, nhưng các file trọng số (ví dụ: `.pth`) không được bao gồm.

1.  Tìm và tải các file trọng số cho mô hình RIFE (ví dụ: từ kho lưu trữ GitHub gốc của RIFE).
2.  Tạo thư mục `rife_1/train_log/` trong thư mục gốc của dự án.
3.  Đặt các file trọng số (`.pth`) và file định nghĩa mô hình (`RIFE_HDv3.py` nếu thiếu) vào thư mục `rife_1/train_log/`. Mã nguồn tham chiếu đến `model.load_model(modelDir, -1)`, cho thấy nó mong đợi các file mô hình trong thư mục đó.

### 4\. Thiết lập File Video

Đây là bước quan trọng nhất. Dự án sẽ không chạy nếu không có các video này.

1.  Trong thư mục gốc `Capstone-project-master`, tạo một thư mục tên là `videos`.
2.  Bên trong `videos`, tạo một thư mục tên là `root`.
3.  Đặt hai video (ví dụ: `video1.mp4` và `video2.mp4`) vào trong `videos/root`. Đây là các video vòng lặp chính.
4.  Xem file `hint.json`. File này chứa các câu hỏi và các chuỗi "trả lời" ngắn gọn (ví dụ: `"Giơnevơ chia nước ở vĩ tuyến 17_..."`).
5.  Đối với *mỗi* câu hỏi trong `hint.json`, hãy tạo một thư mục mới bên trong `videos`. Tên của thư mục này *phải chính xác* là chuỗi "trả lời" trong `hint.json`.
      * Ví dụ: `videos/Giơnevơ chia nước ở vĩ tuyến 17_ Bắc khôi phục, Nam do Diệm (Mỹ hậu thuẫn), nhiệm vụ_ giữ lực lượng và chuẩn bị thống nhất/`
6.  Bên trong mỗi thư mục chủ đề này, đặt *một* file video (ví dụ: `video.mp4`). Tên của file video bên trong không quan trọng, miễn là nó là một định dạng video hợp lệ (`.mp4`, `.avi`, `.mov`).

## Cách chạy Dự án

Sau khi hoàn thành tất cả các bước cài đặt:

1.  Đảm bảo môi trường ảo của bạn đã được kích hoạt.

2.  Từ thư mục gốc `Capstone-project-master` (nơi chứa `requirements.txt`), chạy lệnh sau:

    ```bash
    python app/main.py
    ```

3.  Một cửa sổ OpenCV có tên "Video Loop" sẽ xuất hiện và bắt đầu phát vòng lặp video root.

## Cách sử dụng Ứng dụng

Khi ứng dụng đang chạy:

  * **Để chọn chủ đề bằng Giọng nói:** Nói rõ ràng tên của một chủ đề hoặc giai đoạn (ví dụ: "Giai đoạn 1" hoặc "Sự kiện Vịnh Bắc Bộ"). Microphone của bạn sẽ lắng nghe tự động.
  * **Để chọn chủ đề bằng Menu:** Nhấp vào nút tròn 'G' ở góc dưới bên trái. Sử dụng chuột để nhấp vào một giai đoạn, sau đó nhấp vào một câu hỏi cụ thể.
  * **Để chọn chủ đề bằng Văn bản:** Nhấp vào nút tròn 'T' ở góc dưới bên phải. Một hộp nhập văn bản sẽ xuất hiện. Gõ tên thư mục (hoặc một phần của nó) và nhấn `Enter`.
  * **Để Thoát:** Nhấn phím `ESC` bất cứ lúc nào để đóng ứng dụng.