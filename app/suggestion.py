import cv2
import numpy as np
from PIL import Image, ImageDraw
import json  # Thêm import json để đọc file

# --- CẤU TRÚC HINT TƯƠNG TÁC MỚI ---
# (Lấy từ main.py cũ của chúng ta)
with open('./hint.json', 'r', encoding='utf-8') as f:
    HINT_DATA = json.load(f)
# --- KẾT THÚC CẤU TRÚC HINT ---

# --- HÀM HELPER VẼ CHỮ ---
def draw_text_pil_suggestion(img, text, position, font, color_bgr):
    try:
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0]) # BGR to RGB
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=font, fill=color_rgb)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR) # RGB to BGR
    except Exception as e:
        print(f"Loi ve van ban (Suggestion): {e}")
    
        return img

# --- HÀM NGẮT DÒNG (FIX LỖI TRÀN CHỮ) ---
def wrap_text(text, font, max_width):
    """Ngắt một đoạn text dài thành nhiều dòng ngắn hơn"""
    lines = []
    if font.getlength(text) <= max_width:
        return [text]
    
    words = text.split(' ')
    current_line = ""
    for word in words:
        if font.getlength(current_line + " " + word) <= max_width:
            current_line += " " + word
        else:
            lines.append(current_line.strip())
            current_line = word
    lines.append(current_line.strip()) # Thêm dòng cuối
    return lines

class SuggestionHandler:
    def __init__(self, target_height, target_width, video_dir, folder_queue, get_current_mode_func, get_waiting_for_transition_func, 
                 font_title, font_item, font_button, font_regular):
        self.show_suggestions = False
        self.show_input_box = False  # Flag cho input overlay
        self.input_text = ""  # Text đang nhập
        self.selected_index = -1
        self.menu_x = 90
        self.menu_y = 150 # Nâng menu lên cao hơn một chút
        self.menu_width = 600 # Tăng chiều rộng
        self.menu_height = 450 # Tăng chiều cao
        self.item_height = 30
        self.scroll_offset = 0
        self.button_center_suggest = (50, target_height - 50)  # Nút 'G' cho suggestions
        self.button_center_input = (target_width - 50, target_height - 50)  # Nút 'T' cho input
        self.button_radius = 30
        self.video_dir = video_dir
        self.folder_queue = folder_queue
        self.get_current_mode = get_current_mode_func
        self.get_waiting_for_transition = get_waiting_for_transition_func
        self.target_height = target_height
        self.target_width = target_width
        
        # --- LOGIC MENU MỚI ---
        self.current_menu_level = "main" # 'main' hoặc số (1-6)
        self.visible_items_data = [] # Lưu trữ (text, data, rect) của các mục đang hiển thị
        self.back_button_rect = None # Lưu tọa độ nút "Quay lại"
        # --- KẾT THÚC LOGIC MENU ---

        # --- LƯU TRỮ FONT ---
        self.font_title = font_title
        self.font_item = font_item
        self.font_button = font_button
        self.font_input = font_regular  # Font cho input text, dùng font_regular từ main
        # --- KẾT THÚC LƯU TRỮ FONT ---

    def is_over_suggest_button(self, x, y):
        dist = np.sqrt((x - self.button_center_suggest[0])**2 + (y - self.button_center_suggest[1])**2)
        return dist <= self.button_radius

    def is_over_input_button(self, x, y):
        dist = np.sqrt((x - self.button_center_input[0])**2 + (y - self.button_center_input[1])**2)
        return dist <= self.button_radius

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.show_suggestions:
                # --- LOGIC CLICK MỚI ---
                
                # 1. Kiểm tra nút "Quay lại" (nếu có)
                if self.back_button_rect:
                    bx, by, bw, bh = self.back_button_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self.current_menu_level = "main"
                        self.scroll_offset = 0
                        return # Đã xử lý, thoát

                # 2. Kiểm tra các mục trong danh sách
                for (text, data, rect) in self.visible_items_data:
                    rx, ry, rw, rh = rect
                    if rx <= x <= rx + rw and ry <= y <= ry + rh:
                        if self.current_menu_level == "main":
                            # Click vào giai đoạn -> Chuyển menu
                            self.current_menu_level = data # data là số (1-6)
                            self.scroll_offset = 0
                        else:
                            # Click vào câu hỏi -> Gửi lệnh
                            folder_name = data # data là tên thư mục
                            self.folder_queue.put(folder_name)
                            self.show_suggestions = False
                            self.current_menu_level = "main" # Reset
                        return # Đã xử lý, thoát
                
                # 3. Click ra ngoài để đóng
                if not (self.menu_x <= x <= self.menu_x + self.menu_width and self.menu_y <= y <= self.menu_y + self.menu_height):
                    self.show_suggestions = False
                    self.current_menu_level = "main"
                    self.scroll_offset = 0
                # --- KẾT THÚC LOGIC CLICK ---
            
            else:
                # Check circular buttons click
                if self.get_current_mode() != "root":
                    return
                if self.is_over_suggest_button(x, y) and not self.get_waiting_for_transition():
                    self.show_suggestions = not self.show_suggestions  # Toggle suggestions
                    self.show_input_box = False  # Tắt input nếu đang mở
                    self.input_text = ""
                    self.current_menu_level = "main"
                    self.scroll_offset = 0
                elif self.is_over_input_button(x, y) and not self.get_waiting_for_transition():
                    self.show_input_box = not self.show_input_box  # Toggle input box
                    self.show_suggestions = False  # Tắt suggestions nếu đang mở
                    self.input_text = ""  # Reset text

        elif event == cv2.EVENT_MOUSEWHEEL:
            if self.show_suggestions and self.menu_x <= x <= self.menu_x + self.menu_width and self.menu_y <= y <= self.menu_y + self.menu_height:
                if flags > 0:  # Scroll up
                    self.scroll_offset = max(0, self.scroll_offset - 1)
                else:  # Scroll down
                    # Tính max_scroll dựa trên menu hiện tại
                    items_count = 0
                    if self.current_menu_level == "main":
                        items_count = len(HINT_DATA)
                    else:
                        items_count = len(HINT_DATA[self.current_menu_level]["questions"])
                    
                    visible_items_count = (self.menu_height - 60) // self.item_height
                    max_scroll = max(0, items_count - visible_items_count)
                    self.scroll_offset = min(max_scroll, self.scroll_offset + 1)

    # (Các hàm draw_rounded_rect và draw_filled_rounded_rect giữ nguyên)
    def draw_rounded_rect(self, img, rect_start, rect_end, color, thickness, radius):
        x, y = rect_start
        w, h = rect_end[0] - x, rect_end[1] - y
        cv2.line(img, (x + radius, y), (x + w - radius, y), color, thickness)
        cv2.line(img, (x + w, y + radius), (x + w, y + h - radius), color, thickness)
        cv2.line(img, (x + w - radius, y + h), (x + radius, y + h), color, thickness)
        cv2.line(img, (x, y + h - radius), (x, y + radius), color, thickness)
        cv2.ellipse(img, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, thickness)
        cv2.ellipse(img, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, thickness)

    def draw_filled_rounded_rect(self, img, rect_start, rect_end, color, radius, alpha=0.7):
        x1, y1 = rect_start
        x2, y2 = rect_end
        overlay = img.copy()
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x1 + radius, y2 - radius), color, -1)
        cv2.rectangle(overlay, (x2 - radius, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return img

    def draw_suggestion_overlay(self, frame):
        self.visible_items_data = [] # Reset
        self.back_button_rect = None # Reset
        
        # Vẽ nền
        frame = self.draw_filled_rounded_rect(frame, (self.menu_x, self.menu_y), (self.menu_x + self.menu_width, self.menu_y + self.menu_height), (255, 255, 255), 20)
        # Vẽ viền
        self.draw_rounded_rect(frame, (self.menu_x, self.menu_y), (self.menu_x + self.menu_width, self.menu_y + self.menu_height), (0, 0, 0), 2, 20)
        
        y_pos = self.menu_y + 20 # Vị trí bắt đầu vẽ
        
        if self.current_menu_level == "main":
            # --- VẼ MENU CHÍNH (CÁC GIAI ĐOẠN) ---
            frame = draw_text_pil_suggestion(frame, "Chọn Giai Đoạn (Nhấn 'G' để đóng)", (self.menu_x + 10, y_pos), self.font_title, (0, 0, 0))
            y_pos += 40
            
            items_to_draw = list(HINT_DATA.items())
            visible_items_count = (self.menu_height - 60) // self.item_height
            
            for i in range(self.scroll_offset, min(self.scroll_offset + visible_items_count, len(items_to_draw))):
                stage_num, data = items_to_draw[i]
                item_text = f"{stage_num}. {data['name']}"
                
                # Ngắt dòng text
                wrapped_lines = wrap_text(item_text, self.font_item, self.menu_width - 40)
                
                item_x = self.menu_x + 10
                item_y = y_pos
                item_w = self.menu_width - 20
                item_h = len(wrapped_lines) * self.item_height
                
                # Lưu tọa độ để click
                self.visible_items_data.append((item_text, stage_num, (item_x, item_y, item_w, item_h)))
                
                for line in wrapped_lines:
                    frame = draw_text_pil_suggestion(frame, line, (item_x + 10, y_pos), self.font_item, (0, 0, 0))
                    y_pos += self.item_height
                
                y_pos += 10 # Thêm khoảng cách giữa các mục
        
        else:
            # --- VẼ MENU CON (CÁC CÂU HỎI) ---
            stage_num = self.current_menu_level
            stage_title = HINT_DATA[stage_num]["name"]
            questions_data = HINT_DATA[stage_num]["questions"]
            
            frame = draw_text_pil_suggestion(frame, stage_title, (self.menu_x + 10, y_pos), self.font_title, (0, 0, 200)) # Màu xanh
            y_pos += 40
            
            # Vẽ nút "Quay lại"
            back_text = "< Quay lai"
            self.back_button_rect = (self.menu_x + self.menu_width - 120, self.menu_y + 15, 110, 30)
            frame = draw_text_pil_suggestion(frame, back_text, (self.back_button_rect[0], self.back_button_rect[1]), self.font_item, (150, 0, 0))
            
            visible_items_count = (self.menu_height - 80) // self.item_height
            
            for i in range(self.scroll_offset, min(self.scroll_offset + visible_items_count, len(questions_data))):
                question_text, folder_name = questions_data[i]
                item_text = f"{i+1}. {question_text}"
                
                wrapped_lines = wrap_text(item_text, self.font_item, self.menu_width - 40)
                
                item_x = self.menu_x + 10
                item_y = y_pos
                item_w = self.menu_width - 20
                item_h = len(wrapped_lines) * self.item_height

                self.visible_items_data.append((item_text, folder_name, (item_x, item_y, item_w, item_h)))
                
                for line in wrapped_lines:
                    frame = draw_text_pil_suggestion(frame, line, (item_x + 10, y_pos), self.font_item, (0, 0, 0))
                    y_pos += self.item_height
                
                y_pos += 10
        
        # (Vẽ thanh cuộn - logic cũ giữ nguyên, nhưng điều chỉnh)
        
        return frame

    def draw_input_overlay(self, frame):
        # Vị trí giữa dưới khung hình
        input_width = 600
        input_height = 100
        input_x = (self.target_width - input_width) // 2  # Giữa ngang
        input_y = self.target_height - input_height - 50  # Dưới cùng, cách đáy 50 pixel
        
        # Vẽ nền
        frame = self.draw_filled_rounded_rect(frame, (input_x, input_y), (input_x + input_width, input_y + input_height), (255, 255, 255), 20)
        # Vẽ viền
        self.draw_rounded_rect(frame, (input_x, input_y), (input_x + input_width, input_y + input_height), (0, 0, 0), 2, 20)
        
        # Vẽ tiêu đề
        frame = draw_text_pil_suggestion(frame, "Nhập tên thư mục (Enter để gửi, ESC để hủy):", (input_x + 10, input_y + 10), self.font_title, (0, 0, 0))
        
        # Vẽ hộp text
        text_box_y = input_y + 50
        cv2.rectangle(frame, (input_x + 10, text_box_y), (input_x + input_width - 10, text_box_y + 40), (200, 200, 200), -1)  # Nền xám
        cv2.rectangle(frame, (input_x + 10, text_box_y), (input_x + input_width - 10, text_box_y + 40), (0, 0, 0), 2)  # Viền đen
        
        # Vẽ text đang nhập (hỗ trợ ngắt dòng nếu dài)
        wrapped_lines = wrap_text(self.input_text, self.font_input, input_width - 40)
        y_pos = text_box_y + 5
        for line in wrapped_lines:
            frame = draw_text_pil_suggestion(frame, line, (input_x + 20, y_pos), self.font_input, (0, 0, 0))
            y_pos += 30  # Khoảng cách dòng
        
        # Thêm cursor blinking (giả, chỉ vẽ '|')
        cursor_x = input_x + 20 + self.font_input.getlength(self.input_text)
        frame = draw_text_pil_suggestion(frame, "|", (cursor_x, text_box_y + 5), self.font_input, (0, 0, 0))
        
        return frame

    def draw_circular_buttons(self, frame):
        # Vẽ nút 'G' cho suggestions (giữ nguyên)
        cv2.circle(frame, self.button_center_suggest, self.button_radius, (255, 255, 255), -1)  # Filled white
        cv2.circle(frame, self.button_center_suggest, self.button_radius, (0, 0, 0), 2)  # Black border
        text_pos_x = self.button_center_suggest[0] - 12
        text_pos_y = self.button_center_suggest[1] - 15
        frame = draw_text_pil_suggestion(frame, "G", (text_pos_x, text_pos_y), 
                                         self.font_button, (0, 0, 0))
        
        # Vẽ nút 'T' cho input
        cv2.circle(frame, self.button_center_input, self.button_radius, (255, 255, 255), -1)  # Filled white
        cv2.circle(frame, self.button_center_input, self.button_radius, (0, 0, 0), 2)  # Black border
        text_pos_x = self.button_center_input[0] - 12
        text_pos_y = self.button_center_input[1] - 15
        frame = draw_text_pil_suggestion(frame, "T", (text_pos_x, text_pos_y), 
                                         self.font_button, (0, 0, 0))
        
        return frame