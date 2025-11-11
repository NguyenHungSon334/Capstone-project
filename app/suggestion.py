import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont # THÊM IMPORT

# --- CẤU TRÚC HINT TƯƠNG TÁC MỚI ---
# (Lấy từ main.py cũ của chúng ta)
HINT_DATA = {
    1: {
        "name": "Giai đoạn 1: Bối cảnh và Hiệp định Giơnevơ (1954)",
        "questions": [
            ("Tình hình sau Hiệp định Giơnevơ 1954?", "Giơnevơ chia nước ở vĩ tuyến 17_ Bắc khôi phục, Nam do Diệm (Mỹ hậu thuẫn), nhiệm vụ_ giữ lực lượng và chuẩn bị thống nhất"),
            ("Mục tiêu pháp lý của Hiệp định Giơnevơ là gì?", "mục tiêu pháp lý giơnevo"),
            ("Sự kiện ngày 20-21 tháng 7 năm 1954 là gì?", "Ngày 20–21 tháng 7 năm 1954 – một mốc lịch sử quan trọng đã thay đổi cục diện Đông Dương"),
            ("Sau Hiệp định Giơnevơ 1954, quân đội hai bên làm gì?", "Sau Hiệp định Giơnevơ năm 1954, quân đội hai bên thực hiện điều gì"),
            ("Sau Hiệp định Giơnevơ, miền Bắc đã làm gì?", "Sau Hiệp định Giơnevơ, lực lượng cách mạng ở miền Bắc đã làm gì để chuẩn bị cho nhiệm vụ lâu dài"),
            ("Sau Hiệp định Giơnevơ, thời cơ nào để tổ chức tổng tuyển cử?", "Sau Hiệp định Giơnevơ, thời cơ nào đã tạo điều kiện cho Việt Nam tổ chức tổng tuyển cử thống nhất đất nước"),
            ("Ý nghĩa của vĩ tuyến 17 theo Hiệp định Giơnevơ?", "Theo Hiệp định Giơnevơ, việc lấy vĩ tuyến 17 làm ranh giới quân sự tạm thời có ý nghĩa gì"),
            ("Ý nghĩa của chiến thắng Điện Biên Phủ là gì?", "ý nghĩa sau chiến thắng điện biên phủ")
        ]
    },
    2: {
        "name": "Giai đoạn 2: Thời kỳ đầu chia cắt (1954 - 1960)",
        "questions": [
            ("Tình hình 2 miền Nam-Bắc sau khi Diệm hủy tổng tuyển cử?", "Diệm hủy tổng tuyển cử, siết cai trị với Mỹ hậu thuẫn_ miền Nam bắt bớ khiến cách mạng vào bí mật, còn miền Bắc xây kinh tế–quốc phòng làm chỗ dựa"),
            ("Miền Bắc đã làm gì sau cải cách ruộng đất?", "Miền Bắc hoàn tất cải cách ruộng đất, kinh tế phục hồi thành chỗ dựa_ miền Nam giữ hạt nhân bí mật, manh nha tự vệ vũ trang—thế Bắc hậu phương, Nam chiến trường ngày càng rõ"),
            ("Miền Bắc đã chi viện cho miền Nam như thế nào?", "Miền Bắc tăng tốc xây dựng để chi viện_ miền Nam củng cố cơ sở, nhen nhóm tự vệ dù bị truy quét_ toàn dân chuẩn bị cho chuyển thế lớn"),
            ("Nghị quyết 15 (NQ15) và việc mở đường Trường Sơn?", "NQ15 xác định bạo lực cách mạng ở miền Nam_ Sài Gòn ban Luật 10-59 đàn áp cực độ_ Đoàn 559 mở tuyến Trường Sơn chi viện_ Nam Bộ nổi dậy—bước ngoặt từ giữ lực lượng sang thế tiến công"),
            ("Chuyện gì xảy ra với cuộc tổng tuyển cử?", "Tổng tuyển cử theo Giơnevơ bị phá_ miền Nam siết kiểm soát, mở nhà giam, còn miền Bắc chỉnh đốn tổ chức, sản xuất và sẵn sàng chi viện người–vũ khí cho miền Nam"),
            ("Kể về phong trào Đồng Khởi ở Bến Tre.", "Đồng Khởi bùng nổ từ Bến Tre lan khắp Nam Bộ–Tây Nguyên, phá thế kìm kẹp nông thôn, dẫn tới ra đời Mặt trận Dân tộc Giải phóng miền Nam (20-12-1960), nhân dân giành quyền làm chủ ở nhiều vùng rộng lớn")
        ]
    },
    3: {
        "name": "Giai đoạn 3: Chiến tranh Đặc biệt (1961 - 1964)",
        "questions": [
            ("MACV là gì và vai trò của nó ra sao?", "Mỹ lập MACV, tăng trực thăng vận và cơ giới hóa, nhưng gặp kháng cự rộng khắp_ phong trào phá ấp lan nhanh làm rỗng chương trình, lực lượng giải phóng trưởng thành mạnh"),
            ("'Chiến tranh đặc biệt' của Mỹ-Diệm là gì?", "Mỹ–Diệm mở “Chiến tranh đặc biệt” (cố vấn, trực thăng, ấp chiến lược) nhưng ta bám dân phá ấp, phát triển du kích, mở rộng vùng giải phóng, khiến Sài Gòn khó kiểm soát và cán cân không nghiêng về phía Mỹ như kỳ vọng"),
            ("Sự kiện Vịnh Bắc Bộ là gì?", "Sài Gòn bất ổn vì đảo chính_ sự kiện Vịnh Bắc Bộ cho Mỹ cái cớ mở rộng chiến tranh, “đặc biệt” coi như thất bại, Washington chuẩn bị đổ quân và ném bom miền Bắc—cục diện bước sang giai đoạn khốc liệt hơn"),
            ("Kể về trận Ấp Bắc và khủng hoảng Phật giáo 1963.", "Thắng lợi Ấp Bắc cùng khủng hoảng Phật giáo làm Sài Gòn rúng động, Diệm–Nhu bị lật đổ (11-1963), “chiến tranh đặc biệt” bế tắc và Mỹ đứng trước ngã rẽ _leo thang_")
        ]
    },
    4: {
        "name": "Giai đoạn 4: Chiến tranh Cục bộ (1965 - 1968)",
        "questions": [
            ("Chiến dịch Mùa khô lần 2 (Junction City) diễn ra sao?", "Mùa khô lần 2 (có Junction City) khiến thương vong Mỹ tăng và phản chiến nhen nhóm, ta vừa tiêu diệt sinh lực vừa giữ lực lượng mở bàn đạp, trong khi Sài Gòn loay hoay bầu cử"),
            ("Kể về sự kiện Mậu Thân 1968.", "Mậu Thân 1968_ ta đồng loạt đánh vào hầu hết đô thị, Khe Sanh hút quân Mỹ_ cú sốc chính trị–tâm lý buộc Johnson ngừng ném bom hạn chế, không tái tranh cử và mở đàm phán Paris"),
            ("Mỹ bắt đầu 'chiến tranh cục bộ' và Rolling Thunder như thế nào?", "Mỹ đổ bộ Đà Nẵng mở “chiến tranh cục bộ” và ném bom miền Bắc (Rolling Thunder)_ ta thắng Vạn Tường bẻ “tìm diệt”, miền Bắc vừa sản xuất vừa chiến đấu vẫn chi viện, chiến tranh bước vào giai đoạn ác liệt"),
            ("Chiến dịch mùa khô lần thứ nhất diễn ra như thế nào?", "Mỹ–ngụy mở _mùa khô lần 1_ với hỏa lực lớn nhưng ta đánh bại nhiều cuộc càn, giữ vững vùng và chủ lực_ “tìm diệt” thất bại, miền Bắc duy trì giao thông–sản xuất–chiến đấu nhịp nhàng, và Mỹ bắt đầu sa lầy")
        ]
    },
    5: {
        "name": "Giai đoạn 5: Việt Nam hóa Chiến tranh (1969 - 1973)",
        "questions": [
            ("Nội dung của Hiệp định Paris 1973 là gì?", "Hiệp định Paris (27-1-1973)_ Mỹ rút quân, trao trả tù binh_ ta củng cố lực lượng, mở rộng vùng giải phóng, miền Bắc tăng sản xuất–chi viện, trong khi Sài Gòn phá hoại nhưng yếu thế"),
            ("Chiến dịch Lam Sơn 719 diễn ra như thế nào?", "Lam Sơn 719_ quân Sài Gòn (có Mỹ yểm trợ) đánh sang Nam Lào nhằm cắt Trường Sơn nhưng bị ta đánh bại nặng, lộ rõ điểm yếu, tuyến chi viện vẫn an toàn, ta giữ thế chủ động và củng cố niềm tin tất thắng"),
            ("Tại sao Mỹ và Sài Gòn đánh sang Campuchia?", "Mỹ–Sài Gòn đánh sang Campuchia để cắt chi viện, nhưng ta phối hợp ba nước Đông Dương nối liền hành lang–hậu cứ, mở rộng vùng giải phóng_ miền Bắc tăng sản xuất–chi viện, bộc lộ rõ bất cập của “Việt Nam hoá”"),
            ("Kể về Mùa hè đỏ lửa 1972 (Tiến công Trị-Thiên, Tây Nguyên).", "Năm 1972, ta tiến công lớn ở Trị–Thiên, Tây Nguyên, Đông Nam Bộ_ Mỹ đáp trả bằng Linebacker và B-52 “Điện Biên Phủ trên không” nhưng Hà Nội–Hải Phòng đứng vững, buộc Mỹ nhượng bộ và mở đường ký Hiệp định Paris"),
            ("'Việt Nam hóa chiến tranh' của Nixon là gì?", "Nixon thực hiện “Việt Nam hoá”_ rút dần quân Mỹ, giao gánh nặng cho quân Sài Gòn_ ta vừa tiến công vừa đấu tranh ngoại giao_ Bác Hồ qua đời, phong trào phản chiến Mỹ lan rộng—thế chiến lược nghiêng về phía ta")
        ]
    },
    6: {
        "name": "Giai đoạn 6: Hướng tới Thống nhất (1973 - 1975)",
        "questions": [
            ("Kể về sự kiện 30 tháng 4 năm 1975.", "Ta thắng Buôn Ma Thuột, Huế–Đà Nẵng sụp_ Hồ Chí Minh toàn thắng, 30-4-1975 Dương Văn Minh đầu hàng_ miền Nam giải phóng, thống nhất"),
            ("Tình hình Sài Gòn trước 1975 như thế nào?", "Ta thử lửa chiến dịch vừa, rạn phòng ngự địch_ Sài Gòn khủng hoảng_ Bộ Chính trị chốt phương án tổng tiến công")
        ]
    }
}
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
    def __init__(self, target_height, video_dir, folder_queue, get_current_mode_func, get_waiting_for_transition_func, 
                 font_title, font_item, font_button):
        self.show_suggestions = False
        self.selected_index = -1
        self.menu_x = 90
        self.menu_y = 150 # Nâng menu lên cao hơn một chút
        self.menu_width = 600 # Tăng chiều rộng
        self.menu_height = 450 # Tăng chiều cao
        self.item_height = 30
        self.scroll_offset = 0
        self.button_center = (50, target_height - 50)
        self.button_radius = 30
        self.video_dir = video_dir
        self.folder_queue = folder_queue
        self.get_current_mode = get_current_mode_func
        self.get_waiting_for_transition = get_waiting_for_transition_func
        
        # --- LOGIC MENU MỚI ---
        self.current_menu_level = "main" # 'main' hoặc số (1-6)
        self.visible_items_data = [] # Lưu trữ (text, data, rect) của các mục đang hiển thị
        self.back_button_rect = None # Lưu tọa độ nút "Quay lại"
        # --- KẾT THÚC LOGIC MENU ---

        # --- LƯU TRỮ FONT ---
        self.font_title = font_title
        self.font_item = font_item
        self.font_button = font_button
        # --- KẾT THÚC LƯU TRỮ FONT ---

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
                            print(f"📁 Selected suggestion: '{folder_name}'")
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
                # Check circular button click
                if self.get_current_mode() != "root":
                    return
                dist = np.sqrt((x - self.button_center[0])**2 + (y - self.button_center[1])**2)
                if dist <= self.button_radius and not self.get_waiting_for_transition():
                    print("🖱️ Suggestion button clicked! Showing overlay...")
                    self.show_suggestions = True
                    self.current_menu_level = "main"
                    self.scroll_offset = 0

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

    def draw_circular_button(self, frame):
        # Draw white circle
        cv2.circle(frame, self.button_center, self.button_radius, (255, 255, 255), -1)  # Filled white
        # Draw border
        cv2.circle(frame, self.button_center, self.button_radius, (0, 0, 0), 2)  # Black border
        
        text_pos_x = self.button_center[0] - 12
        text_pos_y = self.button_center[1] - 15
        frame = draw_text_pil_suggestion(frame, "G", (text_pos_x, text_pos_y), 
                                         self.font_button, (0, 0, 0))
        
        return frame