import cv2
import numpy as np
import os

class SuggestionHandler:
    def __init__(self, target_height, video_dir, folder_queue, get_current_mode_func, get_waiting_for_transition_func):
        self.show_suggestions = False
        self.suggestions = []
        self.selected_index = -1
        self.menu_x = 90
        self.menu_y = 300
        self.menu_width, self.menu_height = 400, 400
        self.item_height = 30
        self.scroll_offset = 0
        self.button_center = (50, target_height - 50)
        self.button_radius = 30
        self.video_dir = video_dir
        self.folder_queue = folder_queue
        self.get_current_mode = get_current_mode_func
        self.get_waiting_for_transition = get_waiting_for_transition_func

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.show_suggestions:
                # Check if click inside menu
                if self.menu_x <= x <= self.menu_x + self.menu_width and self.menu_y <= y <= self.menu_y + self.menu_height:
                    # Calculate clicked item
                    item_y = y - self.menu_y - 30  # Subtract header height
                    if item_y >= 0:
                        item_index = (item_y // self.item_height) + self.scroll_offset
                        if 0 <= item_index < len(self.suggestions):
                            folder_name = self.suggestions[item_index]
                            print(f"📁 Selected suggestion: '{folder_name}'")
                            self.folder_queue.put(folder_name)
                            self.show_suggestions = False
                else:
                    # Click outside menu closes it
                    self.show_suggestions = False
            else:
                # Check circular button click (distance from center <= radius) only in root mode
                if self.get_current_mode() != "root":
                    return
                dist = np.sqrt((x - self.button_center[0])**2 + (y - self.button_center[1])**2)
                if dist <= self.button_radius and not self.get_waiting_for_transition():
                    print("🖱️ Suggestion button clicked! Showing overlay...")
                    # Load suggestions if not loaded
                    if not self.suggestions:
                        self.suggestions = [f for f in os.listdir(self.video_dir) if os.path.isdir(os.path.join(self.video_dir, f)) and f != 'root']
                        if not self.suggestions:
                            print("⚠️ No suggestion folders found!")
                            return
                    self.show_suggestions = True
                    self.scroll_offset = 0

        elif event == cv2.EVENT_MOUSEWHEEL:
            if self.show_suggestions and self.menu_x <= x <= self.menu_x + self.menu_width and self.menu_y <= y <= self.menu_y + self.menu_height:
                if flags > 0:  # Scroll up
                    self.scroll_offset = max(0, self.scroll_offset - 1)
                else:  # Scroll down
                    max_scroll = max(0, len(self.suggestions) - (self.menu_height - 30) // self.item_height)
                    self.scroll_offset = min(max_scroll, self.scroll_offset + 1)

    def draw_rounded_rect(self, img, rect_start, rect_end, color, thickness, radius):
        x, y = rect_start
        w, h = rect_end[0] - x, rect_end[1] - y
        # Draw straight lines
        cv2.line(img, (x + radius, y), (x + w - radius, y), color, thickness)
        cv2.line(img, (x + w, y + radius), (x + w, y + h - radius), color, thickness)
        cv2.line(img, (x + w - radius, y + h), (x + radius, y + h), color, thickness)
        cv2.line(img, (x, y + h - radius), (x, y + radius), color, thickness)
        # Draw arcs for corners
        cv2.ellipse(img, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, thickness)
        cv2.ellipse(img, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, thickness)

    def draw_filled_rounded_rect(self, img, rect_start, rect_end, color, radius, alpha=0.7):
        x1, y1 = rect_start
        x2, y2 = rect_end
        overlay = img.copy()
        # Main body
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        # Side bars
        cv2.rectangle(overlay, (x1, y1 + radius), (x1 + radius, y2 - radius), color, -1)
        cv2.rectangle(overlay, (x2 - radius, y1 + radius), (x2, y2 - radius), color, -1)
        # Corners
        cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
        # Blend
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def draw_suggestion_overlay(self, frame):
        # Draw semi-transparent white background with rounded corners
        self.draw_filled_rounded_rect(frame, (self.menu_x, self.menu_y), (self.menu_x + self.menu_width, self.menu_y + self.menu_height), (255, 255, 255), 20)
        
        # Draw border (rounded)
        self.draw_rounded_rect(frame, (self.menu_x, self.menu_y), (self.menu_x + self.menu_width, self.menu_y + self.menu_height), (0, 0, 0), 2, 20)
        
        # Draw header
        cv2.putText(frame, "Goi y cau hoi", (self.menu_x + 10, self.menu_y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw items (with clipping for scroll and truncation)
        visible_items = (self.menu_height - 30) // self.item_height
        max_text_length = 50  # Max characters before truncating
        for i in range(self.scroll_offset, min(self.scroll_offset + visible_items, len(self.suggestions))):
            item_text = self.suggestions[i]
            if len(item_text) > max_text_length:
                item_text = item_text[:max_text_length] + "..."
            y_pos = self.menu_y + 30 + (i - self.scroll_offset) * self.item_height + 20
            cv2.putText(frame, item_text, (self.menu_x + 10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Draw scrollbar if needed
        if len(self.suggestions) > visible_items:
            scrollbar_x = self.menu_x + self.menu_width - 10
            scrollbar_height = self.menu_height - 10
            thumb_height = max(10, (visible_items / len(self.suggestions)) * scrollbar_height)
            thumb_y = self.menu_y + 5 + (self.scroll_offset / (len(self.suggestions) - visible_items)) * (scrollbar_height - thumb_height)
            cv2.rectangle(frame, (scrollbar_x, self.menu_y + 5), (scrollbar_x + 5, self.menu_y + self.menu_height - 5), (200, 200, 200), -1)  # Light gray track
            cv2.rectangle(frame, (scrollbar_x, int(thumb_y)), (scrollbar_x + 5, int(thumb_y + thumb_height)), (100, 100, 100), -1)  # Dark gray thumb

    def draw_circular_button(self, frame):
        # Draw white circle
        cv2.circle(frame, self.button_center, self.button_radius, (255, 255, 255), -1)  # Filled white
        # Draw border
        cv2.circle(frame, self.button_center, self.button_radius, (0, 0, 0), 2)  # Black border
        # Draw text or icon (e.g., "G" for Gợi ý)
        cv2.putText(frame, "G", (self.button_center[0] - 10, self.button_center[1] + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)  # Black "G"