import collections
import statistics

class LabelSmoother:
    def __init__(self, window_size=10):
        self.emotion_buffer = collections.deque(maxlen=window_size)
        self.gesture_buffer = collections.deque(maxlen=window_size)

    def update(self, raw_emotion, raw_gesture):
        """
        Nhận nhãn thô từ AI, đẩy vào hàng đợi và trả về nhãn đã được làm mượt.
        """
        # 1. Đẩy dữ liệu mới vào cuối hàng đợi
        self.emotion_buffer.append(raw_emotion)
        self.gesture_buffer.append(raw_gesture)
        
        # 2. Dùng thuật toán Bầu chọn đa số (Voting)
        smoothed_emotion = statistics.mode(self.emotion_buffer)
        smoothed_gesture = statistics.mode(self.gesture_buffer)
        
        return smoothed_emotion, smoothed_gesture