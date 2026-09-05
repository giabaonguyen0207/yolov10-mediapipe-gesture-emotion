import collections

class LabelSmoother:
    def __init__(self, window_size=7, decay=0.85):
        """
        :param window_size: Kích thước hàng đợi lưu nhãn (nhỏ hơn giúp phản hồi nhanh hơn, 5-7 là tối ưu)
        :param decay: Trọng số suy giảm theo thời gian cho các frame cũ (0.8 - 0.9)
        """
        self.window_size = window_size
        self.decay = decay
        self.emotion_buffer = collections.deque(maxlen=window_size)
        self.gesture_buffer = collections.deque(maxlen=window_size)

    def update(self, raw_emotion, raw_gesture, emotion_conf=1.0, gesture_conf=1.0):
        """
        Nhận nhãn thô từ AI kèm độ tin cậy (confidence), trả về nhãn đã được làm mượt.
        """
        self.emotion_buffer.append((raw_emotion, float(emotion_conf)))
        self.gesture_buffer.append((raw_gesture, float(gesture_conf)))
        
        return (
            self._get_weighted_label(self.emotion_buffer),
            self._get_weighted_label(self.gesture_buffer)
        )

    def _get_weighted_label(self, buffer):
        if not buffer:
            return "Đang chờ..."

        scores = {}
        n = len(buffer)
        
        for idx, (label, conf) in enumerate(buffer):
            if label is None or label == "None" or label == "Đang chờ..." or conf <= 0:
                continue
            
            # Trọng số suy giảm theo thời gian: Frame mới hơn (idx gần n-1) có trọng số lớn hơn
            weight = (self.decay ** (n - 1 - idx)) * conf
            scores[label] = scores.get(label, 0.0) + weight

        if not scores:
            return "None"

        return max(scores, key=scores.get)
