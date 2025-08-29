import cv2

class MainPipeline:
    def __init__(
        self,
        video_source,
        safety_detector,
        vessel_detector,
        gesture_detector,
        action_scorer,
        safety_interval=5,
        vessel_interval=1,
        gesture_interval=1,
        output_callback=None
    ):
        
        self.video_source = video_source
        self.safety_detector = safety_detector
        self.vessel_detector = vessel_detector
        self.gesture_detector = gesture_detector
        self.action_scorer = action_scorer
        self.safety_interval = safety_interval
        self.vessel_interval = vessel_interval
        self.gesture_interval = gesture_interval
        self.output_callback = output_callback

        self.last_safety_info = None
        self.last_vessel_info = None
        self.last_gesture_info = None
        self.frame_idx = 0

    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def register_output_callback(self, callback):
        self.output_callback = callback

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_idx += 1

                # 安全检测抽帧
                if self.frame_idx % self.safety_interval == 0:
                    _, safety_info = self.safety_detector.detect_frame(frame)
                    self.last_safety_info = safety_info

                # 容器检测抽帧
                if self.frame_idx % self.vessel_interval == 0:
                    _, vessel_info = self.vessel_detector.detect_frame(frame)
                    self.last_vessel_info = vessel_info

                # 手势检测抽帧
                if self.frame_idx % self.gesture_interval == 0:
                    _, gesture_info = self.gesture_detector.detect_frame(frame)
                    self.last_gesture_info = gesture_info

                # 评分器调用（需有新容器和手势检测结果）
                if self.last_vessel_info and self.last_gesture_info:
                    score_result = self.action_scorer.score_frame(self.last_vessel_info, self.last_gesture_info)
                else:
                    score_result = []

                # 输出
                result = {
                    "frame_idx": self.frame_idx,
                    "safety_info": self.last_safety_info,
                    "vessel_info": self.last_vessel_info,
                    "gesture_info": self.last_gesture_info,
                    "score_result": score_result
                }
                if self.output_callback:
                    self.output_callback(result)
        finally:
            cap.release()