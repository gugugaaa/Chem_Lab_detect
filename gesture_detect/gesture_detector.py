# STEP 1: Import the necessary modules.
import mediapipe as mp
import cv2
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.draw_hand import draw_landmarks
from utils.fps_caculator import FpsCalculator
from utils.draw_fps import draw_fps

# BUG: Incomplete detection_info
# new mediapipe hand landmarker has no api to fetch hand detection/presence/visibility score
# using handedness score as score. cannot provide keypoint confidence.

class GestureDetector:
    def __init__(self, model_path='models/hand.task', num_hands=2, 
                 min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, 
                 min_tracking_confidence=0.5):
        """
        初始化手势检测器
        
        Args:
            model_path: 模型路径
            num_hands: 检测手的数量
            min_hand_detection_confidence: 手部检测置信度阈值
            min_hand_presence_confidence: 手部存在置信度阈值
            min_tracking_confidence: 手部跟踪置信度阈值
        """
        # STEP 2: 创建一个HandLandmarker对象
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.fps_calculator = FpsCalculator(buffer_len=100)  # 创建FPS计算器实例
    
    def __del__(self):
        """析构函数，释放检测器资源"""
        if hasattr(self, 'detector'):
            del self.detector
    
    def detect_frame(self, frame):
        """
        检测单帧图像中的手势
        
        Args:
            frame: 输入的图像帧(BGR格式)
            
        Returns:
            processed_frame: 处理后的帧(带有手部关键点)
            detection_info: 检测结果信息
        """
        # 关键点名称列表
        hand_landmark_names = [
            "wrist",
            "thumb_cmc",
            "thumb_mcp",
            "thumb_ip",
            "thumb_tip",
            "index_finger_mcp",
            "index_finger_pip",
            "index_finger_dip",
            "index_finger_tip",
            "middle_finger_mcp",
            "middle_finger_pip",
            "middle_finger_dip",
            "middle_finger_tip",
            "ring_finger_mcp",
            "ring_finger_pip",
            "ring_finger_dip",
            "ring_finger_tip",
            "pinky_mcp",
            "pinky_pip",
            "pinky_dip",
            "pinky_tip"
        ]
        # 预处理图像，to_rgb=True可以检测肉色/白色手套，蓝色手套应该使用False（BGR里面蓝色就是肉色）
        pre_process, scale = self._pre_process_image(frame, 288, to_rgb=False)
        # pre_process 已经是 RGB，无需再转换
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=pre_process)
        
        # 检测手势
        detection_result = self.detector.detect(mp_image)
        
        # 用原图尺寸的副本用于绘制
        processed_frame = frame.copy()

        hands_info = []
        if detection_result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                keypoints = []
                landmark_point = []
                for lm_idx, landmark in enumerate(hand_landmarks):
                    # 还原到原图尺寸
                    x = int(round(landmark.x * pre_process.shape[1] / scale))
                    y = int(round(landmark.y * pre_process.shape[0] / scale))
                    # 移除presence和visibility相关代码
                    keypoints.append({
                        "x": x,
                        "y": y,
                        "name": hand_landmark_names[lm_idx]
                    })
                    landmark_point.append((x, y))
                # handedness
                handedness = detection_result.handedness[hand_idx][0] if hasattr(detection_result, "handedness") else None
                if handedness is not None:
                    chirality = "left" if handedness.index == 0 else "right"
                    score = round(handedness.score, 2)
                else:
                    chirality = None
                    score = None
                hands_info.append({
                    "chirality": chirality,
                    "score": score,
                    "keypoints": keypoints
                })
                # 绘制关键点（在原图尺寸上）
                processed_frame = draw_landmarks(processed_frame, landmark_point)

        avg_fps = self.fps_calculator.get()
        processed_frame = draw_fps(processed_frame, avg_fps)

        detection_info = {
            "hands": hands_info,
            "fps": avg_fps,
            "hand_detected": len(hands_info)
        }
        return processed_frame, detection_info
    
    def _pre_process_image(self, image, imgsz=320, to_rgb=True):
        """
        等比例缩放到长边==imgsz
        然后变换到BGR空间，这样蓝色的手套就会被映射为肉色
        可选是否输出RGB格式（适用于不同模型）
        最后做高斯模糊
        Args:
            image: 输入图像
            imgsz: 缩放后长边尺寸
            to_rgb: 是否输出RGB格式（否则为BGR）
        Returns:
            处理后的图像, 缩放比例
        """
        h, w = image.shape[:2]
        if h > w:
            scale = imgsz / float(h)
            new_h = imgsz
            new_w = int(w * scale)
        else:
            scale = imgsz / float(w)
            new_w = imgsz
            new_h = int(h * scale)
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # 若不是BGR则转为BGR
        if len(resized_img.shape) == 2 or resized_img.shape[2] == 1:
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
        # 高斯模糊
        blurred_img = cv2.GaussianBlur(resized_img, (5, 5), 0)
        # 可选转换为RGB
        if to_rgb:
            out_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)
        else:
            out_img = blurred_img
        return out_img, scale

    
    def process_video(self, video_source=0, display=True):
        """
        处理视频流
        
        Args:
            video_source: 视频源，可以是摄像头索引或视频文件路径
            display: 是否显示处理结果
            
        Returns:
            None
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return
            
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, detection_info = self.detect_frame(frame)
                
                if display:
                    cv2.imshow("Hand Gesture Detection", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                        break
                        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()


    def debug_image_predict(self, image_path):
        """
        临时debug方法：读取一张图片
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return
        processed_frame, detection_info = self.detect_frame(img)
        cv2.imshow("Gesture Detection", processed_frame)
        cv2.waitKey(0)
        # cv2.imwrite("examples/results/eazy_hand.png", processed_frame)
        # print(detection_info)


# 示例用法
if __name__ == "__main__":
    detector = GestureDetector()
    detector.debug_image_predict("examples/test/safety_test.png")