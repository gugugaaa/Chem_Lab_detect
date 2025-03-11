# STEP 1: Import the necessary modules.
import mediapipe as mp
import cv2
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from utils.draw_hand import draw_landmarks
from utils.fps_caculator import FpsCalculator
from utils.draw_fps import draw_fps

class GestureDetector:
    def __init__(self, model_path=r'models\hand_landmarker.task', num_hands=2, 
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
        # 转换BGR为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转换为MediaPipe图像格式
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 检测手势
        detection_result = self.detector.detect(mp_image)
        
        # 转换回BGR用于显示
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # 绘制手部关键点和连接线
        hand_landmarks_data = []
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                landmark_point = []
                for landmark in hand_landmarks:
                    x = int(landmark.x * processed_frame.shape[1])
                    y = int(landmark.y * processed_frame.shape[0])
                    landmark_point.append((x, y))
                
                # 调用draw_landmarks方法
                processed_frame = draw_landmarks(processed_frame, landmark_point)
                hand_landmarks_data.append(landmark_point)
        
        # 计算FPS并显示
        avg_fps = self.fps_calculator.get()
        processed_frame = draw_fps(processed_frame, avg_fps)
        
        # 返回处理后的帧和检测信息
        detection_info = {
            'hand_landmarks': hand_landmarks_data,
            'handedness': detection_result.handedness if hasattr(detection_result, 'handedness') else None,
            'hands_detected': len(detection_result.hand_landmarks) if detection_result.hand_landmarks else 0,
            'fps': avg_fps
        }
        
        return processed_frame, detection_info
    
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

# 示例用法
if __name__ == "__main__":
    detector = GestureDetector()
    detector.process_video(0)  # 使用摄像头索引0
