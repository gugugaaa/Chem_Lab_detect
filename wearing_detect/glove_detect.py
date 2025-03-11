import cv2
import mediapipe as mp
import numpy as np
import time

from utils.draw_bbox import draw_detection_boxes
from utils.draw_fps import draw_fps
from utils.fps_caculator import FpsCalculator  # 导入FPS计算器

# Import MediaPipe task modules
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class GloveDetector:
    def __init__(self, model_path=r"models\glove_detect.tflite", score_threshold=0.3, max_results=3):
        """
        初始化手套检测器
        
        Args:
            model_path: 模型路径
            score_threshold: 检测阈值
            max_results: 最大检测结果数
        """
        # Configure object detector options
        self.BaseOptions = mp.tasks.BaseOptions
        self.ObjectDetector = mp.tasks.vision.ObjectDetector
        self.ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        
        self.options = self.ObjectDetectorOptions(
            base_options=self.BaseOptions(model_asset_path=model_path),
            max_results=max_results,
            running_mode=self.VisionRunningMode.VIDEO,
            score_threshold=score_threshold
        )
        
        self.detector = self.ObjectDetector.create_from_options(self.options)
        self.fps_calculator = FpsCalculator(buffer_len=100)  # 创建FPS计算器实例
    
    def __del__(self):
        """析构函数，释放检测器资源"""
        if hasattr(self, 'detector'):
            self.detector.close()
    
    def detect_frame(self, frame):
        """
        检测单帧图像
        
        Args:
            frame: 输入的图像帧(BGR格式)
            
        Returns:
            processed_frame: 处理后的帧(带有检测框)
            detection_info: 检测结果字典，包含glove_detected, bare_hand_detected等信息
        """
        # 转换颜色并检测
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect_for_video(mp_image, int(time.time() * 1000))

        # 检查是否检测到手套和裸露的手
        glove_detected = any(detection.categories[0].category_name == "blue_glove" for detection in detection_result.detections)
        bare_hand_detected = any(detection.categories[0].category_name == "naked_hand" for detection in detection_result.detections)
        
        # 使用工具函数绘制检测框
        processed_frame = draw_detection_boxes(frame, detection_result.detections)

        # 计算FPS并显示
        avg_fps = self.fps_calculator.get()  # 使用正确的API计算FPS
        
        # 使用工具函数绘制FPS
        processed_frame = draw_fps(processed_frame, avg_fps)

        # 返回处理后的帧和检测信息
        detection_info = {
            'glove_detected': glove_detected,
            'bare_hand_detected': bare_hand_detected,
            'detections': detection_result.detections,
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
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, detection_info = self.detect_frame(frame)
                
                if display:
                    cv2.imshow("Object Detection", processed_frame)
                    if cv2.waitKey(15) & 0xFF == 27:  # 按ESC退出
                        break
                        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

# 示例用法
if __name__ == "__main__":
    detector = GloveDetector()
    detector.process_video(0)  # 使用摄像头索引0