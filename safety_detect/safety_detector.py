import cv2
import time
import numpy as np

from safety_detect.glove_detector import GloveDetector
from safety_detect.coat_detector import CoatDetector
from utils.fps_caculator import FpsCalculator
from utils.draw_fps import draw_fps

class SafetyDetector:
    def __init__(self, 
                 glove_model_path=r"models\glove_detect.tflite", 
                 coat_model_path=r"models\coat_detect.tflite",
                 glove_score_threshold=0.3,
                 coat_score_threshold=0.85,
                 max_results=3):
        """
        初始化安全检测器
        
        Args:
            glove_model_path: 手套检测模型路径
            coat_model_path: 实验服检测模型路径
            score_threshold: 检测阈值
            max_results: 最大检测结果数
        """
        # 初始化子检测器
        self.glove_detector = GloveDetector(
            model_path=glove_model_path, 
            score_threshold=glove_score_threshold,
            max_results=max_results
        )
        
        self.coat_detector = CoatDetector(
            model_path=coat_model_path,
            score_threshold=coat_score_threshold,
            max_results=max_results
        )
        
        self.fps_calculator = FpsCalculator(buffer_len=100)
    
    def detect_frame(self, frame):
        """
        检测单帧图像，同时进行手套和实验服检测
        
        Args:
            frame: 输入的图像帧(BGR格式)
            
        Returns:
            processed_frame: 处理后的帧(带有检测框和安全状态)
            safety_info: 安全状态信息字典
        """
        # 分别进行手套和实验服检测
        glove_frame, glove_info = self.glove_detector.detect_frame(frame.copy())
        coat_frame, coat_info = self.coat_detector.detect_frame(frame.copy())
        
        # 合并结果到原始帧上
        processed_frame = frame.copy()
        
        # 从glove_frame和coat_frame提取检测框并绘制到processed_frame上
        for detection in glove_info['detections']:
            bbox = detection.bounding_box
            x, y = bbox.origin_x, bbox.origin_y
            w, h = bbox.width, bbox.height
            score = detection.categories[0].score
            label = detection.categories[0].category_name
            
            # 为不同类别设置不同颜色
            color = (0, 255, 0) if label == "blue_glove" else (0, 0, 255)  # 绿色表示手套，红色表示裸手
            
            # 绘制边界框
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)
            
            # 绘制标签和置信度
            text = f"{label}: {score:.2f}"
            cv2.putText(processed_frame, text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        for detection in coat_info['detections']:
            bbox = detection.bounding_box
            x, y = bbox.origin_x, bbox.origin_y
            w, h = bbox.width, bbox.height
            score = detection.categories[0].score
            label = detection.categories[0].category_name
            
            # 为不同类别设置不同颜色
            color = (0, 255, 0) if label == "lab_coat" else (0, 0, 255)  # 绿色表示实验服，红色表示无实验服
            
            # 绘制边界框
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)
            
            # 绘制标签和置信度
            text = f"{label}: {score:.2f}"
            cv2.putText(processed_frame, text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 判断安全状态
        is_safe = coat_info['coat_detected'] and not glove_info['bare_hand_detected']
        safety_status = "SAFE" if is_safe else "UNSAFE"
        safety_color = (0, 255, 0) if is_safe else (0, 0, 255)  # 绿色表示安全，红色表示不安全
        
        # 在画面上显示安全状态
        cv2.putText(
            processed_frame, 
            f"Status: {safety_status}", 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            safety_color, 
            2
        )
        
        # 显示问题
        if not coat_info['coat_detected']:
            cv2.putText(processed_frame, "No Lab Coat!", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        if glove_info['bare_hand_detected']:
            cv2.putText(processed_frame, "Bare Hands Detected!", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 计算并显示FPS
        avg_fps = self.fps_calculator.get()
        processed_frame = draw_fps(processed_frame, avg_fps)
        
        # 整合安全信息
        safety_info = {
            'is_safe': is_safe,
            'glove_info': glove_info,
            'coat_info': coat_info,
            'fps': avg_fps
        }
        
        return processed_frame, safety_info
    
    def process_video(self, video_source=0, display=True, save_path=None):
        """
        处理视频流
        
        Args:
            video_source: 视频源，可以是摄像头索引或视频文件路径
            display: 是否显示处理结果
            save_path: 如果不为None，将处理后的视频保存到指定路径
            
        Returns:
            None
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return
        
        # 视频写入器
        writer = None
        if save_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, safety_info = self.detect_frame(frame)
                
                if writer:
                    writer.write(processed_frame)
                    
                if display:
                    cv2.imshow("Safety Detection", processed_frame)
                    if cv2.waitKey(15) & 0xFF == 27:  # 按ESC退出
                        break
                        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

# 示例用法
if __name__ == "__main__":
    detector = SafetyDetector()
    detector.process_video(0)  # 使用摄像头索引0
