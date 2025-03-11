import cv2
import random
import time
from ultralytics import YOLO

from utils.draw_bbox import draw_detection_boxes
from utils.draw_fps import draw_fps
from utils.fps_caculator import FpsCalculator

class VesselDetector:
    def __init__(self, model_path=r"models\vessels-bbox-nano.pt"):
        """
        初始化容器检测器
        
        Args:
            model_path: 模型路径
        """
        # 加载模型
        self.model = YOLO(model_path, verbose=False)
        # 创建FPS计算器
        self.fps_calculator = FpsCalculator(buffer_len=100)
        # 字典来存储类别和颜色的映射
        self.category_colors = {}
    
    def __del__(self):
        """析构函数，释放资源"""
        if hasattr(self, 'model'):
            del self.model
    
    def get_random_color(self):
        """生成随机颜色"""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def detect_frame(self, frame):
        """
        检测单帧图像
        
        Args:
            frame: 输入的图像帧(BGR格式)
            
        Returns:
            processed_frame: 处理后的帧(带有检测框)
            detection_info: 检测结果信息
        """
        # 进行物体检测
        results = self.model(frame)
        
        # 复制一份用于绘制
        processed_frame = frame.copy()
        detections = []
        
        # 解析结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                score = box.conf.item()
                label = self.model.names[cls_id]
                
                # 获取颜色，如果类别没有颜色则生成一个新的
                if label not in self.category_colors:
                    self.category_colors[label] = self.get_random_color()
                color = self.category_colors[label]
                
                # 获取边框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # 创建类似于MediaPipe的检测结果格式
                detection = {
                    'bounding_box': {
                        'origin_x': x1,
                        'origin_y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1
                    },
                    'categories': [{
                        'category_name': label,
                        'score': score
                    }]
                }
                detections.append(detection)
                
                # 在原图上绘制
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(processed_frame, f'{label}: {score:.2f}', (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 计算FPS并显示
        avg_fps = self.fps_calculator.get()
        processed_frame = draw_fps(processed_frame, avg_fps)
        
        # 返回处理后的帧和检测信息
        detection_info = {
            'detections': detections,
            'fps': avg_fps,
            'detected_labels': [self.model.names[int(box.cls.item())] for result in results for box in result.boxes]
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
                    cv2.imshow("Vessel Detection", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                        break
                        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

# 示例用法
if __name__ == "__main__":
    detector = VesselDetector()
    detector.process_video(0)  # 使用摄像头索引0