import cv2
import random
import time
from ultralytics import YOLO
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.draw_bbox import draw_boxes
from utils.draw_fps import draw_fps
from utils.fps_calculator import FpsCalculator

vessel_bbox_colors = [(34, 69, 34), (69, 34, 34), (69, 34, 102)]

class VesselDetector:
    def __init__(self, model_path="models/vessel_box.pt"):
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

    def detect_frame(self, frame, bbox_colors=vessel_bbox_colors, show_names=False, draw_bbox=True):
        """
        检测单帧图像

        Args:
            frame: 输入的图像帧(BGR格式)
            bbox_colors: 每个类别的颜色列表
            show_names: 是否显示标签
            draw_bbox: 是否绘制检测框

        Returns:
            processed_frame: 处理后的帧(带有检测框)
            detection_info: 检测结果信息
        """
        # 进行物体检测
        results = self.model.predict(frame, imgsz=320)

        bboxes_info = []

        # 解析结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                score = round(box.conf.item(), 2)  # 保留两位小数
                label = self.model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                xyxy = [x1, y1, x2, y2]
                bbox_info = {
                    'class_id': cls_id,
                    'label': label,
                    'score': score,
                    'box': xyxy
                }
                bboxes_info.append(bbox_info)

        # 计算FPS并显示
        avg_fps = self.fps_calculator.get()

        detection_info = {
            'bboxes': bboxes_info,
            'fps': avg_fps,
        }

        # 使用draw_boxes绘制
        processed_frame = draw_boxes(frame, detection_info, bbox_colors=bbox_colors, show_names=show_names, draw_bbox=draw_bbox)
        processed_frame = draw_fps(processed_frame, avg_fps)

        return processed_frame, detection_info
    
    def debug_image_predict(self, image_path):
        """
        临时debug方法：读取一张图片
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return
        # 使用detect_frame方法
        processed_frame, detection_info = self.detect_frame(img, show_names=True)
        cv2.imshow("BBox Detection", processed_frame)   
        cv2.waitKey(0)
        # cv2.imwrite("examples/results/vessel_bbox_test.png", processed_frame)
        print(detection_info)

# 示例用法
if __name__ == "__main__":
    detector = VesselDetector()
    detector.debug_image_predict("examples/test/gesture_test.jpg")