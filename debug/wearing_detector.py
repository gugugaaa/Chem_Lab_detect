import cv2
from ultralytics import YOLO
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.draw_bbox import draw_boxes
from utils.draw_fps import draw_fps
from utils.fps_caculator import FpsCalculator

"""
识别结果：
0-blue_glove
1-naked_hand
2-lab_coat
"""

# BGR格式颜色
wearing_bbox_colors = [
    (255, 204, 153),  # 浅蓝色
    (120, 180, 255),  # 肉色偏黄
    (220, 230, 245),  # 米白色
]

class WearingDetector:
    def __init__(self, model_path="models/wearing.pt"):
        """
        初始化穿戴检测器

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

    def detect_frame(self, frame, bbox_colors=wearing_bbox_colors, show_names=False, draw_bbox=True):
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
        results = self.model.predict(frame, imgsz=224)

        bboxes_info = []

        # 解析结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                score = round(box.conf.item(), 2)
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
                    cv2.imshow("Wearing Detection", processed_frame)
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
        # 使用detect_frame方法
        processed_frame, detection_info = self.detect_frame(img, show_names=True)
        cv2.imshow("Wearing Detection", processed_frame)   
        cv2.waitKey(0)
        print(detection_info)

# 示例用法
if __name__ == "__main__":
    detector = WearingDetector()
    detector.debug_image_predict("examples/safety_test.png")