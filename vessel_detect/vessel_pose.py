import cv2
import sys
import os
from ultralytics import YOLO

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.draw_keypoints import draw_keypoints
from utils.draw_fps import draw_fps
from utils.fps_caculator import FpsCalculator

class VesselPoseDetector:
    def __init__(self, model_path=r"models\vesse-pose-nano.pt", conf=0.5):
        """
        初始化容器姿态检测器
        
        Args:
            model_path: 模型路径
            conf: 置信度阈值
        """
        self.model = YOLO(model_path, verbose=False)
        self.conf = conf
        # 创建FPS计算器
        self.fps_calculator = FpsCalculator(buffer_len=100)
    
    def __del__(self):
        """析构函数，释放资源"""
        if hasattr(self, 'model'):
            del self.model

    def detect(self, frame):
        """
        检测图像中的容器姿态
        返回：原始检测结果
        """
        results = self.model.predict(frame, conf=self.conf)
        return results
    
    def detect_frame(self, frame):
        """
        检测单帧图像
        
        Args:
            frame: 输入的图像帧(BGR格式)
            
        Returns:
            processed_frame: 处理后的帧(带有关键点)
            detection_info: 检测结果信息
        """
        # 进行姿态检测
        results = self.detect(frame)
        
        # 复制一份用于绘制
        processed_frame = frame.copy()
        poses_info = []
        
        for result in results:
            # 使用自定义绘制函数绘制关键点
            if result.keypoints is not None:
                processed_frame = draw_keypoints(
                    processed_frame, 
                    result.keypoints.xy, 
                    result.keypoints.conf, 
                    result.boxes.cls, 
                    result.boxes.conf
                )
                
                # 提取关键点信息
                # 确保关键点数据存在且可迭代
                if result.keypoints is not None and hasattr(result.keypoints, 'xy') and result.keypoints.xy is not None:
                    keypoints_xy_data = result.keypoints.xy
                    keypoints_conf_data = result.keypoints.conf if hasattr(result.keypoints, 'conf') and result.keypoints.conf is not None else None
                    
                    for i in range(len(keypoints_xy_data)):
                        if i < len(result.boxes.cls):
                            cls_id = int(result.boxes.cls[i].item())
                            score = result.boxes.conf[i].item()
                            label = self.model.names[cls_id]
                            
                            # 构建关键点信息字典
                            keypoints_data = []
                            if keypoints_conf_data is not None:
                                for j, (xy, conf) in enumerate(zip(keypoints_xy_data[i], keypoints_conf_data[i])):
                                    keypoints_data.append({
                                        'x': xy[0].item(),
                                        'y': xy[1].item(),
                                        'confidence': conf.item(),
                                        'name': f'keypoint_{j}'  # 可以根据实际情况命名
                                    })
                            else:
                                for j, xy in enumerate(keypoints_xy_data[i]):
                                    keypoints_data.append({
                                        'x': xy[0].item(),
                                        'y': xy[1].item(),
                                        'confidence': None,
                                        'name': f'keypoint_{j}'
                                    })
                            
                            # 创建姿态信息
                            pose_info = {
                                'class_id': cls_id,
                                'label': label,
                                'score': score,
                                'keypoints': keypoints_data
                            }
                            poses_info.append(pose_info)
        
        # 计算FPS并显示
        avg_fps = self.fps_calculator.get()
        processed_frame = draw_fps(processed_frame, avg_fps)
        
        # 返回处理后的帧和检测信息
        detection_info = {
            'poses': poses_info,
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
                    cv2.imshow("Vessel Pose Detection", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                        break
                        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()


# 示例用法
if __name__ == "__main__":
    detector = VesselPoseDetector()
    detector.process_video(1)  # 使用摄像头索引1