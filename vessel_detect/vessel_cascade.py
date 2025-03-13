import cv2
import numpy as np
from vessel_detect.vessel_bbox import VesselDetector
from vessel_detect.vessel_pose import VesselPoseDetector
from utils.fps_caculator import FpsCalculator
from utils.draw_fps import draw_fps
from utils.draw_keypoints import draw_keypoints

class VesselCascadeDetector:
    def __init__(
        self, 
        bbox_model_path=r"models\vessels-bbox-nano.pt",
        pose_model_path=r"models\vessels-pose-nano.pt", 
        bbox_conf=0.5,
        pose_conf=0.7,
        margin=5
    ):
        """
        初始化容器级联检测器
        
        Args:
            bbox_model_path: 边界框检测模型路径
            pose_model_path: 姿态检测模型路径
            bbox_conf: 边界框检测置信度阈值
            pose_conf: 姿态检测置信度阈值
            margin: 裁剪边界框时的边距
        """
        # 初始化两个检测器
        self.bbox_detector = VesselDetector(model_path=bbox_model_path)
        self.pose_detector = VesselPoseDetector(model_path=pose_model_path, conf=pose_conf)
        self.margin = margin
        self.fps_calculator = FpsCalculator(buffer_len=100)
    
    def __del__(self):
        """析构函数，释放资源"""
        if hasattr(self, 'bbox_detector'):
            del self.bbox_detector
        if hasattr(self, 'pose_detector'):
            del self.pose_detector
    
    def detect_frame(self, frame):
        """
        级联检测单帧图像
        
        Args:
            frame: 输入的图像帧(BGR格式)
            
        Returns:
            processed_frame: 处理后的帧(带有检测框和关键点)
            cascade_info: 级联检测结果信息
        """
        # 步骤1: 进行边界框检测
        _, bbox_info = self.bbox_detector.detect_frame(frame)
        
        # 复制一份用于绘制
        processed_frame = frame.copy()
        cascade_results = []
        
        # 步骤2: 处理每个检测到的容器
        for detection in bbox_info['detections']:
            # 获取边界框信息
            bbox = detection['bounding_box']
            x = bbox['origin_x']
            y = bbox['origin_y']
            w = bbox['width']
            h = bbox['height']
            
            # 添加边距
            x1 = max(0, x - self.margin)
            y1 = max(0, y - self.margin)
            x2 = min(frame.shape[1], x + w + self.margin)
            y2 = min(frame.shape[0], y + h + self.margin)
            
            # 裁剪容器区域
            cropped_region = frame[y1:y2, x1:x2]
            
            # 确保裁剪区域有效
            if cropped_region.size > 0:
                # 步骤3: 对裁剪区域进行姿态检测
                results = self.pose_detector.detect(cropped_region)
                
                # 初始化姿态信息字典
                pose_info = {'poses': []}
                
                # 处理检测结果
                for result in results:
                    if result.keypoints is not None:
                        # 提取关键点信息
                        keypoints_xy_data = result.keypoints.xy
                        keypoints_conf_data = result.keypoints.conf if hasattr(result.keypoints, 'conf') else None
                        
                        # 准备绘制关键点所需的数据
                        keypoints_list = []
                        keypoint_conf_list = []
                        classes_list = []
                        box_conf_list = []
                        
                        for i in range(len(keypoints_xy_data)):
                            if i < len(result.boxes.cls):
                                cls_id = int(result.boxes.cls[i].item())
                                score = result.boxes.conf[i].item()
                                
                                # 提取当前对象的关键点
                                kpts = []
                                confs = []
                                
                                # 将坐标从裁剪区域映射回原图
                                for j, xy in enumerate(keypoints_xy_data[i]):
                                    orig_x = xy[0].item() + x1
                                    orig_y = xy[1].item() + y1
                                    kpts.append([orig_x, orig_y])
                                    
                                    if keypoints_conf_data is not None:
                                        confs.append(keypoints_conf_data[i][j].item())
                                    else:
                                        confs.append(1.0)
                                
                                # 收集处理后的数据
                                keypoints_list.append(kpts)
                                keypoint_conf_list.append(confs)
                                classes_list.append(cls_id)
                                box_conf_list.append(score)
                                
                                # 为每个检测到的物体构建关键点信息
                                keypoints_data = []
                                for j, (xy, conf) in enumerate(zip(kpts, confs)):
                                    keypoints_data.append({
                                        'x': xy[0],
                                        'y': xy[1],
                                        'confidence': conf,
                                        'name': f'keypoint_{j}'
                                    })
                                
                                # 添加到姿态信息中
                                pose_info['poses'].append({
                                    'class_id': cls_id,
                                    'label': self.pose_detector.model.names[cls_id],
                                    'score': score,
                                    'keypoints': keypoints_data
                                })
                        
                        # 在原图上绘制关键点
                        if keypoints_list:
                            processed_frame = draw_keypoints(
                                processed_frame,
                                np.array(keypoints_list),
                                np.array(keypoint_conf_list),
                                np.array(classes_list),
                                np.array(box_conf_list)
                            )
                
                # 收集检测信息
                result = {
                    'bbox': {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'category': detection['categories'][0]['category_name'],
                        'score': detection['categories'][0]['score']
                    },
                    'pose': pose_info
                }
                cascade_results.append(result)
        
        # 计算FPS并显示
        avg_fps = self.fps_calculator.get()
        processed_frame = draw_fps(processed_frame, avg_fps)
        
        # 返回处理后的帧和检测信息
        cascade_info = {
            'results': cascade_results,
            'fps': avg_fps,
            'bbox_detected_labels': bbox_info['detected_labels']
        }
        
        return processed_frame, cascade_info
    
    def process_video(self, video_source=0, display=True):
        """
        级联处理视频流
        
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
                
                processed_frame, cascade_info = self.detect_frame(frame)
                
                if display:
                    cv2.imshow("Vessel Cascade Detection", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                        break
                        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

# 示例用法
if __name__ == "__main__":
    cascade_detector = VesselCascadeDetector()
    cascade_detector.process_video(0)  # 使用摄像头索引0
