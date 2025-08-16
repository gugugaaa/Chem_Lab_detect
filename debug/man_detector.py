import cv2
import sys
import os
from ultralytics import YOLO

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.draw_keypoints import draw_keypoints
from utils.draw_fps import draw_fps
from utils.fps_caculator import FpsCalculator

man_keypoint_names = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
    'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle'
]

# 肉色-粉色渐变，左右对称（BGR顺序）
man_keypoint_colors = [(189,224,255),(203,192,255),(203,192,255),(220,184,255),(220,184,255),(233,174,255),(233,174,255),(245,170,255),(245,170,255),(255,182,193),(255,182,193),(255,192,203),(255,192,203),(255,204,229),(255,204,229),(255,228,240),(255,228,240)]

class ManPoseDetector:
    def __init__(self, model_path="models/man_pose.pt", conf=0.5, show_kpt_names=False):
        """
        初始化人体姿态检测器
        
        Args:
            model_path: 模型路径
            conf: 置信度阈值，会导致result.boxes中conf小的被过滤，但keypoints是完整的即使不可见（conf=0）
        """
        self.model = YOLO(model_path, verbose=False)
        self.conf = conf
        # 是否在关键点旁显示名称
        self.show_kpt_names = show_kpt_names
        # 创建FPS计算器
        self.fps_calculator = FpsCalculator(buffer_len=100)
    
    def __del__(self):
        """析构函数，释放资源"""
        if hasattr(self, 'model'):
            del self.model

    def detect(self, frame):
        """
        检测图像中的人的姿态
        返回：原始检测结果
        """
        results = self.model.predict(frame, conf=self.conf, imgsz=224)
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
            # result是这张图片所有结果，要遍历boxes和keypoints才能拿到单个目标
            if result.keypoints is not None and len(result.boxes) > 0:
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    xyxy = [int(round(coord)) for coord in box.xyxy[0].tolist()]  # 获取第i个框的[x1, y1, x2, y2]，取整
                    keypoints_xy = [[int(round(x)), int(round(y))] for x, y in result.keypoints.xy[i].tolist()]  # 关键点坐标取整

                    # 提取关键点信息
                    cls_id = int(box.cls.item())
                    score = round(box.conf.item(), 2)  # 置信度保留两位小数
                    label = self.model.names[cls_id]
                    keypoints_conf_data = result.keypoints.conf[i] if hasattr(result.keypoints, 'conf') and result.keypoints.conf is not None else None
                    keypoints_data = []
                    if keypoints_conf_data is not None:
                        for j, (xy, conf) in enumerate(zip(keypoints_xy, keypoints_conf_data)):
                            name = man_keypoint_names[j] if j < len(man_keypoint_names) else f'keypoint_{j}'
                            keypoints_data.append({
                                'x': xy[0],
                                'y': xy[1],
                                'confidence': round(conf.item(), 2),  # 置信度保留两位小数
                                'name': name
                            })
                    else:
                        for j, xy in enumerate(keypoints_xy):
                            name = man_keypoint_names[j] if j < len(man_keypoint_names) else f'keypoint_{j}'
                            keypoints_data.append({
                                'x': xy[0],
                                'y': xy[1],
                                'confidence': None,
                                'name': name
                            })
                    pose_info = {
                        'class_id': cls_id,
                        'label': label,
                        'score': score,
                        'keypoints': keypoints_data,
                        'box': xyxy
                    }
                    poses_info.append(pose_info)
        
        # 计算FPS并显示
        avg_fps = self.fps_calculator.get()
        processed_frame = draw_fps(processed_frame, avg_fps)

        # 构建detection_info，fps已知
        detection_info = {
            'poses': poses_info,
            'fps': avg_fps,
            'man_detected': len(poses_info)
        }
        
        # 使用新的绘制函数绘制关键点（遍历完成后统一绘制）
        processed_frame = draw_keypoints(
            processed_frame,
            detection_info,
            keypoint_colors=man_keypoint_colors,
            show_names=self.show_kpt_names,
            draw_bbox=True  # 可以根据需要设置
        )
        
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

    def debug_image_predict(self, image_path):
        """
        临时debug方法：读取一张图片
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return
        # 使用detect_frame方法
        processed_frame, detection_info = self.detect_frame(img)
        cv2.imshow("Pose Detection", processed_frame)   
        cv2.waitKey(0)
        print(detection_info)


# 示例用法
if __name__ == "__main__":
    detector = ManPoseDetector()
    # 示例：请替换为你的图片路径
    detector.debug_image_predict("examples/safety_test.png")