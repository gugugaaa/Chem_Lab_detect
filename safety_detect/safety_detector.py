import cv2
import sys
import os
from ultralytics import YOLO

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.draw_fps import draw_fps
from utils.fps_caculator import FpsCalculator
from utils.draw_safety import draw_safety

man_keypoint_names = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
    'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle'
]

class SafetyDetector:
    def __init__(self, man_model_path="models/man_pose.pt", wearing_model_path="models/wearing.pt", conf=0.5, show_kpt_names=False):
        """
        初始化安全检测器
        
        Args:
            man_model_path: 人体姿态模型路径
            wearing_model_path: 穿戴检测模型路径
            conf: 置信度阈值
            show_kpt_names: 是否在关键点旁显示名称
        """
        self.man_model = YOLO(man_model_path, verbose=False)
        self.wearing_model = YOLO(wearing_model_path, verbose=False)
        self.conf = conf
        self.show_kpt_names = show_kpt_names
        self.fps_calculator = FpsCalculator(buffer_len=100)
    
    def __del__(self):
        """析构函数，释放资源"""
        if hasattr(self, 'man_model'):
            del self.man_model
        if hasattr(self, 'wearing_model'):
            del self.wearing_model

    def detect_frame(self, frame):
        """
        检测单帧图像
        
        Args:
            frame: 输入的图像帧(BGR格式)
            
        Returns:
            processed_frame: 处理后的帧(带有关键点和检测框)
            detection_info: 检测结果信息
        """
        # 进行人体姿态检测
        man_results = self.man_model.predict(frame, conf=self.conf, imgsz=224)
        
        poses_info = []

        for result in man_results:
            if result.keypoints is not None and len(result.boxes) > 0:
                # 为了稳定，只处理第一个人
                box = result.boxes[0]
                xyxy = box.xyxy[0].tolist()
                keypoints_xy = result.keypoints.xy[0].tolist()
                cls_id = int(box.cls.item())
                score = box.conf.item()
                label = self.man_model.names[cls_id]
                keypoints_conf_data = result.keypoints.conf[0].tolist() if hasattr(result.keypoints, 'conf') and result.keypoints.conf is not None else [0.0] * len(keypoints_xy)
                keypoints_data = []
                for j, xy in enumerate(keypoints_xy):
                    name = man_keypoint_names[j] if j < len(man_keypoint_names) else f'keypoint_{j}'
                    keypoints_data.append({
                        'x': xy[0],
                        'y': xy[1],
                        'confidence': keypoints_conf_data[j],
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
                break  # 只取第一个人

        if len(poses_info) == 0:
            # 无人检测到
            hand_safety = {'left_hand': 'invisible', 'right_hand': 'invisible'}
            coat_safety = 'invisible'
            bboxes_info = []
        else:
            # 进行穿戴检测
            wearing_results = self.wearing_model.predict(frame, conf=self.conf, imgsz=224)
            bboxes_info = []
            for result in wearing_results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls.item())
                    score = box.conf.item()
                    label = self.wearing_model.names[cls_id]
                    # 坐标全部转为int，防止OpenCV报错
                    xyxy = list(map(int, box.xyxy[0].tolist()))
                    bbox_info = {
                        'class_id': cls_id,
                        'label': label,
                        'score': score,
                        'box': xyxy
                    }
                    bboxes_info.append(bbox_info)

            # 匹配逻辑
            pose = poses_info[0]
            # coat_safety
            has_lab_coat = any(bbox['class_id'] == 2 for bbox in bboxes_info)
            upper_body_names = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip']
            visible_upper_count = sum(1 for kp in pose['keypoints'] if kp['name'] in upper_body_names and kp['confidence'] > 0.5)
            all_shoulders_hips_visible = all(kp['confidence'] > 0.5 for kp in pose['keypoints'] if kp['name'] in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'])
            if has_lab_coat:
                coat_safety = 'lab_coat'
            elif all_shoulders_hips_visible:
                coat_safety = 'no_lab_coat'
            elif visible_upper_count < 4:
                coat_safety = 'invisible'
            else:
                coat_safety = 'too_close'

            # hand_safety
            body_width = pose['box'][2] - pose['box'][0]
            dist_thresh = 0.2 * body_width  # 距离阈值为人体框宽度的20%
            def get_kp(name):
                return next((kp for kp in pose['keypoints'] if kp['name'] == name), None)

            def determine_hand_safety(wrist_kp, bboxes):
                if wrist_kp is None or wrist_kp['confidence'] <= 0.5:
                    return 'invisible'
                wx, wy = wrist_kp['x'], wrist_kp['y']
                matched_glove = False
                matched_naked = False
                for bbox in bboxes:
                    if bbox['class_id'] not in [0, 1]:  # 0: blue_glove, 1: naked_hand
                        continue
                    bx1, by1, bx2, by2 = bbox['box']
                    cx = (bx1 + bx2) / 2
                    cy = (by1 + by2) / 2
                    dist = ((wx - cx) ** 2 + (wy - cy) ** 2) ** 0.5
                    if dist < dist_thresh:
                        if bbox['class_id'] == 0:
                            matched_glove = True
                        elif bbox['class_id'] == 1:
                            matched_naked = True
                if matched_glove:
                    return 'glove'
                elif matched_naked:
                    return 'naked_hand'
                else:
                    return 'invisible'  # 无匹配，视为invisible

            left_wrist = get_kp('left_wrist')
            right_wrist = get_kp('right_wrist')
            hand_safety = {
                'left_hand': determine_hand_safety(left_wrist, bboxes_info),
                'right_hand': determine_hand_safety(right_wrist, bboxes_info)
            }

        # 计算FPS
        avg_fps = self.fps_calculator.get()

        detection_info = {
            'hand_safety': hand_safety,
            'coat_safety': coat_safety,
            'fps': avg_fps,
            'man_detected': len(poses_info)
        }

        # 绘制
        processed_frame = frame.copy()
        
        # 添加安全检测结果绘制
        processed_frame = draw_safety(
            processed_frame, 
            detection_info, 
            poses_info=poses_info,
            show_status=True,
            show_debug=True
        )
        
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
        processed_frame, detection_info = self.detect_frame(img)
        cv2.imshow("Safety Detection", processed_frame)   
        cv2.waitKey(0)
        cv2.imwrite("examples/results/safety_test.png", processed_frame)
        print(detection_info)

# 示例用法
if __name__ == "__main__":
    detector = SafetyDetector()
    # 示例：请替换为你的图片路径
    detector.debug_image_predict("examples/safety_test.png")