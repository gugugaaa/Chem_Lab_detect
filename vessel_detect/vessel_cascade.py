"""
使用 models/vessel_box.pt 检测容器，针对每个box区域裁切后，zero padding为正方形，输入对应的pose模型（models/{box_label}.pt）推理关键点。
- 只保留中心点在pure_img中心的检测结果，并还原关键点到原图坐标。
- 关键点名称需根据模型字典顺序读取。
- 其他逻辑与 man_detector.py 类似，yolo pose模型用法相同。
"""

import cv2
import sys
import os
import numpy as np
from ultralytics import YOLO

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.draw_keypoints import draw_keypoints
from utils.draw_fps import draw_fps
from utils.fps_calculator import FpsCalculator

# 各容器关键点名称
beaker_kpt_names = ['tip', 'mouth_center', 'bottom_center']
graduated_cylinder_kpt_names = ['tip', 'mouth_center', 'bottom_outer', 'top_quarter', 'bottom_quarter']
volumetric_flask_kpt_names = ['bottom_center', 'mouth_center', 'stopper', 'scale_mark']

# 紫罗兰-茄子色配色（BGR）
vessel_keypoint_colors = [
    (130, 76, 130),   # 紫罗兰 (Violet) BGR
    (180, 82, 120),   # 茄子色 (Eggplant) BGR
    (211, 160, 221),  # 淡紫色 (Thistle) BGR
    (148, 87, 235),   # 紫色 (Purple) BGR
    (204, 153, 255),  # 淡茄子 (Light Eggplant) BGR
]

class VesselCascadeDetector:
    def __init__(self, vessel_box_model="models/vessel_box.pt", box_conf=0.5, pts_conf=0.5, show_kpt_names=False):
        """
        初始化容器cascade检测器
        """
        self.vessel_box_model = YOLO(vessel_box_model, verbose=False)
        self.box_conf = box_conf
        self.pts_conf = pts_conf
        self.show_kpt_names = show_kpt_names
        self.fps_calculator = FpsCalculator(buffer_len=100)
        # pose模型路径映射
        self.pose_model_paths = {
            'beaker': "models/beaker.pt",
            'graduated_cylinder': "models/graduated_cylinder.pt",
            'volumetric_flask': "models/volumetric_flask.pt"
        }
        self.pose_models = {}
        for label, path in self.pose_model_paths.items():
            if os.path.exists(path):
                self.pose_models[label] = YOLO(path, verbose=False)
        # 关键点名称映射
        self.kpt_names_map = {
            'beaker': beaker_kpt_names,
            'graduated_cylinder': graduated_cylinder_kpt_names,
            'volumetric_flask': volumetric_flask_kpt_names
        }

    def __del__(self):
        if hasattr(self, 'vessel_box_model'):
            del self.vessel_box_model
        for m in self.pose_models.values():
            del m

    def detect(self, frame):
        """
        检测图像中的容器及关键点
        返回：原始检测结果
        """
        results = self.vessel_box_model.predict(frame, conf=self.box_conf, imgsz=320)
        return results

    def crop_and_pad_square(self, img, box, pad=5):
        """
        裁切box区域，向外pad像素，zero padding补成正方形
        返回pure_img, xyxy_in_origin, pad_left, pad_top
        """
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1p = max(x1 - pad, 0)
        y1p = max(y1 - pad, 0)
        x2p = min(x2 + pad, w)
        y2p = min(y2 + pad, h)
        crop = img[y1p:y2p, x1p:x2p]
        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        pad_top = (side - ch) // 2
        pad_bottom = side - ch - pad_top
        pad_left = (side - cw) // 2
        pad_right = side - cw - pad_left
        pure_img = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        return pure_img, (x1p, y1p, x2p, y2p), pad_left, pad_top

    def detect_frame(self, frame):
        """
        检测单帧图像，返回处理后帧和检测信息
        """
        vessel_results = self.detect(frame)
        processed_frame = frame.copy()
        poses_info = []

        for result in vessel_results:
            if result.boxes is not None and len(result.boxes) > 0:
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    xyxy = [int(round(coord)) for coord in box.xyxy[0].tolist()]
                    cls_id = int(box.cls.item())
                    score = round(box.conf.item(), 2)
                    label = self.vessel_box_model.names[cls_id]
                    # 裁切pure_img
                    pure_img, (x1p, y1p, x2p, y2p), pad_left, pad_top = self.crop_and_pad_square(frame, xyxy, pad=5)
                    # 级联pose模型
                    pose_model = self.pose_models.get(label)
                    kpt_names = self.kpt_names_map.get(label, [])
                    if pose_model is None or not kpt_names:
                        continue
                    pose_results = pose_model.predict(pure_img, conf=self.pts_conf, imgsz=384)
                    # 只保留中心点在pure_img中心附近的检测
                    side = pure_img.shape[0]
                    cx_c, cy_c = side // 2, side // 2
                    valid_idx = -1
                    for j, pres in enumerate(pose_results):
                        if pres.boxes is not None and len(pres.boxes) > 0:
                            for k in range(len(pres.boxes)):
                                kbox = pres.boxes[k]
                                kxywh = kbox.xywh[0].tolist()
                                kcx, kcy = int(round(kxywh[0])), int(round(kxywh[1]))
                                # 允许中心点在中心±5%区域
                                if abs(kcx - cx_c) < side * 0.05 and abs(kcy - cy_c) < side * 0.05:
                                    valid_idx = k
                                    break
                        if valid_idx != -1:
                            break
                    if valid_idx == -1:
                        continue
                    pres = pose_results[0]
                    # 还原关键点坐标到原图
                    keypoints_xy = [[int(round(x)), int(round(y))] for x, y in pres.keypoints.xy[valid_idx].tolist()]
                    keypoints_conf_data = pres.keypoints.conf[valid_idx] if hasattr(pres.keypoints, 'conf') and pres.keypoints.conf is not None else None
                    keypoints_data = []
                    for j, xy in enumerate(keypoints_xy):
                        # 还原到原图
                        x_img = xy[0] - pad_left + x1p
                        y_img = xy[1] - pad_top + y1p
                        name = kpt_names[j] if j < len(kpt_names) else f'keypoint_{j}'
                        conf = round(keypoints_conf_data[j].item(), 2) if keypoints_conf_data is not None else None
                        keypoints_data.append({
                            'x': x_img,
                            'y': y_img,
                            'confidence': conf,
                            'name': name
                        })
                    pose_info = {
                        'class_id': cls_id,
                        'label': label,
                        'score': score,
                        'box': xyxy,
                        'keypoints': keypoints_data
                    }
                    poses_info.append(pose_info)

        avg_fps = self.fps_calculator.get()
        processed_frame = draw_fps(processed_frame, avg_fps)
        detection_info = {
            'poses': poses_info,
            'fps': avg_fps
        }
        processed_frame = draw_keypoints(
            processed_frame,
            detection_info,
            keypoint_colors=vessel_keypoint_colors,
            show_names=self.show_kpt_names,
            draw_bbox=True
        )
        return processed_frame, detection_info

    def process_video(self, video_source=0, display=True):
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
                    cv2.imshow("Vessel Cascade Detection", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

    def debug_image_predict(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return
        processed_frame, detection_info = self.detect_frame(img)
        cv2.imshow("Vessel Cascade Detection", processed_frame)
        cv2.waitKey(0)
        cv2.imwrite("examples/test/temp/beaker.png", processed_frame)
        print(detection_info)


# 示例用法
if __name__ == "__main__":
    detector = VesselCascadeDetector(show_kpt_names=True)
    detector.debug_image_predict("examples/test/temp/image3.png")