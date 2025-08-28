import os
import cv2
import torch
import mediapipe as mp
import numpy as np
import xgboost as xgb
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression

# --- 为每个任务创建独立的全局缓存字典 ---
_MODELS_CACHE_CF = {} # 用于 量筒 -> 容量瓶
_MODELS_CACHE_BF = {} # 用于 烧杯 -> 容量瓶

# ==============================================================================
# --- 函数一：量筒 -> 容量瓶 预测 ---
# ==============================================================================
def get_cylinder_to_flask_confidence(img: np.ndarray) -> float:
    """
    一个完全自包含的函数，用于预测“量筒->容量瓶”倾倒动作的置信度。

    它会按需加载并缓存所有必要的模型，对输入的单张OpenCV图像
    执行完整的特征提取和模型预测流水线。

    参数:
    img (np.ndarray): 一个通过cv2.imread()等方式加载的OpenCV图像 (BGR格式)。

    返回:
    float: 介于0.0到100.0之间的置信度分数。
           如果发生错误，返回0.0。
    """
    global _MODELS_CACHE_CF

    # --- 初始化与设置 (只在第一次调用时执行) ---
    if not _MODELS_CACHE_CF:
        print("首次调用 (量筒->容量瓶)，正在加载所有模型，请稍候...")
        try:            
            _MODELS_CACHE_CF['YOLO_MODELS'] = {
                'box_model': YOLO(r"./model/yolov11-vessels.pt"),
                'pose_models': {
                    'graduated_cylinder': YOLO(r"./model/yolov11-pose-graduatedcylinder.pt"),
                    'volumetric_flask': YOLO(r"./model/yolov11-pose-volumetricflask.pt")
                }
            }
            _MODELS_CACHE_CF['HANDS_DETECTOR'] = mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
            
            xgb_model = xgb.XGBClassifier()
            # 推荐使用 .ubj 二进制格式以避免编码错误
            xgb_model_path = r"./model/xgboost_pour_model_volumetricflask_graduatedcylinder.json"

            xgb_model.load_model(xgb_model_path)
            _MODELS_CACHE_CF['XGB_MODEL'] = xgb_model

            print("所有模型加载成功并已缓存！")
        except Exception as e:
            print(f"严重错误：模型加载失败！请检查所有模型文件路径是否正确。")
            print(f"错误详情: {e}")
            _MODELS_CACHE_CF.clear()
            return 0.0

    # --- 定义常量 ---
    BOX_CONFIDENCE_THRESHOLD = 0.5
    KEYPOINT_VISIBILITY_THRESHOLD = 0.05
    VESSEL_IDS = {'cylinder': 0, 'flask': 2}
    KEYPOINT_IDS = {
        'cylinder': {'spout': 0, 'mouth': 1, 'bottom': 2, 'top_quarter': 3, 'bottom_quarter': 4},
        'flask': {'bottom': 0, 'mouth': 1, 'stopper': 2, 'line': 3}
    }

    # --- 特征提取流水线 (作为嵌套函数) ---
    def get_box_center(box): return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    def get_diagonal(box): return np.linalg.norm(np.array([box[0], box[1]]) - np.array([box[2], box[3]]))
    def get_hand_details(hand_landmarks, image_shape):
        h, w, _ = image_shape
        points = np.array([(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark])
        center = np.mean(points[[0, 5, 9, 13, 17]], axis=0)
        xmin, ymin, xmax, ymax = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
        return {'center': center, 'box': [xmin, ymin, xmax, ymax]}
    def get_line_angle(points):
        if len(points) < 2: return None
        if len(points) > 2:
            try:
                X, y = np.array(points)[:, 0].reshape(-1, 1), np.array(points)[:, 1]
                reg = LinearRegression().fit(X, y)
                p1 = np.array([X.min(), reg.predict(X.min().reshape(1, -1))[0]])
                p2 = np.array([X.max(), reg.predict(X.max().reshape(1, -1))[0]])
            except Exception: p1, p2 = np.array(points[0]), np.array(points[-1])
        else: p1, p2 = np.array(points[0]), np.array(points[-1])
        return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    def initial_detection(image, box_model, hands_detector):
        all_detected_cylinders, all_detected_flasks = [], []
        box_results = box_model(image, verbose=False)
        for r in box_results[0].boxes:
            if r.conf[0] > BOX_CONFIDENCE_THRESHOLD:
                item = {'box': r.xyxy[0].cpu().numpy(), 'conf': r.conf[0].cpu().numpy(), 'keypoints': None}
                if int(r.cls[0]) == VESSEL_IDS['cylinder']: all_detected_cylinders.append(item)
                elif int(r.cls[0]) == VESSEL_IDS['flask']: all_detected_flasks.append(item)
        detected_hands = []
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = hands_detector.process(img_rgb)
        if hand_results.multi_hand_landmarks:
            for hand_lms in hand_results.multi_hand_landmarks:
                detected_hands.append(get_hand_details(hand_lms, image.shape))
        return all_detected_cylinders, all_detected_flasks, detected_hands
    def select_target_pair(cylinders, flasks, hands):
        if not cylinders and flasks: return None, flasks[0]
        if cylinders and not flasks: return cylinders[0], None
        if not cylinders and not flasks: return None, None
        if len(cylinders) == 1 and len(flasks) == 1: return cylinders[0], flasks[0]
        if len(hands) == 2:
            min_dist, best_pair = float('inf'), (None, None)
            for cyl in cylinders:
                for flsk in flasks:
                    c_c, f_c = get_box_center(cyl['box']), get_box_center(flsk['box'])
                    h0_c, h1_c = hands[0]['center'], hands[1]['center']
                    dist1, dist2 = np.linalg.norm(f_c-h0_c)+np.linalg.norm(c_c-h1_c), np.linalg.norm(f_c-h1_c)+np.linalg.norm(c_c-h0_c)
                    if min(dist1, dist2) < min_dist: min_dist, best_pair = min(dist1, dist2), (cyl, flsk)
            return best_pair[0], best_pair[1]
        elif len(hands) == 1:
            h_c, selected_flask, selected_cylinder = hands[0]['center'], None, None
            all_vessels = cylinders + flasks
            if not all_vessels: return None, None
            closest_vessel = min(all_vessels, key=lambda v: np.linalg.norm(get_box_center(v['box']) - h_c))
            if closest_vessel in flasks:
                selected_flask = closest_vessel
                if cylinders: selected_cylinder = min(cylinders, key=lambda c: np.linalg.norm(get_box_center(c['box']) - get_box_center(selected_flask['box'])))
            else:
                selected_cylinder = closest_vessel
                if flasks: selected_flask = min(flasks, key=lambda f: np.linalg.norm(get_box_center(f['box']) - get_box_center(selected_cylinder['box'])))
            return selected_cylinder, selected_flask
        else:
            min_dist, best_pair = float('inf'), (None, None)
            for cyl in cylinders:
                for flsk in flasks:
                    dist = np.linalg.norm(get_box_center(cyl['box']) - get_box_center(flsk['box']))
                    if dist < min_dist: min_dist, best_pair = dist, (cyl, flsk)
            return best_pair[0], best_pair[1]
    def run_pose_detection(image, vessel, vessel_type, pose_models):
        if vessel is None: return
        box = vessel['box'].astype(int)
        box[0], box[1] = max(0, box[0]), max(0, box[1])
        box[2], box[3] = min(image.shape[1], box[2]), min(image.shape[0], box[3])
        crop = image[box[1]:box[3], box[0]:box[2]]
        if crop.size == 0: return
        pose_results = pose_models[vessel_type](crop, verbose=False)
        if hasattr(pose_results[0], 'keypoints') and pose_results[0].keypoints is not None and pose_results[0].keypoints.xy.numel() > 0:
            vessel['keypoints'] = {'xy': pose_results[0].keypoints.xy[0].cpu().numpy(),'conf': pose_results[0].keypoints.conf[0].cpu().numpy() if pose_results[0].keypoints.conf is not None else np.ones(len(pose_results[0].keypoints.xy[0]))}
    def extract_features(cylinder, flask, hands):
        features = {f'f{i}': -1 for i in range(1, 19)}
        for i in range(11, 18): features[f'f{i}'] = 0
        features['f1'] = 1 if cylinder else 0
        features['f2'] = 1 if flask else 0
        if features['f1'] and features['f2']:
            cyl_center, flsk_center = get_box_center(cylinder['box']), get_box_center(flask['box'])
            cyl_diag = get_diagonal(cylinder['box'])
            features['f3'] = np.linalg.norm(cyl_center - flsk_center) / cyl_diag if cyl_diag > 0 else -1
            cyl_kps, flsk_kps = cylinder.get('keypoints'), flask.get('keypoints')
            if cyl_kps is not None and flsk_kps is not None:
                if cyl_kps['conf'][KEYPOINT_IDS['cylinder']['spout']] > KEYPOINT_VISIBILITY_THRESHOLD and flsk_kps['conf'][KEYPOINT_IDS['flask']['mouth']] > KEYPOINT_VISIBILITY_THRESHOLD:
                    cyl_spout_pt, flsk_mouth_pt = cyl_kps['xy'][KEYPOINT_IDS['cylinder']['spout']], flsk_kps['xy'][KEYPOINT_IDS['flask']['mouth']]
                    features['f4'] = np.linalg.norm(cyl_spout_pt - flsk_mouth_pt) / cyl_diag if cyl_diag > 0 else -1
                    features['f5'] = 1
        features['f6'] = len(hands)
        if features['f1'] and features['f6'] > 0:
            cyl_center = get_box_center(cylinder['box'])
            closest_hand_cyl = min(hands, key=lambda h: np.linalg.norm(h['center'] - cyl_center))
            dist_cyl_hand = np.linalg.norm(closest_hand_cyl['center'] - cyl_center)
            if features['f6'] == 2:
                hand_diag = get_diagonal(closest_hand_cyl['box'])
                features['f7'] = dist_cyl_hand / hand_diag if hand_diag > 0 else -1
            elif features['f6'] == 1:
                dist_flsk_hand = float('inf')
                if features['f2']: dist_flsk_hand = np.linalg.norm(hands[0]['center'] - get_box_center(flask['box']))
                if dist_cyl_hand <= dist_flsk_hand:
                     hand_diag = get_diagonal(closest_hand_cyl['box'])
                     features['f7'] = dist_cyl_hand / hand_diag if hand_diag > 0 else -1
        if features['f2'] and features['f6'] > 0:
            flsk_center = get_box_center(flask['box'])
            closest_hand_flsk = min(hands, key=lambda h: np.linalg.norm(h['center'] - flsk_center))
            dist_flsk_hand = np.linalg.norm(closest_hand_flsk['center'] - flsk_center)
            if features['f6'] == 2:
                hand_diag = get_diagonal(closest_hand_flsk['box'])
                features['f8'] = dist_flsk_hand / hand_diag if hand_diag > 0 else -1
            elif features['f6'] == 1:
                dist_cyl_hand = float('inf')
                if features['f1']: dist_cyl_hand = np.linalg.norm(hands[0]['center'] - get_box_center(cylinder['box']))
                if dist_flsk_hand < dist_cyl_hand:
                     hand_diag = get_diagonal(closest_hand_flsk['box'])
                     features['f8'] = dist_flsk_hand / hand_diag if hand_diag > 0 else -1
        if features['f2']:
            w, h = flask['box'][2] - flask['box'][0], flask['box'][3] - flask['box'][1]
            features['f9'] = w / h if h > 0 else -1
        if features['f1']:
            w, h = cylinder['box'][2] - cylinder['box'][0], cylinder['box'][3] - cylinder['box'][1]
            features['f10'] = w / h if h > 0 else -1
        if features['f2'] and flask.get('keypoints') is not None:
            kps_conf = flask['keypoints']['conf']
            features['f11'] = 1 if kps_conf[KEYPOINT_IDS['flask']['bottom']] > KEYPOINT_VISIBILITY_THRESHOLD else 0
            features['f12'] = 1 if kps_conf[KEYPOINT_IDS['flask']['mouth']] > KEYPOINT_VISIBILITY_THRESHOLD else 0
            features['f13'] = 1 if kps_conf[KEYPOINT_IDS['flask']['stopper']] > KEYPOINT_VISIBILITY_THRESHOLD else 0
            features['f14'] = 1 if kps_conf[KEYPOINT_IDS['flask']['line']] > KEYPOINT_VISIBILITY_THRESHOLD else 0
        if features['f1'] and cylinder.get('keypoints') is not None:
            kps_conf = cylinder['keypoints']['conf']
            features['f15'] = 1 if kps_conf[KEYPOINT_IDS['cylinder']['bottom']] > KEYPOINT_VISIBILITY_THRESHOLD else 0
            features['f16'] = 1 if kps_conf[KEYPOINT_IDS['cylinder']['mouth']] > KEYPOINT_VISIBILITY_THRESHOLD else 0
            features['f17'] = 1 if kps_conf[KEYPOINT_IDS['cylinder']['spout']] > KEYPOINT_VISIBILITY_THRESHOLD else 0
        if features['f1'] and features['f2'] and cylinder.get('keypoints') is not None and flask.get('keypoints') is not None:
            flask_pts_to_fit = [flask['keypoints']['xy'][i] for i in [0, 1, 3] if flask['keypoints']['conf'][i] > KEYPOINT_VISIBILITY_THRESHOLD]
            cyl_pts_to_fit = [cylinder['keypoints']['xy'][i] for i in [1, 2, 3, 4] if cylinder['keypoints']['conf'][i] > KEYPOINT_VISIBILITY_THRESHOLD]
            angle1, angle2 = get_line_angle(flask_pts_to_fit), get_line_angle(cyl_pts_to_fit)
            if angle1 is not None and angle2 is not None:
                angle_diff = abs(angle1 - angle2)
                features['f18'] = min(angle_diff, 180 - angle_diff) if angle_diff <= 180 else min(angle_diff - 180, 180 - (angle_diff - 180))
        return list(features.values())

    # --- 执行流水线 ---
    if not isinstance(img, np.ndarray) or img.size == 0: return 0.0
    all_cylinders, all_flasks, all_hands = initial_detection(img, _MODELS_CACHE_CF['YOLO_MODELS']['box_model'], _MODELS_CACHE_CF['HANDS_DETECTOR'])
    selected_cylinder, selected_flask = select_target_pair(all_cylinders, all_flasks, all_hands)
    run_pose_detection(img, selected_cylinder, 'graduated_cylinder', _MODELS_CACHE_CF['YOLO_MODELS']['pose_models'])
    run_pose_detection(img, selected_flask, 'volumetric_flask', _MODELS_CACHE_CF['YOLO_MODELS']['pose_models'])
    feature_vector = extract_features(selected_cylinder, selected_flask, all_hands)
    input_data = np.array(feature_vector).reshape(1, -1)
    prediction_probabilities = _MODELS_CACHE_CF['XGB_MODEL'].predict_proba(input_data)
    confidence_score = prediction_probabilities[0][1]
    return confidence_score * 100