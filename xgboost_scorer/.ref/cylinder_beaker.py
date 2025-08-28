import cv2
import mediapipe as mp
import numpy as np
import xgboost as xgb
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression

# 创建一个全局缓存字典，用于存储已加载的模型
# 这样可以确保模型只在第一次调用函数时加载一次
_MODELS_CACHE = {}

def get_pouring_confidence(img: np.ndarray) -> float:
    """
    用于预测“量筒->烧杯”倾倒动作的置信度。

    它会按需加载并缓存所有必要的模型，对输入的单张OpenCV图像
    执行完整的特征提取和模型预测流水线。

    参数:
    img (np.ndarray): 一个通过cv2.imread()等方式加载的OpenCV图像 (BGR格式)。

    返回:
    float: 介于0.0到100.0之间的置信度分数。
           如果发生错误，返回0.0。
    """
    global _MODELS_CACHE

    # --- 第一部分：初始化与设置 (只在第一次调用时执行) ---
    if not _MODELS_CACHE:
        print("首次调用，正在加载所有模型，请稍候...")
        try:            
            _MODELS_CACHE['YOLO_MODELS'] = {
                'box_model': YOLO("./model/yolov11-vessels.pt"),
                'pose_models': {
                    'graduated_cylinder': YOLO("./model/yolov11-pose-graduatedcylinder.pt"),
                    'beaker': YOLO("./model/yolov11-pose-beaker.pt")
                }
            }
            _MODELS_CACHE['HANDS_DETECTOR'] = mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
            
            xgb_model = xgb.XGBClassifier()
            # 推荐使用 .ubj 二进制格式以避免编码错误
            xgb_model_path = "./model/xgboost_pour_model_cylinder_beaker.json"
            xgb_model.load_model(xgb_model_path)
            _MODELS_CACHE['XGB_MODEL'] = xgb_model

            print("所有模型加载成功并已缓存！")
        except Exception as e:
            print(f"严重错误：模型加载失败！请检查所有模型文件路径是否正确。")
            print(f"错误详情: {e}")
            _MODELS_CACHE.clear() # 加载失败，清空缓存
            return 0.0

    # --- 定义常量 ---
    BOX_CONFIDENCE_THRESHOLD = 0.5
    KEYPOINT_VISIBILITY_THRESHOLD = 0.05
    VESSEL_IDS = {'cylinder': 0, 'beaker': 1}
    KEYPOINT_IDS = {
        'cylinder': {'spout': 0, 'mouth': 1, 'bottom': 2, 'top_quarter': 3, 'bottom_quarter': 4},
        'beaker': {'spout': 0, 'mouth': 1, 'bottom': 2}
    }

    # --- 第二部分：特征提取流水线 (作为嵌套函数) ---
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
                X = np.array(points)[:, 0].reshape(-1, 1)
                y = np.array(points)[:, 1]
                reg = LinearRegression().fit(X, y)
                p1 = np.array([X.min(), reg.predict(X.min().reshape(1, -1))[0]])
                p2 = np.array([X.max(), reg.predict(X.max().reshape(1, -1))[0]])
            except Exception: p1, p2 = np.array(points[0]), np.array(points[-1])
        else: p1, p2 = np.array(points[0]), np.array(points[-1])
        return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    def initial_detection(image, box_model, hands_detector):
        all_detected_cylinders, all_detected_beakers = [], []
        box_results = box_model(image, verbose=False)
        for r in box_results[0].boxes:
            if r.conf[0] > BOX_CONFIDENCE_THRESHOLD:
                item = {'box': r.xyxy[0].cpu().numpy(), 'conf': r.conf[0].cpu().numpy(), 'keypoints': None}
                if int(r.cls[0]) == VESSEL_IDS['cylinder']: all_detected_cylinders.append(item)
                elif int(r.cls[0]) == VESSEL_IDS['beaker']: all_detected_beakers.append(item)
        detected_hands = []
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = hands_detector.process(img_rgb)
        if hand_results.multi_hand_landmarks:
            for hand_lms in hand_results.multi_hand_landmarks:
                detected_hands.append(get_hand_details(hand_lms, image.shape))
        return all_detected_cylinders, all_detected_beakers, detected_hands
    def select_target_pair(cylinders, beakers, hands):
        if not cylinders and beakers: return None, beakers[0]
        if cylinders and not beakers: return cylinders[0], None
        if not cylinders and not beakers: return None, None
        if len(cylinders) == 1 and len(beakers) == 1: return cylinders[0], beakers[0]
        if len(hands) == 2:
            min_dist, best_pair = float('inf'), (None, None)
            for cyl in cylinders:
                for bk in beakers:
                    c_c, b_c = get_box_center(cyl['box']), get_box_center(bk['box'])
                    h0_c, h1_c = hands[0]['center'], hands[1]['center']
                    dist1 = np.linalg.norm(b_c - h0_c) + np.linalg.norm(c_c - h1_c)
                    dist2 = np.linalg.norm(b_c - h1_c) + np.linalg.norm(c_c - h0_c)
                    if min(dist1, dist2) < min_dist: min_dist, best_pair = min(dist1, dist2), (cyl, bk)
            return best_pair[0], best_pair[1]
        elif len(hands) == 1:
            h_c, selected_beaker, selected_cylinder = hands[0]['center'], None, None
            all_vessels = cylinders + beakers
            if not all_vessels: return None, None
            closest_vessel = min(all_vessels, key=lambda v: np.linalg.norm(get_box_center(v['box']) - h_c))
            if closest_vessel in beakers:
                selected_beaker = closest_vessel
                if cylinders: selected_cylinder = min(cylinders, key=lambda c: np.linalg.norm(get_box_center(c['box']) - get_box_center(selected_beaker['box'])))
            else:
                selected_cylinder = closest_vessel
                if beakers: selected_beaker = min(beakers, key=lambda b: np.linalg.norm(get_box_center(b['box']) - get_box_center(selected_cylinder['box'])))
            return selected_cylinder, selected_beaker
        else: # 0 hands
            min_dist, best_pair = float('inf'), (None, None)
            for cyl in cylinders:
                for bk in beakers:
                    dist = np.linalg.norm(get_box_center(cyl['box']) - get_box_center(bk['box']))
                    if dist < min_dist: min_dist, best_pair = dist, (cyl, bk)
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
    def extract_features(cylinder, beaker, hands):
        features = {f'f{i}': -1 for i in range(1, 19)}
        for i in range(11, 18): features[f'f{i}'] = 0
        features['f1'] = 1 if cylinder else 0
        features['f2'] = 1 if beaker else 0
        if features['f1'] and features['f2']:
            cyl_center, beaker_center = get_box_center(cylinder['box']), get_box_center(beaker['box'])
            cyl_diag = get_diagonal(cylinder['box'])
            features['f3'] = np.linalg.norm(cyl_center - beaker_center) / cyl_diag if cyl_diag > 0 else -1
            cyl_kps, bk_kps = cylinder.get('keypoints'), beaker.get('keypoints')
            if cyl_kps is not None and bk_kps is not None:
                if cyl_kps['conf'][KEYPOINT_IDS['cylinder']['spout']] > KEYPOINT_VISIBILITY_THRESHOLD and bk_kps['conf'][KEYPOINT_IDS['beaker']['mouth']] > KEYPOINT_VISIBILITY_THRESHOLD:
                    cyl_spout_pt, bk_mouth_pt = cyl_kps['xy'][KEYPOINT_IDS['cylinder']['spout']], bk_kps['xy'][KEYPOINT_IDS['beaker']['mouth']]
                    features['f4'] = np.linalg.norm(cyl_spout_pt - bk_mouth_pt) / cyl_diag if cyl_diag > 0 else -1
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
                dist_bk_hand = float('inf')
                if features['f2']: dist_bk_hand = np.linalg.norm(hands[0]['center'] - get_box_center(beaker['box']))
                if dist_cyl_hand <= dist_bk_hand:
                     hand_diag = get_diagonal(closest_hand_cyl['box'])
                     features['f7'] = dist_cyl_hand / hand_diag if hand_diag > 0 else -1
        if features['f2'] and features['f6'] > 0:
            beaker_center = get_box_center(beaker['box'])
            closest_hand_bk = min(hands, key=lambda h: np.linalg.norm(h['center'] - beaker_center))
            dist_bk_hand = np.linalg.norm(closest_hand_bk['center'] - beaker_center)
            if features['f6'] == 2:
                hand_diag = get_diagonal(closest_hand_bk['box'])
                features['f8'] = dist_bk_hand / hand_diag if hand_diag > 0 else -1
            elif features['f6'] == 1:
                dist_cyl_hand = float('inf')
                if features['f1']: dist_cyl_hand = np.linalg.norm(hands[0]['center'] - get_box_center(cylinder['box']))
                if dist_bk_hand < dist_cyl_hand:
                     hand_diag = get_diagonal(closest_hand_bk['box'])
                     features['f8'] = dist_bk_hand / hand_diag if hand_diag > 0 else -1
        if features['f2']:
            w, h = beaker['box'][2] - beaker['box'][0], beaker['box'][3] - beaker['box'][1]
            features['f9'] = w / h if h > 0 else -1
        if features['f1']:
            w, h = cylinder['box'][2] - cylinder['box'][0], cylinder['box'][3] - cylinder['box'][1]
            features['f10'] = w / h if h > 0 else -1
        if features['f2'] and beaker.get('keypoints') is not None:
            kps_conf = beaker['keypoints']['conf']
            features['f11'] = 1 if kps_conf[KEYPOINT_IDS['beaker']['bottom']] > KEYPOINT_VISIBILITY_THRESHOLD else 0
            features['f12'] = 1 if kps_conf[KEYPOINT_IDS['beaker']['mouth']] > KEYPOINT_VISIBILITY_THRESHOLD else 0
            features['f17'] = 1 if kps_conf[KEYPOINT_IDS['beaker']['spout']] > KEYPOINT_VISIBILITY_THRESHOLD else 0
        features['f13'], features['f14'] = 0, 0
        if features['f1'] and cylinder.get('keypoints') is not None:
            kps_conf = cylinder['keypoints']['conf']
            features['f15'] = 1 if kps_conf[KEYPOINT_IDS['cylinder']['bottom']] > KEYPOINT_VISIBILITY_THRESHOLD else 0
            features['f16'] = 1 if kps_conf[KEYPOINT_IDS['cylinder']['mouth']] > KEYPOINT_VISIBILITY_THRESHOLD else 0
        if features['f1'] and features['f2'] and cylinder.get('keypoints') is not None and beaker.get('keypoints') is not None:
            cyl_pts_to_fit = [cylinder['keypoints']['xy'][i] for i in [1, 2, 3, 4] if cylinder['keypoints']['conf'][i] > KEYPOINT_VISIBILITY_THRESHOLD]
            bk_pts_to_fit = []
            if beaker['keypoints']['conf'][KEYPOINT_IDS['beaker']['bottom']] > KEYPOINT_VISIBILITY_THRESHOLD and beaker['keypoints']['conf'][KEYPOINT_IDS['beaker']['mouth']] > KEYPOINT_VISIBILITY_THRESHOLD:
                bk_pts_to_fit = [beaker['keypoints']['xy'][KEYPOINT_IDS['beaker']['bottom']], beaker['keypoints']['xy'][KEYPOINT_IDS['beaker']['mouth']]]
            angle1, angle2 = get_line_angle(cyl_pts_to_fit), get_line_angle(bk_pts_to_fit)
            if angle1 is not None and angle2 is not None:
                angle_diff = abs(angle1 - angle2)
                features['f18'] = min(angle_diff, 180 - angle_diff) if angle_diff <= 180 else min(angle_diff - 180, 180 - (angle_diff - 180))
        return list(features.values())

    # --- 第三部分：执行流水线 ---
    if not isinstance(img, np.ndarray) or img.size == 0:
        print("错误：输入的不是一个有效的OpenCV图像。")
        return 0.0

    # 步骤A, B, C, D
    all_cylinders, all_beakers, all_hands = initial_detection(img, _MODELS_CACHE['YOLO_MODELS']['box_model'], _MODELS_CACHE['HANDS_DETECTOR'])
    selected_cylinder, selected_beaker = select_target_pair(all_cylinders, all_beakers, all_hands)
    run_pose_detection(img, selected_cylinder, 'graduated_cylinder', _MODELS_CACHE['YOLO_MODELS']['pose_models'])
    run_pose_detection(img, selected_beaker, 'beaker', _MODELS_CACHE['YOLO_MODELS']['pose_models'])
    feature_vector = extract_features(selected_cylinder, selected_beaker, all_hands)
    
    # 步骤E: 预测
    input_data = np.array(feature_vector).reshape(1, -1)
    prediction_probabilities = _MODELS_CACHE['XGB_MODEL'].predict_proba(input_data)
    confidence_score = prediction_probabilities[0][1]
    
    return confidence_score * 100