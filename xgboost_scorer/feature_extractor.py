import numpy as np
from sklearn.linear_model import LinearRegression

class FeatureExtractor:
    """
    一个可重用的特征提取器，用于从检测到的仪器和手的信息中为XGBoost模型生成特征向量。
    """
    def __init__(self, keypoint_visibility_threshold=0.05):
        self.kpt_vis_threshold = keypoint_visibility_threshold
        
        # 定义所有仪器组合的特征提取配置
        self.config = {
            # ('beaker', 'graduated_cylinder')
            ('beaker', 'graduated_cylinder'): {
                'v1_name': 'graduated_cylinder', 'v2_name': 'beaker',
                'v1_kps': {'spout': 0, 'mouth': 1, 'bottom': 2, 'top_quarter': 3, 'bottom_quarter': 4},
                'v2_kps': {'spout': 0, 'mouth': 1, 'bottom': 2},
                'v1_angle_pts': [1, 2, 3, 4],
                'v2_angle_pts': ['bottom', 'mouth'],
                'f_map': {
                    'f11': ('v2', 'bottom'), 'f12': ('v2', 'mouth'), 'f17': ('v2', 'spout'),
                    'f15': ('v1', 'bottom'), 'f16': ('v1', 'mouth'),
                    'f13': None, 'f14': None # 量筒组合中f13, f14未使用
                }
            },
            # ('beaker', 'volumetric_flask')
            ('beaker', 'volumetric_flask'): {
                'v1_name': 'beaker', 'v2_name': 'volumetric_flask',
                'v1_kps': {'spout': 0, 'mouth': 1, 'bottom': 2},
                'v2_kps': {'bottom': 0, 'mouth': 1, 'stopper': 2, 'line': 3},
                'v1_angle_pts': ['bottom', 'mouth'],
                'v2_angle_pts': [0, 1, 3],
                'f_map': {
                    'f11': ('v2', 'bottom'), 'f12': ('v2', 'mouth'), 'f13': ('v2', 'stopper'), 'f14': ('v2', 'line'),
                    'f15': ('v1', 'bottom'), 'f16': ('v1', 'mouth'), 'f17': ('v1', 'spout')
                }
            },
            # ('graduated_cylinder', 'volumetric_flask')
            ('graduated_cylinder', 'volumetric_flask'): {
                'v1_name': 'graduated_cylinder', 'v2_name': 'volumetric_flask',
                'v1_kps': {'spout': 0, 'mouth': 1, 'bottom': 2, 'top_quarter': 3, 'bottom_quarter': 4},
                'v2_kps': {'bottom': 0, 'mouth': 1, 'stopper': 2, 'line': 3},
                'v1_angle_pts': [1, 2, 3, 4],
                'v2_angle_pts': [0, 1, 3],
                'f_map': {
                    'f11': ('v2', 'bottom'), 'f12': ('v2', 'mouth'), 'f13': ('v2', 'stopper'), 'f14': ('v2', 'line'),
                    'f15': ('v1', 'bottom'), 'f16': ('v1', 'mouth'), 'f17': ('v1', 'spout')
                }
            }
        }

    def _get_box_center(self, box):
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    def _get_diagonal(self, box):
        return np.linalg.norm(np.array(box[:2]) - np.array(box[2:]))

    def _get_line_angle(self, points):
        if len(points) < 2: return None
        try:
            X = np.array(points)[:, 0].reshape(-1, 1)
            y = np.array(points)[:, 1]
            reg = LinearRegression().fit(X, y)
            p1 = np.array([X.min(), reg.predict(X.min().reshape(1, -1))[0]])
            p2 = np.array([X.max(), reg.predict(X.max().reshape(1, -1))[0]])
        except Exception:
            p1, p2 = np.array(points[0]), np.array(points[-1])
        return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

    def _get_kps_from_info(self, vessel_info):
        """从新的检测器信息格式中提取关键点和置信度"""
        if not vessel_info or 'keypoints' not in vessel_info:
            return None
        
        kps_data = vessel_info['keypoints']
        xy = np.array([[kp['x'], kp['y']] for kp in kps_data])
        # 如果没有置信度信息，则假设全部可见
        conf = np.array([kp.get('confidence', 1.0) for kp in kps_data])
        return {'xy': xy, 'conf': conf}

    def extract(self, vessel1_info, vessel2_info, hands_info):
        """
        通用的特征提取方法。

        Args:
            vessel1_info (dict): 第一个仪器的检测信息。
            vessel2_info (dict): 第二个仪器的检测信息。
            hands_info (list): 手的检测信息列表。

        Returns:
            list: 包含18个特征的列表，如果无法提取则返回None。
        """
        labels = tuple(sorted((vessel1_info['label'], vessel2_info['label'])))
        if labels not in self.config:
            print(f"Warning: No feature extraction config found for action {labels}")
            return None
        
        cfg = self.config[labels]
        
        # 确保vessel1和vessel2与配置中的v1, v2名称匹配
        v1 = vessel1_info if vessel1_info['label'] == cfg['v1_name'] else vessel2_info
        v2 = vessel2_info if vessel2_info['label'] == cfg['v2_name'] else vessel1_info

        features = {f'f{i}': -1.0 for i in range(1, 19)}
        for i in range(11, 18): features[f'f{i}'] = 0.0

        # 特征 f1, f2: 仪器存在性 (在此函数中总是为1)
        features['f1'] = 1.0
        features['f2'] = 1.0

        v1_center, v2_center = self._get_box_center(v1['box']), self._get_box_center(v2['box'])
        v1_diag = self._get_diagonal(v1['box'])
        
        # 特征 f3: 仪器中心距离
        if v1_diag > 0:
            features['f3'] = np.linalg.norm(v1_center - v2_center) / v1_diag

        v1_kps = self._get_kps_from_info(v1)
        v2_kps = self._get_kps_from_info(v2)

        # 特征 f4, f5: 关键点距离
        if v1_kps is not None and v2_kps is not None:
            v1_spout_idx = cfg['v1_kps']['spout']
            v2_mouth_idx = cfg['v2_kps']['mouth']
            if v1_kps['conf'][v1_spout_idx] > self.kpt_vis_threshold and v2_kps['conf'][v2_mouth_idx] > self.kpt_vis_threshold:
                v1_spout_pt = v1_kps['xy'][v1_spout_idx]
                v2_mouth_pt = v2_kps['xy'][v2_mouth_idx]
                if v1_diag > 0:
                    features['f4'] = np.linalg.norm(v1_spout_pt - v2_mouth_pt) / v1_diag
                features['f5'] = 1.0

        # 特征 f6: 手的数量
        features['f6'] = len(hands_info)

        # 特征 f7, f8: 手-仪器距离
        if hands_info:
            hand_centers = [self._get_box_center(h['box']) for h in hands_info]
            
            # f7: v1
            dist_v1_hands = [np.linalg.norm(hc - v1_center) for hc in hand_centers]
            closest_hand_v1_idx = np.argmin(dist_v1_hands)
            hand_diag_v1 = self._get_diagonal(hands_info[closest_hand_v1_idx]['box'])
            if hand_diag_v1 > 0:
                features['f7'] = dist_v1_hands[closest_hand_v1_idx] / hand_diag_v1

            # f8: v2
            dist_v2_hands = [np.linalg.norm(hc - v2_center) for hc in hand_centers]
            closest_hand_v2_idx = np.argmin(dist_v2_hands)
            hand_diag_v2 = self._get_diagonal(hands_info[closest_hand_v2_idx]['box'])
            if hand_diag_v2 > 0:
                features['f8'] = dist_v2_hands[closest_hand_v2_idx] / hand_diag_v2

        # 特征 f9, f10: 仪器宽高比
        w2, h2 = v2['box'][2] - v2['box'][0], v2['box'][3] - v2['box'][1]
        features['f9'] = w2 / h2 if h2 > 0 else -1.0
        w1, h1 = v1['box'][2] - v1['box'][0], v1['box'][3] - v1['box'][1]
        features['f10'] = w1 / h1 if h1 > 0 else -1.0

        # 特征 f11-f17: 关键点可见性
        for f_key, mapping in cfg['f_map'].items():
            if mapping is None:
                features[f_key] = 0.0
                continue
            
            vessel_obj = v1 if mapping[0] == 'v1' else v2
            kps_obj = v1_kps if mapping[0] == 'v1' else v2_kps
            kps_map = cfg['v1_kps'] if mapping[0] == 'v1' else cfg['v2_kps']
            
            if kps_obj is not None:
                kp_name = mapping[1]
                if kp_name in kps_map:
                    kp_idx = kps_map[kp_name]
                    if kps_obj['conf'][kp_idx] > self.kpt_vis_threshold:
                        features[f_key] = 1.0

        # 特征 f18: 角度差
        if v1_kps is not None and v2_kps is not None:
            def get_pts(kps_obj, kps_map, pt_defs):
                pts = []
                for p_def in pt_defs:
                    idx = kps_map[p_def] if isinstance(p_def, str) else p_def
                    if kps_obj['conf'][idx] > self.kpt_vis_threshold:
                        pts.append(kps_obj['xy'][idx])
                return pts

            v1_pts_to_fit = get_pts(v1_kps, cfg['v1_kps'], cfg['v1_angle_pts'])
            v2_pts_to_fit = get_pts(v2_kps, cfg['v2_kps'], cfg['v2_angle_pts'])
            
            angle1, angle2 = self._get_line_angle(v1_pts_to_fit), self._get_line_angle(v2_pts_to_fit)
            if angle1 is not None and angle2 is not None:
                angle_diff = abs(angle1 - angle2)
                # 确保角度差在[0, 90]之间
                angle_diff = angle_diff % 180
                features['f18'] = min(angle_diff, 180 - angle_diff)

        return list(features.values())
