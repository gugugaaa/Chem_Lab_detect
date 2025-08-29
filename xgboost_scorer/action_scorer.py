"""
操作得分 ActionScorer ：
（只考虑两只手的情况）
加载xgboost模型，模型是models/g-b v-b v-g.json
根据gesture info和vessel info：
分类到双手操作 仪器AB/BC/AC
    首先，选取和手的box计算出cx cy的棋盘距离最小的仪器，作为这只手正在操作的仪器。
    然后，匹配当前正在进行的操作
    'g-b': graduated_cylinder + beaker
    'v-b': volumetric_flask + beaker
    以及'v-g', 'g-g', 'v-v', 'b-b'（同一个仪器的没做）
提取特征，制作向量
输入到对应的模型
返回分数

"""
import os
import sys
import numpy as np
import xgboost as xgb
from itertools import combinations
from scipy.spatial.distance import cdist

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xgboost_scorer.feature_extractor import FeatureExtractor


# 关键点名称映射，用于特征提取
KEYPOINT_NAMES = {
    'beaker': ['tip', 'mouth_center', 'bottom_center'],
    'graduated_cylinder': ['tip', 'mouth_center', 'bottom_outer', 'top_quarter', 'bottom_quarter'],
    'volumetric_flask': ['bottom_center', 'mouth_center', 'stopper', 'scale_mark']
}

class ActionScorer:
    """
    根据检测到的手和仪器信息，对化学实验操作进行评分。
    """
    def __init__(self, model_dir="models"):
        """
        初始化ActionScorer，加载所有XGBoost模型。
        """
        self.model_dir = model_dir
        self.scorers = {}
        self.model_files = {
            ('graduated_cylinder', 'beaker'): 'g_b.json',
            ('volumetric_flask', 'beaker'): 'v_b.json',
            ('volumetric_flask', 'graduated_cylinder'): 'v_g.json'
        }
        self._load_models()
        self.feature_extractor = FeatureExtractor()

    def _load_models(self):
        """加载所有在model_files中定义的XGBoost模型。"""
        print("Loading XGBoost models for action scoring...")
        for labels, filename in self.model_files.items():
            # 使用排序后的元组作为键，确保顺序无关
            sorted_labels = tuple(sorted(labels))
            model_path = os.path.join(self.model_dir, filename)
            if os.path.exists(model_path):
                try:
                    model = xgb.XGBClassifier()
                    model.load_model(model_path)
                    self.scorers[sorted_labels] = model
                    print(f"  - Loaded '{filename}' for {sorted_labels}")
                except Exception as e:
                    print(f"Error loading model {filename}: {e}")
            else:
                print(f"Warning: Model file not found at {model_path}")

    def score_frame(self, vessel_info, gesture_info):
        """
        对单帧的检测结果进行评分，仅考虑单人双手操作，直接返回操作名和分数。

        Args:
            vessel_info (dict): 来自VesselCascadeDetector的检测结果。
            gesture_info (dict): 来自GestureDetector的检测结果。

        Returns:
            dict: {"operation": "beaker+graduated_cylinder", "score": 95.5}
                  若无有效操作，返回 None
        """
        hands = gesture_info.get('hands', [])
        vessels = vessel_info.get('poses', [])
        
        if len(vessels) < 2 or len(hands) != 2:
            return None

        # 1. 匹配手和最近的仪器
        matched_hands = self._match_hands_to_vessels(hands, vessels)
        if len(matched_hands) != 2:
            return None

        # 2. 直接评分单个操作
        result = self._score_single_operation(matched_hands)
        
        print(f"Scoring result: {result}")
        return result

    def _get_box_center(self, box):
        """计算边界框的中心点。"""
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    def _match_hands_to_vessels(self, hands, vessels):
        """将每只手与其最近的仪器进行匹配。"""
        if not hands or not vessels:
            return []

        vessel_centers = np.array([self._get_box_center(v['box']) for v in vessels])
        hand_centers = np.array([self._get_box_center(h['box']) for h in hands])

        # 计算每只手到每个仪器的棋盘距离
        distances = cdist(hand_centers, vessel_centers, 'chebyshev')
        
        matched_hands = []
        for i, hand in enumerate(hands):
            if distances.shape[1] > 0:
                nearest_vessel_idx = np.argmin(distances[i])
                hand['matched_vessel'] = vessels[nearest_vessel_idx]
                matched_hands.append(hand)
        
        return matched_hands

    def _score_single_operation(self, matched_hands):
        """根据匹配的双手评分单个操作。"""
        vessel1 = matched_hands[0]['matched_vessel']
        vessel2 = matched_hands[1]['matched_vessel']

        # 确保两个仪器不是同一个
        if vessel1 is vessel2:
            return None

        labels = tuple(sorted((vessel1['label'], vessel2['label'])))
        
        if labels not in self.scorers:
            return None

        model = self.scorers[labels]
        
        feature_vector = self.feature_extractor.extract(vessel1, vessel2, matched_hands)
        input_data = np.array(feature_vector).reshape(1, -1)
        pred_proba = model.predict_proba(input_data)
        confidence = pred_proba[0][1] * 100  # 取类别为1的概率
        
        return {
            "operation": "+".join(labels), 
            "score": round(confidence, 1)
        }