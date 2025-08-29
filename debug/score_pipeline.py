import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from vessel_detect.vessel_cascade import VesselCascadeDetector
from vessel_detect.vessel_keypoint_corrector import correct_vessel_keypoints
from gesture_detect.gesture_detector import GestureDetector
from xgboost_scorer.action_scorer import ActionScorer
from utils.draw_keypoints import draw_keypoints
from utils.draw_hand import draw_landmarks

vessel_keypoint_colors = [
    (130, 76, 130),   # 紫罗兰 (Violet) BGR
    (180, 82, 120),   # 茄子色 (Eggplant) BGR
    (211, 160, 221),  # 淡紫色 (Thistle) BGR
    (148, 87, 235),   # 紫色 (Purple) BGR
    (204, 153, 255),  # 淡茄子 (Light Eggplant) BGR
]

class ScorerPipeline:
    def __init__(self):
        self.vessel_detector = VesselCascadeDetector()
        self.gesture_detector = GestureDetector()
        self.action_scorer = ActionScorer()

    def detect_frame(self, frame):
        # 1. 容器检测
        vessel_frame, vessel_info = self.vessel_detector.detect_frame(frame)

        # 1.5 关键点矫正
        vessel_info = correct_vessel_keypoints(vessel_info)

        # 2. 手势检测
        gesture_frame, gesture_info = self.gesture_detector.detect_frame(frame)
        # 3. 评分
        score_result = None
        if vessel_info.get("poses") and gesture_info.get("hands"):
            score_result = self.action_scorer.score_frame(vessel_info, gesture_info)
        # 4. 绘制容器关键点
        vis_frame = draw_keypoints(frame, vessel_info, show_names=False, keypoint_colors=vessel_keypoint_colors, draw_bbox=True)
        # 5. 绘制手部关键点
        for hand in gesture_info.get("hands", []):
            points = [(kpt["x"], kpt["y"]) for kpt in hand.get("keypoints", [])]
            vis_frame = draw_landmarks(vis_frame, points)
        # 6. 右上角叠加得分
        if score_result:
            text = f'Score: {score_result["score"]:.1f} ({score_result["operation"]})'
            cv2.putText(vis_frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(vis_frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 1, cv2.LINE_AA)
        return vis_frame, {"vessel_info": vessel_info, "gesture_info": gesture_info, "score_result": score_result}

if __name__ == "__main__":
    
    img_path = "examples/test/test_3.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        exit(1)
    pipeline = ScorerPipeline()
    vis_frame, result = pipeline.detect_frame(img)
    print(result)
    cv2.imshow("Result", vis_frame)
    # cv2.imwrite("examples/results/scorer_test.png", vis_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
