from vessel_detect.vessel_cascade import VesselCascadeDetector
from gesture_detect.gesture_detector import GestureDetector
import cv2
import numpy as np
from utils.draw_fps import draw_fps
from utils.draw_hand import draw_landmarks
from utils.draw_keypoints import draw_keypoints

# 视频源，可以是摄像头索引(如0)或视频文件路径
video_source = 'examples/test.mp4'

# 初始化两个检测器
vessel_cascade_detector = VesselCascadeDetector()
gesture_detector = GestureDetector()

def process_video(video_source=0, display=True):
    """
    同时处理视频流进行手势和容器检测
    
    Args:
        video_source: 视频源，可以是摄像头索引或视频文件路径
        display: 是否显示处理结果
    """
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频源 {video_source}")
        return
        
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 复制原始帧用于分别处理
            frame_for_vessel = frame.copy()
            frame_for_gesture = frame.copy()
            
            # 1. 进行容器级联检测
            vessel_frame, vessel_info = vessel_cascade_detector.detect_frame(frame_for_vessel)
            
            # 2. 进行手势检测
            gesture_frame, gesture_info = gesture_detector.detect_frame(frame_for_gesture)
            
            # 3. 合并两个结果到原始帧
            result_frame = frame.copy()
            
            # 绘制容器检测结果
            # 方法一：直接使用处理后的vessel_frame（已经包含了关键点和边界框）
            # result_frame = vessel_frame.copy()
            
            # 方法二：手动绘制容器检测的边界框和关键点
            for result in vessel_info.get('results', []):
                # 绘制边界框
                bbox = result.get('bbox', {})
                x1, y1, x2, y2 = bbox.get('x1', 0), bbox.get('y1', 0), bbox.get('x2', 0), bbox.get('y2', 0)
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加标签
                label = f"{bbox.get('category', 'Unknown')}: {bbox.get('score', 0):.2f}"
                cv2.putText(result_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 提取姿态信息中的关键点进行绘制
                pose_info = result.get('pose', {})
                poses = pose_info.get('poses', [])
                
                if poses:
                    for pose in poses:
                        keypoints_data = pose.get('keypoints', [])
                        if keypoints_data:
                            # 准备关键点数据，用于draw_keypoints函数
                            keypoints_list = []
                            keypoint_conf_list = []
                            
                            # 收集关键点坐标和置信度
                            kpts = []
                            confs = []
                            for kp in keypoints_data:
                                kpts.append([kp.get('x', 0), kp.get('y', 0)])
                                confs.append(kp.get('confidence', 1.0))
                            
                            # 添加到列表
                            keypoints_list.append(kpts)
                            keypoint_conf_list.append(confs)
                            
                            # 类别和置信度
                            classes_list = [pose.get('class_id', 0)]
                            box_conf_list = [pose.get('score', 1.0)]
                            
                            # 使用draw_keypoints绘制关键点
                            result_frame = draw_keypoints(
                                result_frame,
                                np.array(keypoints_list),
                                np.array(keypoint_conf_list),
                                np.array(classes_list),
                                np.array(box_conf_list)
                            )
            
            # 绘制手势检测的关键点
            if gesture_info.get('hand_landmarks'):
                for landmark_point in gesture_info.get('hand_landmarks', []):
                    # 使用draw_landmarks绘制手部关键点
                    result_frame = draw_landmarks(result_frame, landmark_point)
            
            # 在结果帧上显示FPS信息
            fps = min(vessel_info.get('fps', 0), gesture_info.get('fps', 0))  # 取较小的FPS作为整体FPS
            result_frame = draw_fps(result_frame, fps)
            
            # 显示处理结果
            if display:
                cv2.imshow("双重检测 (容器+手势)", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                    break
                    
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video(video_source)

