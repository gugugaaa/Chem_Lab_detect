from wearing_detect.glove_detect import GloveDetector
from wearing_detect.coat_detect import CoatDetector
from gesture_detect.gesture_detect import GestureDetector
from vessel_detect.vessel_bbox import VesselDetector
from vessel_detect.vessel_pose import VesselPoseDetector

if __name__ == "__main__":
    
    video_path = r"examples\test.mp4"

    vessel_pose_detector = VesselPoseDetector()
    vessel_pose_detector.process_video(video_path, display=True)