from wearing_detect.glove_detect import GloveDetector
from wearing_detect.coat_detect import CoatDetector
from gesture_detect.gesture_detect import GestureDetector

if __name__ == "__main__":
    
    gesture_detector = GestureDetector()
    print("正在运行手势检测...")
    gesture_detector.process_video(0)  # 使用摄像头索引0