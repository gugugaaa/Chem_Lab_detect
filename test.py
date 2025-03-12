from vessel_detect.vessel_cascade import VesselCascadeDetector

if __name__ == "__main__":
    
    video_path = r"examples\test.mp4"

    cascade_detector = VesselCascadeDetector()
    cascade_detector.process_video(video_path)