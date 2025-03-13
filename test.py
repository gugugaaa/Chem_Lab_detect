from vessel_detect.vessel_cascade import VesselCascadeDetector

video_source = 'examples/test.mp4'
vessel_cascade = VesselCascadeDetector()
vessel_cascade.process_video(video_source, display=True)