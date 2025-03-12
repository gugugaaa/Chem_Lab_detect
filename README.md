# Chemistry Lab Detection System 🔬

A computer vision system for detecting and analyzing laboratory vessels and equipment in real-time.

## Project Overview ✨

This project uses a cascade detection approach to identify laboratory vessels and determine their pose/orientation. The system employs **two-stage detection**:
1. First detecting the vessel's boundary box
2. Then analyzing the vessel's precise pose and orientation

## Key Features 🚀

- **Real-time vessel detection** in video streams
- Cascade detection framework (boundary box → pose detection)
- FPS calculation and display
- Support for **multiple vessel types**

## Project Structure 📁

```
Chem_Lab_detect/
├── models/                  # Trained ML models
│   ├── vessels-bbox-nano.pt # Vessel boundary box detection model
│   └── vessel-pose-nano.pt  # Vessel pose detection model
├── vessel_detect/           # Core detection modules
│   ├── vessel_bbox.py       # Boundary box detection
│   ├── vessel_pose.py       # Pose detection
│   └── vessel_cascade.py    # Cascade detection pipeline
├── utils/                   # Utility functions
│   ├── fps_caculator.py     # FPS calculation
│   └── draw_fps.py          # FPS display functions
```

## Usage 💻

Basic usage example:

```python
from vessel_detect.vessel_cascade import VesselCascadeDetector

# Initialize the detector with default models
detector = VesselCascadeDetector()

# Process video from webcam
detector.process_video(0)  # Use camera index 0

# Process a video file
detector.process_video("path/to/video.mp4")

# Process a single frame
processed_frame, detection_info = detector.detect_frame(image)
```

## Requirements ⚙️

- OpenCV
- NumPy
- PyTorch (for the detection models)

## Installation 📋

Clone the repository and ensure you have the **required models** in the `models/` directory:

```bash
git clone https://github.com/yourusername/Chem_Lab_detect.git
cd Chem_Lab_detect
```

---

## A Light Moment 🌸

```
試験管と
カメラの目が
見つめ合う
```

*Test tubes and flasks  
The camera's watchful eye  
They meet in silence*
