# Chemistry Lab Detection System 🔬

[![wakatime](https://wakatime.com/badge/user/9af6799e-0454-4009-b789-fb07d1e221c3/project/b2a58f2a-facb-4d07-b388-e3dd5966933d.svg)](https://wakatime.com/badge/user/9af6799e-0454-4009-b789-fb07d1e221c3/project/b2a58f2a-facb-4d07-b388-e3dd5966933d)

A computer vision system for real-time detection and analysis of laboratory vessels and equipment.

## Project Overview ✨

This project employs a cascade detection approach to identify laboratory vessels and determine their pose/orientation. The system utilizes **two-stage detection**:
1. First detecting the vessel's boundary box
2. Then analyzing the vessel's precise pose and orientation

## Project Structure 📁

```
Chem_Lab_detect/
├── models/                  # Trained ML models
├── safety_detect/           # Safety detection (glove/naked_hand, lab_coat)
├── vessel_detect/           # Vessel detection (vessel_bbox, vessel_cascade)
├── xgboost_scorer/          # Action scoring (action scorer)
├── chem_lab_agent/          # LLM agent for chemical lab operations (Q&A)
├── utils/                   # Utility functions (drawing)
└── examples/                # Examples to debug on
```

## Results Display 🎉

![Safety detection result](examples/results/safety_test.png)
_Safety detection result example_

![Vessel pose detection result](examples/results/vessel_pose_test.png)
_Vessel pose detection result example_