# Chemistry Lab Detection System 🔬

A computer vision system for real-time detection and analysis of laboratory vessels and equipment.

## Project Overview ✨

This project employs a cascade detection approach to identify laboratory vessels and determine their pose/orientation. The system utilizes **two-stage detection**:
1. First detecting the vessel's boundary box
2. Then analyzing the vessel's precise pose and orientation

## Key Features 🚀

- **Real-time vessel detection** in video streams
- Advanced cascade detection framework (boundary box → pose detection)
- Performance monitoring with FPS calculation and display
- Comprehensive support for **multiple vessel types**
- High accuracy pose estimation

## Project Structure 📁

```
Chem_Lab_detect/
├── models/                  # Trained ML models
├── safety_detect/           # Safety detection (glove/naked_hand, lab_coat)
├── vessel_detect/           # Vessel detection (vessel_bbox, vessel_cascade)
├── chem_lab_agent/          # LLM agent for chemical lab operations (Q&A)
├── utils/                   # Utility functions (drawing)
└── examples/                # Examples to debug on
```
