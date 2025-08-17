<div style="max-width:300px; font-family:Arial, sans-serif;">

<h1>Chemistry Lab Detection System 🔬</h1>

<p>
A computer vision system for real-time detection and analysis of laboratory vessels and equipment.
</p>

<h2>Project Overview ✨</h2>

<p>
This project employs a cascade detection approach to identify laboratory vessels and determine their pose/orientation. The system utilizes <b>two-stage detection</b>:
<ol>
  <li>First detecting the vessel's boundary box</li>
  <li>Then analyzing the vessel's precise pose and orientation</li>
</ol>
</p>

<h2>Project Structure 📁</h2>

<pre style="font-size:13px;">
Chem_Lab_detect/
├── models/                  # Trained ML models
├── safety_detect/           # Safety detection (glove/naked_hand, lab_coat)
├── vessel_detect/           # Vessel detection (vessel_bbox, vessel_cascade)
├── chem_lab_agent/          # LLM agent for chemical lab operations (Q&amp;A)
├── utils/                   # Utility functions (drawing)
└── examples/                # Examples to debug on
</pre>

<h2>Results Display 🎉</h2>

<div style="margin-bottom:10px;">
  <img src="examples/results/safety_test.png" alt="Safety detection result" style="width:100%; max-width:300px; border:1px solid #ccc;">
  <div style="font-size:12px; color:#555;">Safety detection result example</div>
</div>

<div>
  <img src="examples/results/vessel_pose_test.png" alt="Vessel pose detection result" style="width:100%; max-width:300px; border:1px solid #ccc;">
  <div style="font-size:12px; color:#555;">Vessel pose detection result example</div>
</div>

</div>