# 各检测器的detect_frame输出

更新于2025/8/16  
现在fps在debug模式下有些计算错误，请忽略

## safety_detect/

### wearing_detector.py

{
  'bboxes': [
    {
      'class_id': 0,
      'label': 'blue_glove',
      'score': 0.92,
      'box': [308, 285, 412, 442]
    },
    {
      'class_id': 1,
      'label': 'naked_hand',
      'score': 0.91,
      'box': [174, 412, 300, 517]
    }
  ],
  'fps': 0.33
}

### man_detector.py

{
  'poses': [
    {
      'class_id': 0,
      'label': 'person',
      'score': 0.92,
      'keypoints': [
        {'x': 192, 'y': 94, 'confidence': 0.88, 'name': 'nose'},
        ...
      ],
      'box': [13, 0, 469, 638]
    }
  ],
  'fps': 0.32,
  'man_detected': 1
}

### safety_detector.py

{
  'hand_safety': {
    'left_hand': 'glove',
    'right_hand': 'naked_hand'
  },
  'coat_safety': 'no_lab_coat',
  'fps': 0.35,
  'man_detected': 1
}

## gesture_detect/

### gesture_detector.py

{
  'hands': [
    {
      'chirality': 'left/right',
      'score': 0.91,
      'keypoints': [
        {'x': 192, 'y': 94, 'name': 'nose'},
        ...
      ]
    },
    {...}
  ],
  'fps': 0.32,
  'hand_detected': 1
}

> BUG: new mediapipe hand landmarker has no api to fetch hand detection/presence/visibility score  
> using handedness score as score. cannot provide keypoint confidence.

## vessel_detect

### vessel_detector.py

{
  'bboxes': [
    {
      'class_id': 1,
      'label': 'beaker',
      'score': 0.95,
      'box': [136, 215, 299, 435]
    },
    {
      'class_id': 0,
      'label': 'graduated cylinder',
      'score': 0.91,
      'box': [322, 79, 453, 512]
    }
  ],
  'fps': 0.33
}