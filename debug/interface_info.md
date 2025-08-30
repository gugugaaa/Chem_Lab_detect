# 接口文档
更新于2025/8/30 
现在fps在debug模式下有些计算错误，请忽略

## safety_detect/

### wearing_detector.py

### safety_detector.py

SafetyDetector.detect_frame()

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

GestureDetector.detect_frame()

{
  'hands': [
    {
      'chirality': 'left/right',
      'score': 0.91,
      'box': [136, 215, 299, 435],
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

> 由于mediapipe的api限制：
> - 关键点无法获得conf字段
> - 手的score由手性置信度替代

## vessel_detect

### vessel_cascade_detector.py

VesselCascadeDetector.detect_frame()

{
  'vessels': [
    {
      'class_id': 1,
      'label': 'beaker',
      'score': 0.95,
      'box': [136, 215, 299, 435],
      'keypoints': [
        {'x': 192, 'y': 94, 'confidence': 0.88, 'name': 'tip'},
        ...
      ],
    },
    ...]
  'fps': 0.33
}
> YOLO关键点检测模型的keypoint confidence字段可能为None，需在后处理时兼容。

### vessel_keypoint_corrector.py

VesselKeypointCorrector.correct_vessel_keypoints()

仅更新vessel_info的keypoints字段，格式不变

## xgboost_scorer/

### action_scorer.py

ActionScorer.score_frame()
（内部调用FeatureExtractor.extract()）

{
  "operation": "beaker+graduated_cylinder",
  "score": 95.5
}

> 假设一张图片只有一人操作。因此返回一份操作的评分信息