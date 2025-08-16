# 各检测器的detect_frame输出

更新于2025/8/16
现在fps在debug模式下有些计算错误，请忽略

## safety_detect/

### wearing_detector.py

{'bboxes': [{'class_id': 0, 'label': 'blue_glove', 'score': 0.92, 'box': [308, 285, 412, 442]}, {'class_id': 1, 'label': 'naked_hand', 'score': 0.91, 'box': [174, 412, 300, 517]}], 'fps': 0.33}  

### man_detector.py

{'poses': [{'class_id': 0, 'label': 'person', 'score': 0.9, 'keypoints': [{'x': 192.46, 'y': 94.95, 'confidence': 1.0, 'name': 'nose'}, ...], 'box': [13.46, 0.0, 469.62, 638.0]}], 'fps': 0.32, 'man_detected': 1}

### safety_detector.py

{'hand_safety': {'left_hand': 'glove', 'right_hand': 'naked_hand'}, 'coat_safety': 'no_lab_coat', 'fps': 0.35, 'man_detected': 1}

## gesture_detect/

### gesture_detector.py
{'hand_landmarks': [[(157, 99), ...], [(50, 162), ...]], 'handedness': [[Category(index=1, score=0.97, display_name='Left', category_name='Left')], [Category(index=0, score=0.95, display_name='Right', category_name='Right')]], 'hands_detected': 2, 'fps': 8.86}


{'hands': [{'chirality': 'left/right', 'score': 0.9, 'keypoints': [{'x': 192.46, 'y': 94.95, 'name': 'nose'}, ...]}, {...}], 'fps': 0.32, 'hand_detected': 1}