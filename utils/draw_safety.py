"""
绘制 safety_detector 结果
传入safety_detector的detection_info，解码

手部关键点：
glove——绿色
naked_hand——红色

衣服：对于man pose的非头部关键点，连线
invisible——黄色连线
lab_coat——绿色连线
no_lab_coat——红色连线

可选show_status——
正常——白底黑字
异常——淡黄底深蓝字，
显示在手部关键点右上角/这个人box内的右上角
调试信息——手部left/right hand invisible或too close——显示在这个人的box内的左上角
（维护一个管理调试信息显示的位置，避免多条调试信息导致重叠绘制。）
"""
import cv2
import numpy as np


def draw_safety(image, detection_info, poses_info=None, show_status=True, show_debug=True):
    """
    根据安全检测信息绘制结果到图像上
    
    参数:
        image: 原始图像
        detection_info: 安全检测信息字典，包含hand_safety和coat_safety
        poses_info: 姿态信息列表，用于绘制连线和状态显示
        show_status: 是否显示状态信息
        show_debug: 是否显示调试信息
    
    返回:
        带有安全检测结果的图像
    """
    img = image.copy()
    
    # 颜色定义
    colors = {
        'glove': (0, 255, 0),      # 绿色
        'naked_hand': (0, 0, 255), # 红色
        'invisible': (0, 255, 255), # 黄色
        'lab_coat': (0, 255, 0),    # 绿色
        'no_lab_coat': (0, 0, 255), # 红色
        'too_close': (0, 165, 255)  # 橙色
    }
    
    # 状态显示颜色
    normal_bg = (255, 255, 255)    # 白底
    normal_text = (0, 0, 0)        # 黑字
    abnormal_bg = (102, 255, 255)  # 淡黄底
    abnormal_text = (139, 69, 19)  # 深蓝字
    debug_bg = (200, 200, 200)     # 灰底
    debug_text = (0, 0, 0)         # 黑字
    
    # 人体框颜色（BGR）
    person_box_color = (189, 224, 255)  # 浅肉色
    
    if not poses_info or len(poses_info) == 0:
        return img
    
    hand_safety = detection_info.get('hand_safety', {})
    coat_safety = detection_info.get('coat_safety', 'invisible')
    
    # 处理每个人的姿态
    for pose in poses_info:
        keypoints = pose.get('keypoints', [])
        box = pose.get('box', None)
        
        if not keypoints:
            continue
        
        # 绘制人体框
        if box:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), person_box_color, 2)
            
        # 绘制手部关键点（覆盖原有颜色）
        _draw_hand_keypoints(img, keypoints, hand_safety, colors)
        
        # 绘制衣服连线
        _draw_coat_connections(img, keypoints, coat_safety, colors)
        
        # 显示状态和调试信息
        if show_status or show_debug:
            _draw_status_info(img, pose, hand_safety, coat_safety, 
                            show_status, show_debug, normal_bg, normal_text,
                            abnormal_bg, abnormal_text, debug_bg, debug_text)
    
    return img


def _draw_hand_keypoints(img, keypoints, hand_safety, colors):
    """绘制手部关键点"""
    hand_keypoints = ['left_wrist', 'right_wrist']
    hand_sides = ['left_hand', 'right_hand']
    
    for hand_kpt_name, hand_side in zip(hand_keypoints, hand_sides):
        # 找到对应的关键点
        hand_kpt = next((kpt for kpt in keypoints if kpt['name'] == hand_kpt_name), None)
        if hand_kpt and hand_kpt.get('confidence', 0) > 0.5:
            x, y = int(hand_kpt['x']), int(hand_kpt['y'])
            safety_status = hand_safety.get(hand_side, 'invisible')
            color = colors.get(safety_status, colors['invisible'])
            # 绘制较大的圆点来覆盖原有的关键点
            cv2.circle(img, (x, y), 8, color, -1)
            cv2.circle(img, (x, y), 8, (0, 0, 0), 2)  # 黑色边框


def _draw_coat_connections(img, keypoints, coat_safety, colors):
    """绘制衣服连线（非头部关键点）"""
    # 定义连接关系（排除头部关键点）
    connections = [
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('right_shoulder', 'right_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('right_hip', 'right_knee'),
        ('left_knee', 'left_ankle'),
        ('right_knee', 'right_ankle')
    ]
    
    # 定义衣服部分的关键点（非头部关键点）
    coat_keypoints = [
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle'
    ]
    
    color = colors.get(coat_safety, colors['invisible'])
    
    # 绘制连线
    for start_name, end_name in connections:
        start_kpt = next((kpt for kpt in keypoints if kpt['name'] == start_name), None)
        end_kpt = next((kpt for kpt in keypoints if kpt['name'] == end_name), None)
        
        if (start_kpt and start_kpt.get('confidence', 0) > 0.5 and
            end_kpt and end_kpt.get('confidence', 0) > 0.5):
            start_point = (int(start_kpt['x']), int(start_kpt['y']))
            end_point = (int(end_kpt['x']), int(end_kpt['y']))
            cv2.line(img, start_point, end_point, color, 2)
    
    # 绘制衣服部分的关键点
    for kpt_name in coat_keypoints:
        kpt = next((kpt for kpt in keypoints if kpt['name'] == kpt_name), None)
        if kpt and kpt.get('confidence', 0) > 0.5:
            x, y = int(kpt['x']), int(kpt['y'])
            cv2.circle(img, (x, y), 3, color, -1)


def _draw_status_info(img, pose, hand_safety, coat_safety, show_status, show_debug,
                     normal_bg, normal_text, abnormal_bg, abnormal_text, debug_bg, debug_text):
    """绘制状态和调试信息"""
    box = pose.get('box')
    keypoints = pose.get('keypoints', [])
    
    if not box:
        return
    
    x1, y1, x2, y2 = map(int, box)
    
    # 收集调试信息
    debug_messages = []
    if show_debug:
        left_status = hand_safety.get('left_hand', 'invisible')
        right_status = hand_safety.get('right_hand', 'invisible')
        
        if left_status == 'invisible':
            debug_messages.append("Left hand invisible")
        if right_status == 'invisible':
            debug_messages.append("Right hand invisible")
        if coat_safety == 'too_close':
            debug_messages.append("Too close to camera")
        
        # 添加整体安全状态到调试信息
        is_abnormal = (left_status == 'naked_hand' or 
                      right_status == 'naked_hand' or 
                      coat_safety == 'no_lab_coat')
        overall_status = "UNSAFE" if is_abnormal else "SAFE"
        debug_messages.append(f"Overall: {overall_status}")
    
    # 绘制调试信息（左上角）
    debug_y_offset = y1 + 20
    for msg in debug_messages:
        _draw_text_with_background(img, msg, (x1 + 5, debug_y_offset), 
                                 debug_bg, debug_text, 0.5)
        debug_y_offset += 25
    
    # 绘制状态信息
    if show_status:
        # 绘制每只手的状态
        _draw_hand_status(img, keypoints, hand_safety, normal_bg, normal_text, 
                         abnormal_bg, abnormal_text)
        
        # 绘制衣服状态（在人框右上角）
        if coat_safety != 'invisible':
            is_coat_abnormal = (coat_safety == 'no_lab_coat')
            bg_color = abnormal_bg if is_coat_abnormal else normal_bg
            text_color = abnormal_text if is_coat_abnormal else normal_text
            
            coat_text = "lab_coat" if coat_safety == 'lab_coat' else "no_lab_coat"
            _draw_text_with_background(img, coat_text, (x2 - 100, y1 + 20), 
                                     bg_color, text_color, 0.6)


def _draw_hand_status(img, keypoints, hand_safety, normal_bg, normal_text, 
                     abnormal_bg, abnormal_text):
    """绘制每只手的状态"""
    hand_mappings = [
        ('left_wrist', 'left_hand'),
        ('right_wrist', 'right_hand')
    ]
    
    for hand_kpt_name, hand_side in hand_mappings:
        hand_kpt = next((kpt for kpt in keypoints if kpt['name'] == hand_kpt_name), None)
        if hand_kpt and hand_kpt.get('confidence', 0) > 0.5:
            hand_status = hand_safety.get(hand_side, 'invisible')
            if hand_status != 'invisible':
                # 判断是否异常
                is_hand_abnormal = (hand_status == 'naked_hand')
                bg_color = abnormal_bg if is_hand_abnormal else normal_bg
                text_color = abnormal_text if is_hand_abnormal else normal_text
                
                # 状态文本
                status_text = "glove" if hand_status == 'glove' else "naked"
                
                # 显示在手部关键点右上角
                pos_x = int(hand_kpt['x']) + 10
                pos_y = int(hand_kpt['y']) - 10
                _draw_text_with_background(img, status_text, (pos_x, pos_y), 
                                         bg_color, text_color, 0.5)


def _draw_text_with_background(img, text, pos, bg_color, text_color, scale):
    """绘制带背景的文本"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    
    # 计算文本尺寸
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    # 绘制背景矩形
    x, y = pos
    cv2.rectangle(img, (x - 5, y - text_h - 5), 
                 (x + text_w + 5, y + baseline + 5), bg_color, -1)
    
    # 绘制文本
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)