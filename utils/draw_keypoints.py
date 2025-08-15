import cv2
import numpy as np


def draw_keypoints(image, detection_info, keypoint_colors=None, show_names=False, 
                   name_color=(0, 0, 0), name_bg_color=(255, 255, 255), draw_bbox=False):
    """
    根据检测信息绘制关键点到图像上
    
    参数:
        image: 原始图像
        detection_info: 检测信息字典，包含poses列表
        keypoint_colors: 关键点颜色列表，如果为None则使用默认绿色
        show_names: 是否显示关键点名称
        name_color: 关键点名称文字颜色
        name_bg_color: 关键点名称背景颜色
        draw_bbox: 是否绘制边界框
    
    返回:
        带有关键点的图像
    """
    img = image.copy()
    
    # 默认关键点颜色（绿色）
    default_color = (0, 255, 0)
    # 边界框颜色（肉色）
    bbox_color = (255, 224, 189)
    
    # 如果没有检测到任何姿态，直接返回原图
    if 'poses' not in detection_info or not detection_info['poses']:
        return img
    
    # 绘制每个检测到的姿态
    for pose in detection_info['poses']:
        keypoints = pose.get('keypoints', [])
        box = pose.get('box', None) if draw_bbox else None
        
        # 绘制边界框
        if box is not None and len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, 2)
            
            # 在边界框上方显示标签信息
            label = f"{pose.get('label', 'person')} {pose.get('score', 0):.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), bbox_color, -1)
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # 绘制关键点
        for j, kpt in enumerate(keypoints):
            x = kpt.get('x', 0)
            y = kpt.get('y', 0)
            confidence = kpt.get('confidence', None)
            name = kpt.get('name', f'kpt_{j}')
            
            # 只绘制置信度大于0.5的关键点，如果没有置信度信息则都绘制
            if confidence is None or confidence > 0.5:
                # 选择关键点颜色
                if keypoint_colors is not None and j < len(keypoint_colors):
                    color = keypoint_colors[j]
                else:
                    color = default_color
                
                # 绘制关键点
                cv2.circle(img, (int(x), int(y)), 5, color, -1)
                
                # 可选：在关键点旁边绘制名称
                if show_names:
                    # 文本位置，稍微右上偏移，避免覆盖圆点
                    text_x = int(x) + 6
                    text_y = int(y) - 6
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.4
                    thickness = 1
                    # 计算文本尺寸并绘制填充矩形作为背景
                    (text_w, text_h), baseline = cv2.getTextSize(name, font, font_scale, thickness)
                    rect_tl = (text_x - 2, text_y - text_h - 2)
                    rect_br = (text_x + text_w + 2, text_y + baseline + 2)
                    cv2.rectangle(img, rect_tl, rect_br, name_bg_color, -1)
                    cv2.putText(img, name, (text_x, text_y), font,
                                font_scale, name_color, thickness, cv2.LINE_AA)
    
    return img