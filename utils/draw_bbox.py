import cv2

def draw_boxes(image, detection_info, bbox_colors=None, show_names=False, draw_bbox=True):
    """
    根据检测信息绘制检测框和标签

    Args:
        image: 输入图像
        detection_info: 检测结果信息，格式为{'bboxes': [{'class_id': int, 'label': str, 'score': float, 'box': [x1, y1, x2, y2]}, ...]}
        bbox_colors: 颜色列表，每个类别对应一个BGR颜色元组
        show_names: 是否绘制白底黑字标签
        draw_bbox: 是否绘制检测框

    Returns:
        processed_image: 处理后的图像
    """
    if not draw_bbox:
        return image

    processed_image = image.copy()
    bboxes = detection_info.get('bboxes', [])

    # 默认淡蓝色
    default_color = (255, 200, 100)
    for bbox_info in bboxes:
        class_id = bbox_info.get('class_id', 0)
        label = bbox_info.get('label', '')
        score = bbox_info.get('score', 0.0)
        x1, y1, x2, y2 = bbox_info.get('box', [0, 0, 0, 0])

        # 选择颜色
        if bbox_colors and class_id < len(bbox_colors):
            color = bbox_colors[class_id]
        else:
            color = default_color

        # 绘制边界框
        cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)

        if show_names:
            label_text = f"{label}: {score:.2f}"
            # 计算文本尺寸
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # 白底
            cv2.rectangle(processed_image, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), (255, 255, 255), -1)
            # 黑字
            cv2.putText(processed_image, label_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return processed_image

