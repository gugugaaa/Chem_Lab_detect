import cv2

def draw_detection_boxes(frame, detections, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制检测框
    
    Args:
        frame: 输入的图像帧
        detections: MediaPipe检测结果
        color: 边界框颜色，默认为绿色(BGR格式)
        thickness: 边界框线条粗细
        
    Returns:
        processed_frame: 处理后带有检测框的图像
    """
    processed_frame = frame.copy()
    
    for detection in detections:
        bbox = detection.bounding_box
        x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
        class_name = detection.categories[0].category_name
        score = detection.categories[0].score
        
        # 绘制边界框
        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, thickness)
        
        # 绘制标签
        label = f"{class_name} ({score:.2f})"
        cv2.putText(processed_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.75, color, thickness)
    
    return processed_frame

