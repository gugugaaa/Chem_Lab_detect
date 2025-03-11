import cv2

def draw_fps(frame, fps, pos=(10, 50), font=cv2.FONT_HERSHEY_SIMPLEX, 
             scale=0.75, color=(200, 200, 200), thickness=2):
    """
    在图像上绘制FPS信息
    
    Args:
        frame: 输入的图像帧
        fps: 帧率数值
        pos: 文本位置
        font: 字体
        scale: 字体大小
        color: 文本颜色
        thickness: 文本粗细
        
    Returns:
        frame: 处理后的图像
    """
    fps_label = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_label, pos, font, scale, color, thickness)
    return frame
