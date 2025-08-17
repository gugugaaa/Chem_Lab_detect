"""
先用models/vessel_box.pt检测画面
选择检测的仪器类别
对于置信度第一大的这个类别的bbox，向外5像素裁切后，zero padding得到正方形画面（分辨率不变）
然后debug单个模型的表现
加载example/test/beaker[volumetric_flask/graduated_cylinder].png
加载models/beaker[volumetric_flask/graduated_cylinder].pt
推理这张图片
绘制结果
（使用Yolo的api）
model = YOLO(model_path)
results = model.predict(frame, conf, imgsz=224)
results.plot()
"""

from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 选择检测的仪器类别
instrument = 'volumetric_flask'  # 可选: 'beaker', 'volumetric_flask', 'graduated_cylinder'

# 2. 加载图片和模型路径
img_path = f'examples/test/{instrument}.png'
model_path = f'models/{instrument}.pt'
vessel_box_model_path = 'models/vessel_box.pt'

# 3. 加载模型
model = YOLO(model_path)
print(model.names)
vessel_box_model = YOLO(vessel_box_model_path)
print(vessel_box_model.names)

# 4. 读取图片
frame = cv2.imread(img_path)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# 5. vessel_box.pt 检测所有仪器，获取bbox
vessel_results = vessel_box_model.predict(frame_rgb, conf=0.25)
boxes = vessel_results[0].boxes
if boxes is None or len(boxes) == 0:
    print("未检测到任何仪器")
    exit(1)

# 6. 选择置信度最高的目标
scores = boxes.conf.cpu().numpy()
idx = np.argmax(scores)
box = boxes[idx]
x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
h, w, _ = frame_rgb.shape

# 7. 向外扩展5像素
pad = 5
x1p = max(x1 - pad, 0)
y1p = max(y1 - pad, 0)
x2p = min(x2 + pad, w - 1)
y2p = min(y2 + pad, h - 1)

# 8. 裁剪并zero padding为正方形
crop = frame_rgb[y1p:y2p, x1p:x2p]
ch, cw, _ = crop.shape
side = max(ch, cw)
pad_top = (side - ch) // 2
pad_bottom = side - ch - pad_top
pad_left = (side - cw) // 2
pad_right = side - cw - pad_left
crop_square = np.pad(
    crop,
    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
    mode='constant',
    constant_values=0
)

# 9. 用选定仪器模型推理
results = model.predict(crop_square, conf=0.25, imgsz=224)

# 10. 仅绘制中心点在正方形中间区域的结果（用yolo的plot api）
side = crop_square.shape[0]
center_min = side // 2-10
center_max = side //2 + 10

for r in results:
    # 过滤中心点(cx, cy)在正方形中心区域的检测框
    filtered_indices = []
    if hasattr(r, 'boxes') and r.boxes is not None:
        for i, box in enumerate(r.boxes):
            cx, cy = box.xywh[0][:2].cpu().numpy()
            if center_min <= cx <= center_max and center_min <= cy <= center_max:
                filtered_indices.append(i)
    # 构造只包含过滤后boxes的新结果对象
    if filtered_indices:
        # 只保留filtered_indices对应的boxes
        r.boxes = r.boxes[filtered_indices]
    else:
        # 没有符合条件的，清空boxes
        r.boxes = None
    im_plot = r.plot()
    plt.imshow(im_plot)
    plt.axis('off')
    plt.title(f'Result: {instrument} (center filtered)')
    plt.show()