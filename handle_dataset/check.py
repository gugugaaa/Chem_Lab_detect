import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 预置6种颜色
COLORS = [
    (255, 0, 0),      # 红
    (0, 255, 0),      # 绿
    (0, 0, 255),      # 蓝
    (255, 255, 0),    # 黄
    (255, 0, 255),    # 紫
    (0, 255, 255),    # 青
]

def plot_yolo_bboxes(img_dir, label_dir, n=8):
    # 支持中文路径
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(img_files) < n:
        n = len(img_files)
    samples = random.sample(img_files, n)

    plt.figure(figsize=(16, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    plt.rcParams['axes.unicode_minus'] = False

    for idx, img_name in enumerate(samples):
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')
        # 读取图片
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        img_np = np.array(img)

        # 绘制bbox
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    # 反归一化
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    color = COLORS[class_id] if class_id < 5 else COLORS[5]
                    # 绘制矩形
                    img_np = cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
                    # 绘制类别id
                    label_text = f'class{class_id}'
                    img_np = cv2.putText(img_np, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, lineType=cv2.LINE_AA)

        plt.subplot(2, 4, idx+1)
        plt.imshow(img_np)
        plt.title(img_name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import cv2  
    # 修改为你的图片和标签路径
    img_dir = r'datasets/yolo/wearing/valid/images'
    label_dir = r'datasets/yolo/wearing/valid/labels'
    plot_yolo_bboxes(img_dir, label_dir)
