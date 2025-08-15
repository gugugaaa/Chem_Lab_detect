from ultralytics import YOLO

yaml_content = """
path: /kaggle/input/wearing
train: train/images
val: valid/images
names:
  0: blue_glove
  1: naked_hand
  2: lab_coat
"""

# 写入到/kaggle/working/wearing.yaml
with open('/kaggle/working/wearing.yaml', 'w', encoding='utf-8') as f:
    f.write(yaml_content)
    print("write/overwrite wearing.yaml")

# 加载YOLO11n模型
model = YOLO('/kaggle/input/yolo11/pytorch/default/1/yolo11n.pt')

# 训练模型
results = model.train(
    data="/kaggle/working/wearing.yaml",
    epochs=50,
    imgsz=224,  # 匹配工作场景
    device=[0, 1],
    batch=-1,  # 自动批大小
    optimizer='AdamW',
    lr0=0.001,
    patience=20,
    workers=8,
    cache=True,
    augment=True,  # 启用简单增强（如旋转、模糊等）
    save_period=10,
    project='runs/train',
    name='wearing_exp'
)