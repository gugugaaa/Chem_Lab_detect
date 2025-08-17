"""
通过修改model.model.names，来对yolo模型检测的labels重命名
yolo pose等模型没有model.names@setter，所以需要直接修改model.model.names
修改后保存到modified_*文件，所以使用记得改名字/路径！
"""

from ultralytics import YOLO

model_name = "graduated_cylinder.pt"
model = YOLO(f"models/{model_name}")
print("原始类名:", model.names)

# 修改底层模型的 names 字典
model.model.names[0] = 'graduated_cylinder' 
print("修改后的类名:", model.names)

# 保存为新文件
model.save(f'models/modified_{model_name}')