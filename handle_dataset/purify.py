"""
数据集格式
gloves
- train
    - images
        - 各种名字
        - *_enhanced.png（增强后的图片）
    - labels.json
- valid（同样结构，但没有增强图片）

labels.json格式
{
    "categories": [{
            "id": 1,
            "name": "blue_glove"
        },
        {
            "id": 2,
            "name": "naked_hand"
        },
        ...
    ],
    "images": [{
            "id": 77,
            "file_name": "image123.png"
        },
        # 假设这是train/labels.json，就能找到train/images/file_name的图片
        ...
    ],
    "annotations": [{
            "id": 111,
            "image_id": 77, # 这个image_id是对照的依据
            "category_id": 2,
            "bbox": [
                222.3,
                413.3,
                110.5,
                85.2
            ] # bbox存的是绝对位置，格式为x, y, w, h（区分于cx）
        }, 
        # 你应该对应写一个file_name.txt，他是Yolo格式的标签，
        class_id, cx, cy, w, h
        # 注意，class_id是从0开始的，坐标是归一化位置
        ...
    ]
}
"""

import os
import json
import shutil

def coco_to_yolo(coco_dir, yolo_dir):
    for split in ['train', 'valid']:
        coco_img_dir = os.path.join(coco_dir, split, 'images')
        coco_label_path = os.path.join(coco_dir, split, 'labels.json')
        yolo_img_dir = os.path.join(yolo_dir, split, 'images')
        yolo_label_dir = os.path.join(yolo_dir, split, 'labels')
        os.makedirs(yolo_img_dir, exist_ok=True)
        os.makedirs(yolo_label_dir, exist_ok=True)

        with open(coco_label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # category_id to yolo class_id (从0开始)
        cat_id2yolo = {cat['id']: idx for idx, cat in enumerate(data['categories'])}

        # image_id到文件名和图片尺寸的映射
        imgid2info = {}
        for img in data['images']:
            fname = img['file_name']
            if fname.endswith('_enhanced.png'):
                continue
            imgid2info[img['id']] = fname

        # 收集每张图片的所有标注
        imgid2anns = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in imgid2info:
                continue
            imgid2anns.setdefault(img_id, []).append(ann)

        for img_id, fname in imgid2info.items():
            src_img_path = os.path.join(coco_img_dir, fname)
            dst_img_path = os.path.join(yolo_img_dir, fname)
            if not os.path.exists(src_img_path):
                continue
            shutil.copyfile(src_img_path, dst_img_path)

            # 获取图片尺寸
            from PIL import Image
            with Image.open(src_img_path) as im:
                w, h = im.size

            # 写yolo标签
            yolo_label_path = os.path.join(yolo_label_dir, os.path.splitext(fname)[0] + '.txt')
            anns = imgid2anns.get(img_id, [])
            with open(yolo_label_path, 'w', encoding='utf-8') as f:
                for ann in anns:
                    class_id = cat_id2yolo[ann['category_id']]
                    x, y, bw, bh = ann['bbox']
                    # 转为yolo格式: cx, cy, w, h (归一化)
                    cx = (x + bw / 2) / w
                    cy = (y + bh / 2) / h
                    bw_norm = bw / w
                    bh_norm = bh / h
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

if __name__ == '__main__':
    coco_dir = 'datasets/coco/lab_coat'
    yolo_dir = 'datasets/yolo/lab_coat'
    coco_to_yolo(coco_dir, yolo_dir)