"""
两个yolo数据集
datasets/yolo/gloves
    - train
        - images
        - labels（图片同名txt）
    - valid（相同）
datasets/yolo/lab_coat
融合成一个
datasets/yolo/wearing
重命名避免文件名冲突
偏移class id避免种类冲突
"""

import os
import shutil


def count_classes(label_dir):
    """统计标签文件夹中的最大类别数（yolo class id为0~N-1）"""
    class_ids = set()
    if not os.path.exists(label_dir):
        return 0
    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        with open(os.path.join(label_dir, fname), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_ids.add(int(parts[0]))
    if not class_ids:
        return 0
    return max(class_ids) + 1


def merge_yolo_datasets(src_dirs, dst_dir):
    """
    src_dirs: 源数据集列表，如 ['datasets/yolo/gloves', 'datasets/yolo/lab_coat']
    dst_dir: 目标数据集目录，如 'datasets/yolo/wearing'
    """
    for split in ['train', 'valid']:
        for subfolder in ['images', 'labels']:
            dst_subdir = os.path.join(dst_dir, split, subfolder)
            os.makedirs(dst_subdir, exist_ok=True)

    # 统计每个数据集的类别数，计算偏移量
    class_offsets = []
    total_classes = 0
    for src in src_dirs:
        label_dir = os.path.join(src, 'train', 'labels')
        num_classes = count_classes(label_dir)
        class_offsets.append(total_classes)
        total_classes += num_classes

    for src_idx, src in enumerate(src_dirs):
        src_name = os.path.basename(src)
        class_offset = class_offsets[src_idx]
        for split in ['train', 'valid']:
            src_img_dir = os.path.join(src, split, 'images')
            src_lbl_dir = os.path.join(src, split, 'labels')
            dst_img_dir = os.path.join(dst_dir, split, 'images')
            dst_lbl_dir = os.path.join(dst_dir, split, 'labels')
            if not os.path.exists(src_img_dir):
                continue
            img_files = [f for f in os.listdir(src_img_dir) if os.path.isfile(os.path.join(src_img_dir, f))]
            img_files.sort()
            for idx, img_fname in enumerate(img_files, 1):
                name_no_ext, ext = os.path.splitext(img_fname)
                new_img_name = f"{src_name}{idx:03d}{ext}"
                src_img_path = os.path.join(src_img_dir, img_fname)
                dst_img_path = os.path.join(dst_img_dir, new_img_name)
                shutil.copy2(src_img_path, dst_img_path)

                # 同步标签并偏移class id
                label_fname = f"{name_no_ext}.txt"
                src_label_path = os.path.join(src_lbl_dir, label_fname)
                if os.path.exists(src_label_path):
                    new_label_name = f"{src_name}{idx:03d}.txt"
                    dst_label_path = os.path.join(dst_lbl_dir, new_label_name)
                    with open(src_label_path, 'r', encoding='utf-8') as fin, \
                         open(dst_label_path, 'w', encoding='utf-8') as fout:
                        for line in fin:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            # 偏移class id
                            parts[0] = str(int(parts[0]) + class_offset)
                            fout.write(' '.join(parts) + '\n')
                else:
                    print(f"Warning: Label file {src_label_path} not found for image {src_img_path}")


if __name__ == "__main__":
    srcs = [
        "datasets/yolo/gloves",
        "datasets/yolo/lab_coat",
    ]
    dst = "datasets/yolo/wearing"
    merge_yolo_datasets(srcs, dst)