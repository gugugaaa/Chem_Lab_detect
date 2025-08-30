import numpy as np
import cv2

def fit_line_and_correct(points, keep_idx=None):
    """
    points: list of (x, y) dicts, e.g. [{'x': 1, 'y': 2, ...}, ...]
    keep_idx: 不参与拟合、保持原位的点索引（如 tip）
    返回矫正后的点列表
    """
    if keep_idx is None:
        keep_idx = []
    arr = np.array([[p['x'], p['y']] for i, p in enumerate(points) if i not in keep_idx])
    if len(arr) < 2:
        return points
    # 拟合直线 y = kx + b, 用最小二乘法
    [vx, vy, x0, y0] = cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(vx), float(vy)
    x0, y0 = float(x0), float(y0)
    # 直线方向向量 (vx, vy)
    corrected = []
    for i, p in enumerate(points):
        if i in keep_idx:
            corrected.append(p.copy())
            continue
        px, py = p['x'], p['y']
        # 投影到直线
        t = ((px - x0) * vx + (py - y0) * vy)
        x_proj = x0 + t * vx
        y_proj = y0 + t * vy
        new_p = p.copy()
        new_p['x'] = int(round(x_proj))
        new_p['y'] = int(round(y_proj))
        corrected.append(new_p)
    return corrected

def correct_vessel_keypoints(info):
    """
    输入 info: vessel_cascade.py 输出的 detection_info 字典
    返回矫正后的 info（格式不变）
    """
    import cv2
    vessel_types = {
        'graduated_cylinder': {
            'fit_idx': [1,2,3,4], # 除 tip 外
            'keep_idx': [0]
        },
        'volumetric_flask': {
            'fit_idx': [0,1,3], # 除 stopper 外
            'keep_idx': [2]
        },
        'beaker': {
            'fit_idx': [1,2], # 除 tip 外
            'keep_idx': [0]
        }
    }
    info = info.copy()
    poses = info.get('poses', [])
    for pose in poses:
        label = pose.get('label')
        keypoints = pose.get('keypoints', [])
        if label in vessel_types:
            fit_idx = vessel_types[label]['fit_idx']
            keep_idx = vessel_types[label]['keep_idx']
            # 只对指定点矫正
            corrected = fit_line_and_correct(keypoints, keep_idx=keep_idx)
            pose['keypoints'] = corrected
    return info