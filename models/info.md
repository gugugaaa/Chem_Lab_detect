# 各模型model.names
> pose模型不会储存keypoints的label，需手动加载，因此也可以自己指定

## safety
### wearing.pt
{0: 'blue_glove', 1: 'naked_hand', 2: 'lab_coat'}

## vessel

### vessel_box.pt
{0: 'graduated_cylinder', 1: 'beaker', 2: 'volumetric_flask'}

### beaker.pt
{0: 'beaker'}

#### beaker pose keypoints
{0: 'tip', 1: 'mouth_center', 2: 'bottom_center'}

### graduated_cylinder.pt
{0: 'graduated_cylinder'}

#### graduated_cylinder pose keypoints
{0: 'tip', 1: 'mouth_center', 2: 'bottom_outer', 3: 'top_quarter', 4: 'bottom_quarter'}

### volumetric_flask.pt
{0: 'volumetric_flask'}

#### volumetric_flask pose keypoints
{0: 'bottom_center', 1: 'mouth_center', 2: 'stopper', 3: 'scale_mark'}