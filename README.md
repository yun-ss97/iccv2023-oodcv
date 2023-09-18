# 2nd Place Solution for ICCV 2023 OOD-CV Challenge Object Detection Track (Any Self-Supervised)

```python
# Setting conda env.
conda env create -f ood.yaml
```

```python
# split train/validation set randomly 1-fold
import os
import numpy as np
import shutil

img_list = os.listdir('data/train/Images')
idx_list = np.random.choice(range(0, len(img_list)), 1000, replace=False)

for idx in idx_list:
    shutil.copyfile(os.path.join('data/train/Images', img_list[idx]), 'data/val/Images')
    shutil.copyfile(os.path.join('data/train/Annotations', img_list[idx][:-4]+'.xml'), 'data/val/Annotations')
```

```python
# To train model with single GPU (masked autoencoder based faster-rcnn)
python train.py --model fasterrcnn_vitdet --epochs 300 --batch 16 --amp --name vitdet_large
```

```python
# To evaluate best model with validation data 
python eval.py --model fasterrcnn_vitdet --weights outputs/training/vitdet_large/best_model.pth
```

```python
# To inference on test data with best model
python inference.py --model fasterrcnn_vitdet --weights outputs/training/vitdet_large/best_model.pth
```

```python
# For multi-gpu, 
export CUDA_VISIBLE_DEVICES=0,1,2,4
python -m torch.distributed.launch --nproc_per_node 4 train.py --model fasterrcnn_vitdet --epochs 300 --batch 16 --amp --name vitdet_large
```

```python
# generate ood-like image with Segment Anything
python amg_cutmix.py
```

```python
# weight boxes fusion(WBF) to ensemble bbox prediction of trained best models
python amg_cutmix.py
```