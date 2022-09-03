# DenseTNT
### [Paper](https://arxiv.org/abs/2108.09640) | [Webpage](https://tsinghua-mars-lab.github.io/DenseTNT/)
- This is the official implementation of the paper: **DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets** (ICCV 2021).
- **DenseTNT v1.0** was released in November 1st, 2021.
- Updates: 
  - July 25th, 2022: Add detailed code comments.

## Quick Start

Requires:

* Python ≥ 3.8
* PyTorch ≥ 1.6

### 1) Install Packages

``` bash
 pip install -r requirements.txt
```

### 2) Install Argoverse 2
Argoverse 2 requires Python ≥ 3.8

https://github.com/argoai/av2-api

### 3) Compile Cython
Compile a .pyx file into a C file using Cython (already installed at step 1):


⚠️*Recompiling is needed every time the pyx files are changed.*
``` bash
cd src/ && cython -a utils_cython.pyx && python setup.py build_ext --inplace && cd ../
```

 

## DenseTNT

### 1) Train
Suppose the training data of Argoverse motion forecasting is at ```./train/```.
```bash
OUTPUT_DIR=argoverse2.densetnt.1; \
GPU_NUM=8; \
python src/run.py --argoverse --argoverse2 --future_frame_num 60 \
  --do_train --data_dir train/ --output_dir ${OUTPUT_DIR} \
  --hidden_size 128 --train_batch_size 64 --use_map \
  --core_num 16 --use_centerline --distributed_training ${GPU_NUM} \
  --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
    lane_scoring complete_traj complete_traj-3 \
```

## Citation
If you find our work useful for your research, please consider citing the paper:
```
@inproceedings{densetnt,
  title={Densetnt: End-to-end trajectory prediction from dense goal sets},
  author={Gu, Junru and Sun, Chen and Zhao, Hang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15303--15312},
  year={2021}
}
```