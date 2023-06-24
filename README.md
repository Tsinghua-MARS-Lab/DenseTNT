# DenseTNT
### [Paper](https://arxiv.org/abs/2108.09640) | [Webpage](https://tsinghua-mars-lab.github.io/DenseTNT/)
- This is the official implementation of the paper: **DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets** (ICCV 2021).
- **DenseTNT v1.0** was released in November 1st, 2021.
- Updates: 
  - June 24th, 2023: Add evaluation metrics for Argoverse 2.
  - Sep 3, 2022: Add training code for Argoverse 2.
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


``` bash
cd src/ && cython -a utils_cython.pyx && python setup.py build_ext --inplace && cd ../
```

 ## Performance

Results on Argoverse 2:

<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh"></th>
    <th class="tg-baqh">brier-minFDE</th>
    <th class="tg-baqh">minADE</th>
    <th class="tg-baqh">minFDE</th>
    <th class="tg-baqh">MR</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh">validation set</td>
    <td class="tg-baqh">2.38</td>
    <td class="tg-baqh">1.00</td>
    <td class="tg-baqh">1.71</td>
    <td class="tg-baqh">0.216</td>
  </tr>
</tbody>
</table>


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

### 2) Evaluate
Suppose the validation data of Argoverse motion forecasting is at ```./val/data/```.

* Optimize minFDE: 
  - Add ```--do_eval --eval_params optimization MRminFDE=0.0 cnt_sample=9 opti_time=0.1``` to the end of the training command.


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