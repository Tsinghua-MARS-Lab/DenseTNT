# DenseTNT
### [Paper](https://arxiv.org/abs/2108.09640) | [Webpage](https://tsinghua-mars-lab.github.io/DenseTNT/)
- This is the official implementation of the paper: **DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets** (ICCV 2021).
- **DenseTNT v1.0** was released in November 1st, 2021.
- Updates: 
  - June 24th, 2023: Add evaluation metrics for Argoverse 2.
  - Sep 3, 2022: Add training code for Argoverse 2.
  - July 25th, 2022: Add detailed code comments.

## Argoverse Version
This branch is for **Argoverse 2**. Code for **Argoverse 1** is at another [branch](https://github.com/Tsinghua-MARS-Lab/DenseTNT/tree/main).

## Quick Start

Requires:

* Python ≥ 3.8
* PyTorch ≥ 1.6

### 1) Install Packages

``` bash
 pip install -r requirements.txt
```

### 2) Install Argoverse 2
[Argoverse 2](https://github.com/argoai/av2-api) requires Python ≥ 3.8

``` bash
pip install av2
```

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
Suppose the training data of Argoverse motion forecasting is at ```./data/train/```.
```bash
OUTPUT_DIR=argoverse2.densetnt.1; \
GPU_NUM=8; \
python src/run.py --argoverse --argoverse2 --future_frame_num 60 \
  --do_train --data_dir data/train/ --output_dir ${OUTPUT_DIR} \
  --hidden_size 128 --train_batch_size 64 --use_map \
  --core_num 16 --use_centerline --distributed_training ${GPU_NUM} \
  --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
    lane_scoring complete_traj complete_traj-3 \
```

### 2) Evaluate
Suppose the validation data of Argoverse motion forecasting is at ```./data/val/```.

* Optimize minFDE: 
  - Add ```--do_eval --eval_params optimization MRminFDE=0.0 cnt_sample=9 opti_time=0.1``` to the end of the training command.

### 3) Train Set Predictor (Optional)
Compared with the optimization algorithm (default setting), the set predictor has similar performance but faster inference speed.


After training DenseTNT, suppose the model path is at ```argoverse2.densetnt.1/model_save/model.16.bin```. The command for training the set predictor is:
```bash
OUTPUT_DIR=argoverse2.densetnt.set_predict.1; \
MODEL_PATH=argoverse2.densetnt.1/model_save/model.16.bin; \
GPU_NUM=8; \
python src/run.py --argoverse --argoverse2 --future_frame_num 60 \
  --do_train --data_dir data/train/ --output_dir ${OUTPUT_DIR} \
  --hidden_size 128 --train_batch_size 64 --use_map \
  --core_num 16 --use_centerline --distributed_training ${GPU_NUM} \
  --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
    lane_scoring complete_traj \
    set_predict=6 set_predict-6 data_ratio_per_epoch=0.4 set_predict-topk=0 set_predict-one_encoder set_predict-MRratio=0.0 \
    set_predict-train_recover=${MODEL_PATH} \
```


To evaluate the set predictor, just add ```--do_eval``` to the end of this training command.

Results of the set predictor on Argoverse 2:

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
    <td class="tg-baqh">2.32</td>
    <td class="tg-baqh">0.96</td>
    <td class="tg-baqh">1.62</td>
    <td class="tg-baqh">0.233</td>
  </tr>
</tbody>
</table>

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