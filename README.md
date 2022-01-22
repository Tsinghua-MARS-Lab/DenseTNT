# DenseTNT
### [Paper](https://arxiv.org/abs/2108.09640) | [Webpage](https://tsinghua-mars-lab.github.io/DenseTNT/)
- This is the official implementation of the paper: **DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets** (ICCV 2021).
- **DenseTNT v1.0** was released in November 1st, 2021.

## Quick Start

Requires:

* Python ≥ 3.6
* PyTorch ≥ 1.6

### 1) Install Packages

``` bash
 pip install -r requirements.txt
```

### 2) Install Argoverse API
The latest version of Argoverse requires Python ≥ 3.7

If using Python 3.6, you can install Argoverse v1.0 

https://github.com/argoai/argoverse-api

### 3) Compile Cython
Compile a .pyx file into a C file using Cython (already installed at step 1):


⚠️*Recompiling is needed every time the pyx files are changed.*
``` bash
cd src/ && cython -a utils_cython.pyx && python setup.py build_ext --inplace && cd ../
```

## Performance

Results on Argoverse motion forecasting validation set:

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-c3ow">minADE</th>
    <th class="tg-c3ow">minFDE</th>
    <th class="tg-c3ow">Miss Rate</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">DenseTNT w/ 100ms optimization (Miss Rate)</td>
    <td class="tg-c3ow">0.80</td>
    <td class="tg-c3ow">1.27</td>
    <td class="tg-c3ow">7.0%</td>
  </tr>
  <tr>
    <td class="tg-0pky">DenseTNT w/ 100ms optimization (minFDE)</td>
    <td class="tg-c3ow">0.73</td>
    <td class="tg-c3ow">1.05</td>
    <td class="tg-c3ow">9.8%</td>
  </tr>
  <tr>
    <td class="tg-0pky">DenseTNT w/ goal set predictor (Miss Rate)</td>
    <td class="tg-c3ow">0.82</td>
    <td class="tg-c3ow">1.37</td>
    <td class="tg-c3ow">7.0%</td>
  </tr>
  <tr>
    <td class="tg-0pky">DenseTNT w/ goal set predictor (minFDE)</td>
    <td class="tg-c3ow">0.75</td>
    <td class="tg-c3ow">1.05</td>
    <td class="tg-c3ow">9.7%</td>
  </tr>
</tbody>
</table>

## DenseTNT

### 1) Train
Suppose the training data of Argoverse motion forecasting is at ```./train/data/```.
```bash
OUTPUT_DIR=models.densetnt.1; \
GPU_NUM=8; \
python src/run.py --argoverse --future_frame_num 30 \
  --do_train --data_dir train/data/ --output_dir ${OUTPUT_DIR} \
  --hidden_size 128 --train_batch_size 64 --use_map \
  --core_num 16 --use_centerline --distributed_training ${GPU_NUM} \
  --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide lazy_points new laneGCN point_sub_graph \
    stage_one stage_one_dynamic=0.95 laneGCN-4 point_level-4-3 complete_traj complete_traj-3 \
```
Training takes 20 minutes per epoch and 5 hours for the default 16 epochs on 8 × 2080Ti. 

### 2) Evaluate
Suppose the validation data of Argoverse motion forecasting is at ```./val/data/```.

* Optimize Miss Rate:
  - Add ```--do_eval --eval_params optimization MRminFDE cnt_sample=9 opti_time=0.1``` to the end of the training command.

* Optimize minFDE: 
  - Add ```--do_eval --eval_params optimization MRminFDE=0.0 cnt_sample=9 opti_time=0.1``` to the end of the training command.


### 3) Train Set Predictor (Optional)
Compared with the optimization algorithm (default setting), the set predictor has similar performance but faster inference speed.


After training DenseTNT, suppose the model path is at ```models.densetnt.1/model_save/model.16.bin```. The command for training the set predictor is:
```bash
OUTPUT_DIR=models.densetnt.set_predict.1; \
MODEL_PATH=models.densetnt.1/model_save/model.16.bin; \
GPU_NUM=8; \
python src/run.py --argoverse --future_frame_num 30 \
  --do_train --data_dir train/data/ --output_dir ${OUTPUT_DIR} \
  --hidden_size 128 --train_batch_size 64 --use_map \
  --core_num 16 --use_centerline --distributed_training ${GPU_NUM} \
  --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide lazy_points new laneGCN point_sub_graph \
    stage_one stage_one_dynamic=0.95 laneGCN-4 point_level-4-3 complete_traj \
    set_predict=6 set_predict-6 data_ratio_per_epoch=0.4 set_predict-topk=0 set_predict-one_encoder set_predict-MRratio=1.0 \
    set_predict-train_recover=${MODEL_PATH} \
```

This training command optimizes Miss Rate. To optimize minFDE, change ```set_predict-MRratio=1.0``` in the command to ```set_predict-MRratio=0.0```.

To evaluate the set predictor, just add ```--do_eval``` to the end of this training command.

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