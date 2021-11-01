# DenseTNT
### [Paper](https://arxiv.org/abs/2108.09640) | [Webpage](https://tsinghua-mars-lab.github.io/DenseTNT/)
- This is the official implementation of the paper: **DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets** (ICCV 2021).
- **DenseTNT v1.0** was released in November 1st, 2021.

## Quick Start

Requires:

* Python 3.6+
* pytorch 1.6+

### 1) Install packages

``` bash
 pip install -r requirements.txt
```

### 2) Install Argoverse API

https://github.com/argoai/argoverse-api

### 3) Compile Cython
Compile a .pyx file into a C file using Cython:


⚠️*Recompiling is needed every time the pyx files are changed.*
``` bash
cd src/
cython -a utils_cython.pyx && python setup.py build_ext --inplace
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
    <td class="tg-0pky">DenseTNT w/ 100ms optimization</td>
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
    <td class="tg-0pky">DenseTNT w/ goal set predictor (online)</td>
    <td class="tg-c3ow">0.82</td>
    <td class="tg-c3ow">1.37</td>
    <td class="tg-c3ow">7.0%</td>
  </tr>
</tbody>
</table>

## Models

Suppose the training data of Argoverse motion forecasting is at ```./train/data/```.

### DenseTNT

#### Train
```bash
OUTPUT_DIR=models.densetnt.1; \
python src/run.py --argoverse --future_frame_num 30 \
--do_train --data_dir train/data/ --output_dir ${OUTPUT_DIR} \
--hidden_size 128 --train_batch_size 64 --sub_graph_batch_size 4096 --use_map \
--core_num 16 --use_centerline --other_params semantic_lane direction l1_loss \
goals_2D enhance_global_graph subdivide lazy_points new laneGCN point_sub_graph \
stage_one stage_one_dynamic=0.95 laneGCN-4 point_level point_level-4 \
point_level-4-3 complete_traj complete_traj-3 \
```

#### Evaluate

Add ```--do_eval --eval_params optimization MRminFDE cnt_sample=9 opti_time=0.1``` to the end of the training command.


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