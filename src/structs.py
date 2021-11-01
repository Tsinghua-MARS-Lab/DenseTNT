import os.path
import pickle
from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np


def save(obj, output_dir, eval_identifier, prefix=None):
    struct = obj.__class__.__name__
    prefix = f'{prefix}.{struct}' if prefix is not None else struct
    with open(os.path.join(output_dir, f'{prefix}.{eval_identifier}'), 'wb') as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


ScenarioID = int
ObjectID = int

Base = NamedTuple


class ScoredTrajectory(Base):
    score: float
    traj: np.ndarray


class MultiScoredTrajectory(Base):
    scores: np.ndarray
    trajs: np.ndarray


class AutoregScoredTrajectory(Base):
    base_traj: ScoredTrajectory
    pred: MultiScoredTrajectory


class AutoregStruct(Base):
    pred_3s_trajs: MultiScoredTrajectory
    idx_in_K_2_scores_vectors_2D_normalizer: Dict[int, Tuple[np.ndarray, np.ndarray, Any]]
    pack: Tuple


FileName = str


class ArgoPred(Dict[FileName, MultiScoredTrajectory]):
    pass
