import argparse
import logging
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm

import structs
import utils
from modeling.vectornet import VectorNet

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
tqdm = partial(tqdm, dynamic_ncols=True)


def eval_instance_argoverse(batch_size, args, pred, mapping, file2pred, file2labels, DEs, iter_bar):
    for i in range(batch_size):
        a_pred = pred[i]
        assert a_pred.shape == (6, args.future_frame_num, 2)
        file_name_int = int(os.path.split(mapping[i]['file_name'])[1][:-4])
        file2pred[file_name_int] = a_pred
        if not args.do_test:
            file2labels[file_name_int] = mapping[i]['origin_labels']

    if not args.do_test:
        DE = np.zeros([batch_size, args.future_frame_num])
        for i in range(batch_size):
            origin_labels = mapping[i]['origin_labels']
            for j in range(args.future_frame_num):
                DE[i][j] = np.sqrt((origin_labels[j][0] - pred[i, 0, j, 0]) ** 2 + (
                        origin_labels[j][1] - pred[i, 0, j, 1]) ** 2)
        DEs.append(DE)
        miss_rate = 0.0
        if 0 in utils.method2FDEs:
            FDEs = utils.method2FDEs[0]
            miss_rate = np.sum(np.array(FDEs) > 2.0) / len(FDEs)

        iter_bar.set_description('Iter (MR=%5.3f)' % (miss_rate))


def do_eval(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print("Loading Evalute Dataset", args.data_dir)
    if args.argoverse:
        from dataset_argoverse import Dataset
    eval_dataset = Dataset(args, args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batch_size,
                                                  sampler=eval_sampler,
                                                  collate_fn=utils.batch_list_to_batch_tensors,
                                                  pin_memory=False)
    model = VectorNet(args)
    print('torch.cuda.device_count', torch.cuda.device_count())

    logger.info("***** Recover model: %s *****", args.model_recover_path)
    if args.model_recover_path is None:
        raise ValueError("model_recover_path not specified.")

    model_recover = torch.load(args.model_recover_path)
    model.load_state_dict(model_recover)

    if 'set_predict-train_recover' in args.other_params and 'complete_traj' in args.other_params:
        model_recover = torch.load(args.other_params['set_predict-train_recover'])
        utils.load_model(model.decoder.complete_traj_cross_attention, model_recover, prefix='decoder.complete_traj_cross_attention.')
        utils.load_model(model.decoder.complete_traj_decoder, model_recover, prefix='decoder.complete_traj_decoder.')

    model.to(device)
    model.eval()
    file2pred = {}
    file2labels = {}
    iter_bar = tqdm(eval_dataloader, desc='Iter (loss=X.XXX)')
    DEs = []
    length = len(iter_bar)

    argo_pred = structs.ArgoPred()

    for step, batch in enumerate(iter_bar):
        pred_trajectory, pred_score, _ = model(batch, device)

        mapping = batch
        batch_size = pred_trajectory.shape[0]
        for i in range(batch_size):
            assert pred_trajectory[i].shape == (6, args.future_frame_num, 2)
            assert pred_score[i].shape == (6,)
            argo_pred[mapping[i]['file_name']] = structs.MultiScoredTrajectory(pred_score[i].copy(), pred_trajectory[i].copy())

        if args.argoverse:
            eval_instance_argoverse(batch_size, args, pred_trajectory, mapping, file2pred, file2labels, DEs, iter_bar)
    if 'optimization' in args.other_params:
        utils.select_goals_by_optimization(None, None, close=True)

    if args.argoverse:
        from dataset_argoverse import post_eval
        post_eval(args, file2pred, file2labels, DEs)


def main():
    parser = argparse.ArgumentParser()
    utils.add_argument(parser)
    args: utils.Args = parser.parse_args()
    utils.init(args, logger)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("device: {}".format(device))

    do_eval(args)


if __name__ == "__main__":
    main()
