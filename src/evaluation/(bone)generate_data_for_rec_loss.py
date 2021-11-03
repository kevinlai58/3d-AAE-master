import argparse
import json
import logging
import random
from importlib import import_module
from os.path import join

import numpy as np
import torch
import h5py
from torch.distributions import Beta
from torch.utils.data import DataLoader
####################################
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 当前程序上一级目录，这里为mycompany
sys.path.append(BASE_DIR)
####################################
# from datasets.shapenet import ShapeNetDataset
# from utils.loggers.basic_logger import setup_logging
from myutils.util import find_latest_epoch, cuda_setup, setup_logging
from myutils.h5_loader import load_data_h5, MakeBatchData
from myutils.normalize_points import rescale


def get_recloss_latentcode(eval_config, filepath):
    # Load hyperparameters as they were during training
    train_results_path = join(eval_config['results_root'], eval_config['arch'],
                              eval_config['experiment_name'])
    with open(join(train_results_path, 'config.json')) as f:
        train_config = json.load(f)

    random.seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed_all(train_config['seed'])

    setup_logging(join(train_results_path, 'results'))
    log = logging.getLogger(__name__)

    weights_path = join(train_results_path, 'weights')
    if eval_config['epoch'] == 0:
        epoch = find_latest_epoch(weights_path)
    else:
        epoch = eval_config['epoch']
    log.debug(f'Starting from epoch: {epoch}')

    device = cuda_setup(eval_config['cuda'], eval_config['gpu'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    # reference dataset
    filename ="D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048\\scaphoid_models_aligned.h5"
    with h5py.File(filename, "r") as f:
        # List all groups
        # print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())
        # Get the data
        faces_list = list(f[a_group_key[0]])
        points_list = list(f[a_group_key[1]])

        normalized_points_list = []
        for ele in points_list:
            normalized_points, _ = rescale(ele)
            normalized_points_list.append(normalized_points)

    Xref = torch.tensor(normalized_points_list).float().to(device)
    if Xref.size(-1) == 3:
        Xref.transpose_(Xref.dim() - 2, Xref.dim() - 1)

    # Dataset to be compared
    filename = filepath
    with h5py.File(filename, "r") as f:
        # List all groups
        # print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())
        # Get the data
        faces_list = list(f[a_group_key[0]])
        points_list = list(f[a_group_key[1]])

        normalized_points_list = []
        for ele in points_list:
            normalized_points, _ = rescale(ele)
            normalized_points_list.append(normalized_points)


    # path = "D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048/"
    # X, partial = load_data_h5(path, "valid")
    #
    # X = X[0:200]
    if filename == "D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048\\scaphoid_models_aligned.h5":
        X = torch.tensor(normalized_points_list).float().to(device)
    else:
        X = torch.tensor(points_list).float().to(device)
    if X.size(-1) == 3:
        X.transpose_(X.dim() - 2, X.dim() - 1)


    if 'distribution' in train_config:
        distribution = train_config['distribution']
    elif 'distribution' in eval_config:
        distribution = eval_config['distribution']
    else:
        log.warning('No distribution type specified. Assumed normal = N(0, 0.2)')
        distribution = 'normal'

    #
    # Models
    #
    # arch = import_module(f"model.architectures.{eval_config['arch']}")
    arch = import_module(f"models.{train_config['arch']}")
    E = arch.Encoder(train_config).to(device)
    G = arch.Generator(train_config).to(device)

    if eval_config['reconstruction_loss'].lower() == 'chamfer':
        from losses.champfer_loss import ChamferLoss2
        reconstruction_loss = ChamferLoss2().to(device)
    elif eval_config['reconstruction_loss'].lower() == 'earth_mover':
        from eval_config.earth_mover_distance import EMD
        reconstruction_loss = EMD().to(device)
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {eval_config["reconstruction_loss"]}')
    #

    #
    # Load saved state
    #
    E.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_E.pth')))
    G.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_G.pth')))

    E.eval()
    G.eval()

    with torch.no_grad():
        z_e = E(X)
        if isinstance(z_e, tuple):
            z_e = z_e[0]
        X_rec = G(z_e)

    # if X_rec.shape[-2:] == (3, 2048):
    #     X_rec.transpose_(1, 2)
    test = reconstruction_loss(Xref.permute(0, 2, 1) + 0.5,
                               X_rec.permute(0, 2, 1) + 0.5)

    test = torch.sum(test, dim=1)
    test = test.numpy()
    print(test)
    return test, z_e

    # np.save(join(train_results_path, 'results', f'{epoch:05}_Xrec'), X_rec)


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='File path for evaluation config')
    args = parser.parse_args()

    evaluation_config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            evaluation_config = json.load(f)
    assert evaluation_config is not None

    list_filepath = []
    list_filepath.append("D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048\\scaphoid_models_aligned.h5")
    list_filepath.append("D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048\\scaphoid_models_aligned_amplified_ratio1.025.h5")
    list_filepath.append("D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048\\scaphoid_models_aligned_amplified_ratio1.05.h5")
    list_filepath.append("D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048\\scaphoid_models_aligned_amplified_ratio1.1.h5")
    list_filepath.append("D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048\\scaphoid_models_aligned_amplified_ratio1.15.h5")
    list_filepath.append("D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048\\scaphoid_models_aligned_amplified_ratio1.5.h5")

    list_recloss = []
    for i, filepath in enumerate(list_filepath):
        recloss, codes = get_recloss_latentcode(evaluation_config, filepath)
        list_recloss.append(recloss)
    print(list_recloss)

    import matplotlib.pyplot as plt
    import pandas as pd
    scores = pd.DataFrame(
        [
            list_recloss[0],
            list_recloss[1],
            list_recloss[2],
            list_recloss[3],
            list_recloss[4],
            list_recloss[5],

        ],
        index=['original', '1.025', '1.05', '1.1', '1.15', '1.5'],

    )
    scores = scores.T

    scores.plot()
    plt.show()
