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

def main(eval_config):
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

    #
    # Dataset
    #
    # dataset_name = train_config['dataset'].lower()
    # if dataset_name == 'shapenet':
    #     dataset = ShapeNetDataset(root_dir=train_config['data_dir'],
    #                               classes=train_config['classes'], split='test')
    # elif dataset_name == 'faust':
    #     from datasets.dfaust import DFaustDataset
    #     dataset = DFaustDataset(root_dir=train_config['data_dir'],
    #                             classes=train_config['classes'], split='test')
    # elif dataset_name == 'mcgill':
    #     from datasets.mcgill import McGillDataset
    #     dataset = McGillDataset(root_dir=train_config['data_dir'],
    #                             classes=train_config['classes'], split='test')
    # else:
    #     raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
    #                      f'`faust`. Got: `{dataset_name}`')
    # classes_selected = ('all' if not train_config['classes']
    #                     else ','.join(train_config['classes']))
    # log.debug(f'Selected {classes_selected} classes. Loaded {len(dataset)} '
    #           f'samples.')

    filename = "D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048\\scaphoid_models_aligned.h5"
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

    X = torch.tensor(normalized_points_list).float().to(device)


    #
    # path = "D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048/"
    # X, partial = load_data_h5(path, "valid")
    #
    # X = X[0:200]
    #
    # X = torch.tensor(X).to(device)

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

    #
    # Load saved state
    #
    E.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_E.pth')))
    G.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_G.pth')))

    E.eval()
    G.eval()

    X= X[102:104,:]
    with torch.no_grad():
        z_e = E(X.transpose(1, 2))
        if isinstance(z_e, tuple):
            z_e = z_e[0]
            z_e_0 = torch.reshape(z_e[0], (1, 2048))
            z_e_1 = torch.reshape(z_e[1], (1, 2048))

        index = 3
        z_intertensor = z_e_0
        for i in range(index):
            z_mid = z_e[0]*(index-i)/(index+1) + z_e[1]*(i+1)/(index+1)
            z_mid = torch.reshape(z_mid, (1, 2048))
            z_intertensor = torch.cat((z_intertensor, z_mid), dim=0)
        z_intertensor = torch.cat((z_intertensor, z_e_1), dim=0)

        X_inter = G(z_intertensor)
    if X_inter.shape[-2:] == (3, 2048):
        X_inter.transpose_(1, 2)

    np.save(join(train_results_path, 'results', f'{epoch:05}_Xg_interpolating'), X_inter)


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

    main(evaluation_config)
