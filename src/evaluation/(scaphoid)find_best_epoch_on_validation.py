import argparse
import json
import logging
import random
import re
from datetime import datetime
from importlib import import_module
from os import listdir
from os.path import join

import numpy as np
import pandas as pd
import torch
import h5py

# from torch.distributions.beta import Beta
from torch.utils.data import DataLoader
####################################
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 当前程序上一级目录，这里为mycompany
sys.path.append(BASE_DIR)
####################################


from datasets.shapenet import ShapeNetDataset
from metrics.jsd import jsd_between_point_cloud_sets
from myutils.util import cuda_setup, setup_logging
from myutils.h5_loader import load_data_h5, MakeBatchData
from myutils.normalize_points import rescale


def _get_epochs_by_regex(path, regex):
    reg = re.compile(regex)
    return {int(w[:6]) for w in listdir(path) if reg.match(w)}


# modify the number of epochs you want to evaluate # not necessary function
def modify_epoch_to_be_evaluated(epoch_list, divisor):
    size_of_list = len(epoch_list)
    new_epoch_list = []
    for epoch in epoch_list:
        if size_of_list % divisor == 0:
            new_epoch_list.append(epoch)
        size_of_list = size_of_list - 1
    return new_epoch_list


def main(eval_config):
    # Load hyperparameters as they were during training
    train_results_path = join(eval_config['results_root'], eval_config['arch'],
                              eval_config['experiment_name'])
    print(train_results_path)
    with open(join(train_results_path, 'config.json')) as f:
        train_config = json.load(f)

    random.seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed_all(train_config['seed'])

    setup_logging(join(train_results_path, 'results'))
    log = logging.getLogger(__name__)

    log.debug('Evaluating JensenShannon divergences on validation set on all '
              'saved epochs.')

    weights_path = join(train_results_path, 'weights')

    # Find all epochs that have saved model weights
    e_epochs = _get_epochs_by_regex(weights_path, r'(?P<epoch>\d{6})_E\.pth')
    g_epochs = _get_epochs_by_regex(weights_path, r'(?P<epoch>\d{6})_G\.pth')
    epochs = sorted(e_epochs.intersection(g_epochs))
    # modify the number of epochs you want to evaluate
    # epochs = modify_epoch_to_be_evaluated(epochs, 10)

    log.debug(f'Testing epochs: {epochs}')

    device = cuda_setup(eval_config['cuda'], eval_config['gpu'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    #
    # Dataset
    filename = "D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048\\scaphoid_models_aligned.h5"

    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())

        # Get the data
        data = list(f[a_group_key[1]])
        normalized_data = []
        for ele in data:
            normalized_points, _ = rescale(ele)
            normalized_data.append(normalized_points)
    #  dimension of X :[N_POINT_CLOUDS, N_POINTS, N_DIM]
    X = normalized_data

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
    arch = import_module(f"models.{train_config['arch']}")
    E = arch.Encoder(train_config).to(device)
    G = arch.Generator(train_config).to(device)

    E.eval()
    G.eval()

    num_samples = len(X)

    # We take 3 times as many samples as there are in test data in order to
    # perform JSD calculation in the same manner as in the reference publication
    noise = torch.FloatTensor(3 * num_samples, train_config['z_size'], 1)
    noise = noise.to(device)

    X = torch.tensor(X).to(device)

    results = {}

    for epoch in reversed(epochs):
        try:
            E.load_state_dict(torch.load(
                join(weights_path, f'{epoch:05}_E.pth')))
            G.load_state_dict(torch.load(
                join(weights_path, f'{epoch:05}_G.pth')))

            start_clock = datetime.now()

            # We average JSD computation from 3 independet trials.
            js_results = []
            for _ in range(3):
                if distribution == 'normal':
                    noise.normal_(0, 0.2)
                elif distribution == 'beta':
                    noise_np = np.random.beta(train_config['z_beta_a'],
                                              train_config['z_beta_b'],
                                              noise.shape)
                    noise = torch.tensor(noise_np).float().round().to(device)

                with torch.no_grad():
                    X_g = G(noise)
                if X_g.shape[-2:] == (3, 2048):
                    X_g.transpose_(1, 2)

                jsd = jsd_between_point_cloud_sets(X, X_g, voxels=28)
                js_results.append(jsd)

            js_result = np.mean(js_results)
            log.debug(f'Epoch: {epoch} JSD: {js_result: .6f} '
                      f'Time: {datetime.now() - start_clock}')
            results[epoch] = js_result
        except KeyboardInterrupt:
            log.debug(f'Interrupted during epoch: {epoch}')
            break

    results = pd.DataFrame.from_dict(results, orient='index', columns=['jsd'])
    log.debug(f"Minimum JSD at epoch {results.idxmin()['jsd']}: "
              f"{results.min()['jsd']: .6f}")


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
