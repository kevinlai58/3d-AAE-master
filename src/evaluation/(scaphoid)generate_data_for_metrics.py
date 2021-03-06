import argparse
import json
import logging
import random
from importlib import import_module
from os.path import join

import numpy as np
import torch
import h5py

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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

    filename = "D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048\\scaphoid_models.h5"

    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())

        # Get the data
        data = list(f[a_group_key[1]])
        X = []
        for ele in data:
            normalized_points, _ = rescale(ele)
            X.append(normalized_points)
    #  dimension of X :[N_POINT_CLOUDS, N_POINTS, N_DIM]

    X = torch.tensor(X).to(device)

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

    num_samples = len(X)

    # We take 3 times as many samples as there are in test data in order to
    # perform JSD calculation in the same manner as in the reference publication
    noise = torch.FloatTensor(3 * num_samples, train_config['z_size'], 1)
    noise = noise.to(device)

    # generate figure of original point clouds and the reconstructed point clouds
    np.save(join(train_results_path, 'results', f'{epoch:05}_X'), X)

    for i in range(3):
        if distribution == 'normal':
            noise.normal_(0, 0.2)
        else:
            noise_np = np.random.beta(train_config['z_beta_a'],
                                      train_config['z_beta_b'],
                                      noise.shape)
            noise = torch.tensor(noise_np).float().round().to(device)
        with torch.no_grad():
            X_g = G(noise)
        if X_g.shape[-2:] == (3, 2048):
            X_g.transpose_(1, 2)

        np.save(join(train_results_path, 'results', f'{epoch:05}_Xg_{i}'), X_g)

    with torch.no_grad():
        z_e = E(X.transpose(1, 2))
        if isinstance(z_e, tuple):
            z_e = z_e[0]
        X_rec = G(z_e)
    if X_rec.shape[-2:] == (3, 2048):
        X_rec.transpose_(1, 2)

    np.save(join(train_results_path, 'results', f'{epoch:05}_Xrec'), X_rec)

    # latent space plt(use TSNE OR PCA)

    numpy_z_e = z_e.numpy()

    # reduced_z = PCA(n_components=2).fit_transform(ndz_e)
    reduced_z = TSNE(n_components=2).fit_transform(numpy_z_e)

    x_list = []
    y_list = []
    for x, y in reduced_z:
        x_list.append(x)
        y_list .append(y)
    plt.scatter(x_list, y_list,s=10,alpha=1)
    plt.title(join("latent space with epoch " +str(epoch)+" (TSNE)"))
    plt.figure(figsize=(16, 16))
    fig = plt.gcf()
    plt.show()
    plt.draw()
    path = join(train_results_path, 'results', f'{epoch:05}_TSNE.png')
    fig.savefig(path, dpi=100)

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

    for i in range(20):
        evaluation_config['epoch'] = (i+1)*10000
        main(evaluation_config)
