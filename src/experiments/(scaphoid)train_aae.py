####################################
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 当前程序上一级目录，这里为mycompany
sys.path.append(BASE_DIR)
####################################

from myutils.pcutil import plot_3d_point_cloud
from myutils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging
from myutils.h5_loader import load_data_h5, MakeBatchData
from myutils.normalize_points import rescale

import h5py
import argparse
import json
import logging
import random
from datetime import datetime
from importlib import import_module
from itertools import chain
from os.path import join, exists

import math
import torch
from torch.autograd import grad
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
cudnn.benchmark = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname in ('Conv1d', 'Linear'):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def main(config):
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    results_dir = prepare_results_dir(config)
    starting_epoch = find_latest_epoch(results_dir) + 1

    if not exists(join(results_dir, 'config.json')):
        with open(join(results_dir, 'config.json'), mode='w') as f:
            json.dump(config, f)

    setup_logging(results_dir)
    log = logging.getLogger(__name__)

    device = cuda_setup(config['cuda'], config['gpu'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    weights_path = join(results_dir, 'weights')

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
            normalized_points,_ = rescale(ele)
            normalized_data.append(normalized_points)

    batchdatalist = MakeBatchData(normalized_data, config['batch_size'])
    #
    # Models
    #
    arch = import_module(f"models.{config['arch']}")
    G = arch.Generator(config).to(device)
    E = arch.Encoder(config).to(device)
    D = arch.Discriminator(config).to(device)

    G.apply(weights_init)
    E.apply(weights_init)
    D.apply(weights_init)

    if config['reconstruction_loss'].lower() == 'chamfer':
        from losses.champfer_loss import ChamferLoss
        reconstruction_loss = ChamferLoss().to(device)
    elif config['reconstruction_loss'].lower() == 'earth_mover':
        from losses.earth_mover_distance import EMD
        reconstruction_loss = EMD().to(device)
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')
    #
    # Float Tensors
    #
    fixed_noise = torch.FloatTensor(config['batch_size'], config['z_size'], 1)
    fixed_noise.normal_(mean=config['normal_mu'], std=config['normal_std'])
    noise = torch.FloatTensor(config['batch_size'], config['z_size'])

    fixed_noise = fixed_noise.to(device)
    noise = noise.to(device)

    #
    # Optimizers
    #
    EG_optim = getattr(optim, config['optimizer']['EG']['type'])
    EG_optim = EG_optim(chain(E.parameters(), G.parameters()),
                        **config['optimizer']['EG']['hyperparams'])

    D_optim = getattr(optim, config['optimizer']['D']['type'])
    D_optim = D_optim(D.parameters(),
                      **config['optimizer']['D']['hyperparams'])

    if starting_epoch > 1:
        G.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_G.pth')))
        E.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_E.pth')))
        D.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_D.pth')))

        D_optim.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_Do.pth')))

        EG_optim.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_EGo.pth')))

    for epoch in range(starting_epoch, config['max_epochs'] + 1):
        start_epoch_time = datetime.now()

        G.train()
        E.train()
        D.train()

        total_loss_d = 0.0
        total_loss_eg = 0.0
        total_loss_e = 0.0
        total_loss_g = 0.0
        total_loss_gp = 0.0

        for i, point_data in enumerate(batchdatalist):

            # log.debug('-' * 20)
            X = point_data.to(device)


            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if X.size(-1) == 3:
                X.transpose_(X.dim() - 2, X.dim() - 1)

            codes, _, _ = E(X)
            noise.normal_(mean=config['normal_mu'], std=config['normal_std'])
            synth_logit = D(codes)
            real_logit = D(noise)
            loss_d = torch.mean(synth_logit) - torch.mean(real_logit)

            alpha = torch.rand(config['batch_size'], 1).to(device)
            differences = codes - noise
            interpolates = noise + alpha * differences
            disc_interpolates = D(interpolates)

            gradients = grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(disc_interpolates).to(device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1))
            gradient_penalty = ((slopes - 1) ** 2).mean()

            # lamdda exponential operation
            # if epoch < 20:
            #     gp_lambda = config['gp_lambda']
            # else:
            #     gp_lambda = config['gp_lambda']*math.exp((20-epoch)/544)
            #     if gp_lambda < 2:
            #         gp_lambda = 2
            # lamdda constant operation
            gp_lambda = config['gp_lambda']

            loss_gp = gp_lambda * gradient_penalty
            ###
            loss_d += loss_gp

            D_optim.zero_grad()
            D.zero_grad()

            loss_d.backward(retain_graph=True)
            total_loss_d += loss_d.item()
            D_optim.step()

            # EG part of training
            X_rec = G(codes)

            loss_e = torch.mean(
                config['reconstruction_coef'] *
                reconstruction_loss(X.permute(0, 2, 1) + 0.5,
                                    X_rec.permute(0, 2, 1) + 0.5))

            synth_logit = D(codes)

            loss_g = -torch.mean(synth_logit)

            loss_eg = loss_e + loss_g
            EG_optim.zero_grad()
            E.zero_grad()
            G.zero_grad()

            loss_eg.backward()
            total_loss_eg += loss_eg.item()
            EG_optim.step()
            total_loss_e += loss_e
            total_loss_g += loss_g
            total_loss_gp += loss_gp

            # log for each mini-batch
            # log.debug(f'[{epoch}: ({i})] '
            #           f'Loss_D: {loss_d.item():.4f} '
            #           f'(GP: {loss_gp.item(): .4f}) '
            #           f'Loss_EG: {loss_eg.item():.4f} '
            #           f'(REC: {loss_e.item(): .4f}) '
            #           f'Time: {datetime.now() - start_epoch_time}')

        log.debug(
            f'[{epoch}/{config["max_epochs"]}] '
            f'Loss_D: {total_loss_d / (i+1):.4f} '
            f'Loss_EG: {total_loss_eg / (i+1):.4f} '
            f'Loss_E: {total_loss_e / (i+1):.4f} '
            f'Loss_G: {total_loss_g / (i+1):.4f} '
            f'Loss_gp: {total_loss_gp / (i+1):.4f} '            
            f'Time: {datetime.now() - start_epoch_time}  '
            f'lambda: {gp_lambda:4f}'
        )


        # Save intermediate results
        #
        G.eval()
        E.eval()
        D.eval()
        with torch.no_grad():
            fake = G(fixed_noise).data.cpu().numpy()
            codes, _, _ = E(X)
            X_rec = G(codes).data.cpu().numpy()

        # intermediate results figure (need a lot of hard disk space)

        for k in range(5):
            fig = plot_3d_point_cloud(X[k][0], X[k][1], X[k][2],
                                      in_u_sphere=True, show=False,
                                      title=str(epoch))
            fig.savefig(
                join(results_dir, 'samples', f'{epoch:05}_{k}_real.png'))
            plt.close(fig)

        for k in range(5):
            fig = plot_3d_point_cloud(fake[k][0], fake[k][1], fake[k][2],
                                      in_u_sphere=True, show=False,
                                      title=str(epoch))
            fig.savefig(
                join(results_dir, 'samples', f'{epoch:05}_{k}_fixed.png'))
            plt.close(fig)

        for k in range(5):
            fig = plot_3d_point_cloud(X_rec[k][0],
                                      X_rec[k][1],
                                      X_rec[k][2],
                                      in_u_sphere=True, show=False,
                                      title=str(epoch))
            fig.savefig(join(results_dir, 'samples',
                             f'{epoch:05}_{k}_reconstructed.png'))
            plt.close(fig)

        if epoch % config['save_frequency'] == 0:
            torch.save(G.state_dict(), join(weights_path, f'{epoch:05}_G.pth'))
            torch.save(D.state_dict(), join(weights_path, f'{epoch:05}_D.pth'))
            torch.save(E.state_dict(), join(weights_path, f'{epoch:05}_E.pth'))

            torch.save(EG_optim.state_dict(),
                       join(weights_path, f'{epoch:05}_EGo.pth'))

            torch.save(D_optim.state_dict(),
                       join(weights_path, f'{epoch:05}_Do.pth'))


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path')
    args = parser.parse_args()

    config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    main(config)