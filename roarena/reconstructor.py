# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 23:14:41 2020

@author: Zhe
"""

import argparse, time, torch
import numpy as np

from jarvis import BaseJob
from jarvis.vision import prepare_datasets
from jarvis.utils import update_default, progress_str, time_str

DEVICE = 'cuda'
BATCH_SIZE = 64
DISP_NUM = 32


class ReconsJob(BaseJob):

    def __init__(self, save_dir, datasets_dir, **kwargs):
        super(ReconsJob, self).__init__(save_dir)
        self.run_config = dict(
            datasets_dir=datasets_dir, **kwargs
            )

    def get_work_config(self, arg_strs):
        model_pth, recons_config = get_configs(arg_strs)
        work_config = {
            'model_pth': model_pth,
            'recons_config': recons_config,
            }
        return work_config

    def main(self, work_config):
        imgs_orig, imgs_reco, acts_orig, acts_reco = main(
            **work_config, **self.run_config
            )
        output = {
            'imgs_orig': imgs_orig,
            'imgs_reco': imgs_reco,
            'acts_orig': acts_orig,
            'acts_reco': acts_reco,
            }
        preview = {
            'img_mse': np.power(imgs_orig-imgs_reco, 2).mean(),
            'act_corr': np.corrcoef(acts_orig.flatten(), acts_reco.flatten())[0, 1],
            }
        return output, preview


def reconstruct(model, dataset, l_idx, noise=0., sigma=0.1, lr=0.01, step_num=1000,
                alpha=0.01, device=DEVICE, batch_size=BATCH_SIZE, disp_num=DISP_NUM):
    if device=='cuda' and not torch.cuda.is_available():
        device = 'cpu'
    model.eval().to(device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    batch_num = len(loader)

    imgs_orig, imgs_reco = [], []
    acts_orig, acts_reco = [], []
    tic = time.time()
    for b_idx, (imgs, _) in enumerate(loader, 1):
        with torch.no_grad():
            activations = model.layer_activations(imgs.to(device))
        activations[l_idx] *= torch.exp(torch.randn_like(activations[l_idx])*noise)
        acts_orig.append(activations[l_idx].flatten(1).cpu())

        _imgs = torch.randn_like(imgs)*sigma+0.5
        _imgs.requires_grad = True
        optimizer = torch.optim.Adam([_imgs], lr=lr)

        for _ in range(step_num):
            _activations = model.layer_activations(_imgs.to(device))
            loss_rec = (_activations[l_idx]-activations[l_idx]).pow(2).mean()
            loss_reg = (_imgs-0.5).pow(2).mean()
            loss = (1-alpha)*loss_rec+alpha*loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            _activations = model.layer_activations(_imgs.to(device))
        acts_reco.append(_activations[l_idx].flatten(1).cpu())
        imgs_orig.append(imgs.data.cpu())
        imgs_reco.append(_imgs.data.cpu())

        if b_idx%(-(-batch_num//disp_num))==0 or b_idx==batch_num:
            toc = time.time()
            print('{} ({})'.format(
                progress_str(b_idx, batch_num),
                time_str(toc-tic, b_idx/batch_num),
                ))
    imgs_orig = torch.cat(imgs_orig).numpy()
    imgs_reco = torch.cat(imgs_reco).numpy()
    acts_orig = torch.cat(acts_orig).numpy()
    acts_reco = torch.cat(acts_reco).numpy()
    return imgs_orig, imgs_reco, acts_orig, acts_reco


def get_configs(arg_strs=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_pth')
    parser.add_argument('--l_idx', default=0, type=int)
    parser.add_argument('--noise', default=0., type=float)
    parser.add_argument('--sigma', default=0.1, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--step_num', default=1000, type=int)
    parser.add_argument('--alpha', default=0.01, type=float)

    args = parser.parse_args(arg_strs)

    model_pth = args.model_pth
    recons_config = {
        'l_idx': args.l_idx,
        'noise': args.noise,
        'sigma': args.sigma,
        'lr': args.lr,
        'step_num': args.step_num,
        'alpha': args.alpha,
        }
    return model_pth, recons_config


def main(model_pth, recons_config, **kwargs):
    print('model path:\n{}'.format(model_pth))
    print('reconstruct config:\n{}'.format(recons_config))
    run_config = update_default({
        'datasets_dir': 'vision_datasets',
        'device': DEVICE,
        'batch_size': BATCH_SIZE,
        'disp_num': DISP_NUM,
        }, kwargs)

    # load model
    saved = torch.load(model_pth)
    model = saved['model']

    # prepare datasets and initialize attacks as original images
    dataset = prepare_datasets(
        saved['config']['model_config']['task'],
        run_config['datasets_dir'],
        )

    # reconstruct from one layer
    imgs_orig, imgs_reco, acts_orig, acts_reco = reconstruct(
        model, dataset, recons_config['l_idx'], recons_config['noise'], recons_config['sigma'],
        recons_config['lr'], recons_config['step_num'], recons_config['alpha'],
        run_config['device'], run_config['batch_size'], run_config['disp_num'],
        )
    return imgs_orig, imgs_reco, acts_orig, acts_reco
