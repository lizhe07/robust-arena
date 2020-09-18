# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:57:18 2020

@author: Zhe
"""

import argparse, torch, time
import numpy as np
import eagerpy as ep
import foolbox as fb

from jarvis import BaseJob
from jarvis.vision import prepare_datasets
from jarvis.utils import update_default, time_str

DEVICE = 'cuda'
EVAL_BATCH_SIZE = 160
WORKER_NUM = 0

ATTACKS = {
    'L2': {
        'PGD': fb.attacks.L2ProjectedGradientDescentAttack(),
        'BI': fb.attacks.L2BasicIterativeAttack(),
        'DF': fb.attacks.L2DeepFoolAttack(),
        'BB': fb.attacks.L2BrendelBethgeAttack(),
        },
    'Linf': {
        'PGD': fb.attacks.LinfProjectedGradientDescentAttack(),
        'BI': fb.attacks.LinfBasicIterativeAttack(),
        'DF': fb.attacks.LinfDeepFoolAttack(),
        'BB': fb.attacks.LinfinityBrendelBethgeAttack(),
        },
    }


class AdvAttackJob(BaseJob):

    def __init__(self, save_dir, datasets_dir, **kwargs):
        super(AdvAttackJob, self).__init__(save_dir)
        self.run_config = dict(
            datasets_dir=datasets_dir, **kwargs
            )

    def get_work_config(self, arg_strs):
        model_pth, attack_config = get_configs(arg_strs)
        work_config = {
            'model_pth': model_pth,
            'attack_config': attack_config,
            }
        return work_config

    def main(self, work_config):
        advs, successes, dists = main(**work_config, **self.run_config)
        output = {
            'advs': advs,
            'successes': successes,
            'dists': dists,
            }
        preview = {
            'success_rate': np.mean(successes),
            'dist_mean': np.mean(dists),
            'dist_01': np.quantile(dists, 0.01),
            'dist_25': np.quantile(dists, 0.25),
            'dist_50': np.quantile(dists, 0.50),
            'dist_75': np.quantile(dists, 0.75),
            'dist_99': np.quantile(dists, 0.99),
            }
        return output, preview


def attack_model(model, dataset, attack, eps, device=DEVICE,
                 batch_size=EVAL_BATCH_SIZE, worker_num=WORKER_NUM):
    if device=='cuda' and not torch.cuda.is_available():
        device = 'cpu'
    model.eval().to(device)
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=worker_num)
    advs, successes = [], []
    for images, labels in loader:
        _images, _labels = ep.astensors(images.to(device), labels.to(device))
        _, _advs, _successes = attack(fmodel, _images, _labels, epsilons=eps)
        advs.append(_advs.raw.cpu())
        successes.append(_successes.raw.cpu())
    advs = torch.cat(advs)
    successes = torch.cat(successes)
    return advs, successes


def dataset_forwardpass(model, dataset, device=DEVICE, batch_size=EVAL_BATCH_SIZE):
    if device=='cuda' and not torch.cuda.is_available():
        device = 'cpu'
    model.eval().to(device)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    images, labels, predicts = [], [], []
    for _images, _labels in loader:
        with torch.no_grad():
            _, _predicts = model(_images.to(device)).max(dim=1)
        images.append(_images)
        labels.append(_labels)
        predicts.append(_predicts.cpu())
    images = torch.cat(images)
    labels = torch.cat(labels)
    predicts = torch.cat(predicts)
    return images, labels, predicts


def get_configs(arg_strs=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_pth')
    parser.add_argument('--metric', default='Linf', choices=['L2', 'Linf'])
    parser.add_argument('--name', default='PGD', choices=['PGD', 'BI', 'DF', 'BB'])
    parser.add_argument('--success_threshold', default=0.999, type=float)
    parser.add_argument('--eps_step', type=float)

    args = parser.parse_args(arg_strs)

    if args.eps_step is None:
        if args.metric=='L2':
            args.eps_step = 0.01
        if args.metric=='Linf':
            args.eps_step = 0.001

    model_pth = args.model_pth
    attack_config = {
        'metric': args.metric,
        'name': args.name,
        'success_threshold': args.success_threshold,
        'eps_step': args.eps_step,
        }
    return model_pth, attack_config


def main(model_pth, attack_config, **kwargs):
    print('model path:\n{}'.format(model_pth))
    print('attack config:\n{}'.format(attack_config))
    run_config = update_default({
        'datasets_dir': 'vision_datasets',
        'device': DEVICE,
        'eval_batch_size': EVAL_BATCH_SIZE,
        'worker_num': WORKER_NUM,
        }, kwargs)

    # load model
    saved = torch.load(model_pth)
    model = saved['model']

    # prepare datasets and initialize attacks as original images
    dataset = prepare_datasets(
        saved['config']['model_config']['task'],
        run_config['datasets_dir'],
        )
    images, labels, predicts = dataset_forwardpass(model, dataset)
    advs, successes = images.clone(), labels!=predicts

    # attack model with increasing eps
    attack = ATTACKS[attack_config['metric']][attack_config['name']]
    if attack_config['name']=='BB':
        if torch.cuda.is_available() and run_config['device']=='cuda':
            device = 'cuda'
        else:
            device = 'cpu'
        model.eval().to(device)
        fmodel = fb.PyTorchModel(model, bounds=(0, 1))

        init_attack = fb.attacks.DatasetAttack()
        loader = torch.utils.data.DataLoader(dataset, batch_size=run_config['eval_batch_size'])
        for _images, _ in loader:
            init_attack.feed(fmodel, ep.astensor(_images.to(device)))
        attack.init_attack = init_attack
        print('dataset attack prepared as initial attack')

    eps = 0
    last_succ_rate = 0.
    while successes.to(torch.float).mean()<attack_config['success_threshold']:
        eps += attack_config['eps_step']
        print('attacking with eps {:.3f}...'.format(eps))
        tic = time.time()
        idxs, = (successes!=True).numpy().nonzero()
        _dataset = torch.utils.data.TensorDataset(images[idxs], labels[idxs])
        _advs, _successes = attack_model(model, _dataset, attack, eps,
                                         run_config['device'],
                                         run_config['eval_batch_size'],
                                         run_config['worker_num'])
        advs[idxs], successes[idxs] = _advs, _successes
        toc = time.time()
        curr_succ_rate = successes.to(torch.float).mean()
        if curr_succ_rate-last_succ_rate>0.01:
            print('success rate {:7.2%} ({})'.format(
                curr_succ_rate, time_str(toc-tic),
                ))
            last_succ_rate = curr_succ_rate
    dists = attack.distance(images, advs).numpy()
    advs = advs.numpy()
    successes = successes.numpy()
    return advs, successes, dists
