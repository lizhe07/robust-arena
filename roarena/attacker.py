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
from jarvis.utils import (
    get_seed, set_seed, update_default, time_str,
    )

DEVICE = 'cuda'
EVAL_BATCH_SIZE = 128
WORKER_NUM = 0

ATTACKS = {
    'L2': {
        'DF': fb.attacks.L2DeepFoolAttack(),
        'BB': fb.attacks.L2BrendelBethgeAttack(),
        },
    'Linf': {
        'DF': fb.attacks.LinfDeepFoolAttack(),
        'BB': fb.attacks.LinfinityBrendelBethgeAttack(),
        },
    }


class AttackJob(BaseJob):

    def __init__(self, save_dir, **kwargs):
        super(AttackJob, self).__init__(save_dir)
        self.run_config = kwargs

    def get_work_config(self, arg_strs):
        attack_config = get_configs(arg_strs)
        work_config = {
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
            'dists': np.min(dists, axis=0),
            }
        return output, preview


def prepare_batch(dataset, batch_size, batch_idx, is_targeted, device, targets_pth=None):
    images, labels = [], []
    idx_min = batch_size*batch_idx
    idx_max = min(batch_size*(batch_idx+1), len(dataset))
    for idx in range(idx_min, idx_max):
        image, label = dataset[idx]
        images.append(image)
        labels.append(torch.tensor(label, dtype=torch.long))
    images, labels = torch.stack(images), torch.stack(labels)
    images = ep.astensor(images.to(device))
    if is_targeted:
        targets = np.load(targets_pth)[idx_min:idx_max]
        targets = torch.tensor(targets, dtype=torch.long)
        assert not np.any(targets.numpy()==labels.numpy())
        targets = ep.astensor(targets.to(device))
        criterion = fb.criteria.TargetedMisclassification(targets)
    else:
        labels = ep.astensor(labels.to(device))
        criterion = fb.criteria.Misclassification(labels)
    return images, criterion


def get_configs(arg_strs=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_pth')
    parser.add_argument('--metric', default='L2', choices=['L2', 'Linf'])
    parser.add_argument('--name', default='BB', choices=['DF', 'BB'])
    parser.add_argument('--is_targeted', action='store_true')
    parser.add_argument('--batch_size', default=25, type=int)
    parser.add_argument('--batch_idx', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)

    args, _ = parser.parse_known_args(arg_strs)

    assert args.model_pth is not None
    attack_config = {
        'model_pth': args.model_pth,
        'metric': args.metric,
        'name': args.name,
        'is_targeted': args.is_targeted,
        'batch_size': args.batch_size,
        'batch_idx': args.batch_idx,
        'seed': get_seed(args.seed),
        }
    return attack_config


def main(attack_config, **kwargs):
    run_config = update_default({
        'datasets_dir': 'vision_datasets',
        'device': DEVICE,
        'eval_batch_size': EVAL_BATCH_SIZE,
        'worker_num': WORKER_NUM,
        }, kwargs)
    if torch.cuda.is_available() and run_config['device']=='cuda':
        device = 'cuda'
    else:
        device = 'cpu'
    set_seed(attack_config['seed'])

    print('\nattack config:\n{}'.format(attack_config))

    # load model
    saved = torch.load(attack_config['model_pth']) # a dictionary with keys 'task', 'model'
    model = saved['model'] # a model that takes images whose pixel values are in [0, 1]
    model.eval().to(device)

    # prepare testing dataset
    dataset = prepare_datasets(
        saved['task'], run_config['datasets_dir'],
        )

    # initialize attack
    attack = ATTACKS[attack_config['metric']][attack_config['name']]
    if attack_config['name']=='BB':
        fmodel = fb.PyTorchModel(model, bounds=(0, 1))

        init_attack = fb.attacks.DatasetAttack()
        loader = torch.utils.data.DataLoader(dataset, batch_size=EVAL_BATCH_SIZE)
        for _images, _ in loader:
            init_attack.feed(fmodel, ep.astensor(_images.to(device)))
        attack.init_attack = init_attack
        print('dataset attack prepared as initial attack')

    # prepare batch for attack
    if attack_config['is_targeted']:
        targets_pth = '{}/{}_targets.npy'.format(
            run_config['datasets_dir'], saved['task'],
            )
    else:
        targets_pth = None
    images, criterion = prepare_batch(
        dataset, attack_config['batch_size'], attack_config['batch_idx'],
        attack_config['is_targeted'], device, targets_pth,
        )

    # attack model with foolbox
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    tic = time.time()
    _, advs, successes = attack(fmodel, images, criterion, epsilons=None)
    dists = attack.distance(images, advs)
    advs, successes, dists = advs.numpy(), successes.numpy(), dists.numpy()
    toc = time.time()
    print('mean distance: {:.3f} ({})'.format(dists.mean(), time_str(toc-tic)))

    return advs, successes, dists
