# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:57:18 2020

@author: Zhe
"""

import os, argparse, pickle, time, torch
import numpy as np
import eagerpy as ep
import foolbox as fb

from jarvis import BaseJob
from jarvis.vision import prepare_datasets
from jarvis.utils import job_parser, get_seed, set_seed, time_str

from . import DEVICE, WORKER_NUM
BATCH_SIZE = 20

METRICS = ['L2', 'Linf']
NAMES = ['PGD', 'BI', 'DF', 'BB']
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
EPS_RESOL = {'L2': 0.01, 'Linf': 1/255}


class AttackJob(BaseJob):

    def __init__(self, store_dir, datasets_dir, device=DEVICE, worker_num=WORKER_NUM):
        if store_dir is None:
            super(AttackJob, self).__init__()
        else:
            super(AttackJob, self).__init__(os.path.join(store_dir, 'adversarial_attacks'))
        self.datasets_dir = datasets_dir
        self.device = 'cuda' if device=='cuda' and torch.cuda.is_available() else 'cpu'
        self.worker_num = worker_num

    def get_config(self, arg_strs):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model_pth')
        parser.add_argument('--seed', default=0, type=int)
        parser.add_argument('--metric', default='L2', choices=METRICS)
        parser.add_argument('--name', default='BB', choices=NAMES)
        parser.add_argument('--targeted', action='store_true')
        parser.add_argument('--eps', type=float)
        parser.add_argument('--batch_idx', default=0, type=int)

        args, _ = parser.parse_known_args(arg_strs)

        assert args.model_pth is not None
        if args.eps is None:
            assert args.name in ['DF', 'BB'], 'eps=None is only implemented for DF and BB attack now'
            eps_level = None
        else:
            eps_level = int(args.eps/EPS_RESOL[args.metric])
            assert eps_level>0
        config = {
            'model_pth': args.model_pth,
            'seed': get_seed(args.seed),
            'metric': args.metric,
            'name': args.name,
            'targeted': args.targeted,
            'eps_level': eps_level,
            'batch_idx': args.batch_idx,
            }
        return config

    def prepare_batch(self, dataset, batch_idx, targeted, targets_pth=None):
        images, labels = [], []
        idx_min = BATCH_SIZE*batch_idx
        idx_max = min(BATCH_SIZE*(batch_idx+1), len(dataset))
        for idx in range(idx_min, idx_max):
            image, label = dataset[idx]
            images.append(image)
            labels.append(torch.tensor(label, dtype=torch.long))
        images, labels = torch.stack(images), torch.stack(labels)
        images = ep.astensor(images.to(self.device))
        if targeted:
            targets = np.load(targets_pth)[idx_min:idx_max]
            targets = torch.tensor(targets, dtype=torch.long)
            assert not np.any(targets.numpy()==labels.numpy())
            targets = ep.astensor(targets.to(self.device))
            criterion = fb.criteria.TargetedMisclassification(targets)
        else:
            labels = ep.astensor(labels.to(self.device))
            criterion = fb.criteria.Misclassification(labels)
        return images, criterion

    def main(self, config, verbose=True):
        if verbose:
            print(config)
        set_seed(config['seed'])

        # load model
        saved = torch.load(config['model_pth'])
        model = saved['model']
        model.eval().to(self.device)
        fmodel = fb.PyTorchModel(model, bounds=(0, 1))

        # prepare testing dataset
        dataset = prepare_datasets(
            saved['task'], self.datasets_dir, grayscale=saved['grayscale'],
            )

        # initialize attack
        attack = ATTACKS[config['metric']][config['name']]
        if config['name']=='BB':
            init_attack = fb.attacks.DatasetAttack()
            loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
            for _images, _ in loader:
                init_attack.feed(fmodel, ep.astensor(_images.to(self.device)))
            attack.init_attack = init_attack
            if verbose:
                print('dataset attack prepared as initial attack')

        # prepare batch for attack
        if config['targeted']:
            targets_pth = '{}/{}_targets.npy'.format(self.datasets_dir, saved['task'])
        else:
            targets_pth = None
        images, criterion = self.prepare_batch(
            dataset, config['batch_idx'], config['targeted'], targets_pth,
            )

        # attack model with foolbox
        if verbose:
            tic = time.time()
        if config['eps_level'] is None:
            eps = None
        else:
            eps = config['eps_level']*EPS_RESOL[config['metric']]
        _, advs, successes = attack(fmodel, images, criterion, epsilons=eps)
        dists = attack.distance(images, advs)
        advs, successes, dists = advs.numpy(), successes.numpy(), dists.numpy()
        if verbose:
            toc = time.time()
            if eps is None:
                print('mean distance: {:.3f} ({})'.format(dists.mean(), time_str(toc-tic)))
            else:
                print('success rate: {:7.2%} ({})'.format(successes.mean(), time_str(toc-tic)))

        result = {
            'advs': advs,
            'successes': successes,
            'dists': dists,
            }
        preview = {
            'successes': successes,
            'dists': dists,
            }
        return result, preview


if __name__=='__main__':
    parser = job_parser()
    parser.add_argument('--store_dir', default='store')
    parser.add_argument('--datasets_dir', default='vision_datasets')
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--worker_num', default=WORKER_NUM, type=int)
    parser.add_argument('--max_seed', default=4, type=int)
    parser.add_argument('--metric', default='L2', choices=METRICS)
    parser.add_argument('--batch_num', default=50, type=int)
    args = parser.parse_args()

    if args.spec_pth is None:
        export_dir = os.path.join(args.store_dir, 'models', 'exported')
        assert os.path.exists(export_dir), "directory of exported models not found"
        search_spec = {
            'model_pth': [os.path.join(export_dir, f) for f in os.listdir(export_dir)],
            'seed': list(range(args.max_seed)),
            'metric': [args.metric],
            'name': NAMES,
            'targeted': [False, True],
            'eps': [0.25, 0.5, 1., 2.] if args.metric=='L2' else [8/255, 16/255],
            'batch_idx': list(range(args.batch_num)),
            }
    else:
        with open(args.spec_pth, 'rb') as f:
            search_spec = pickle.load(f)

    job = AttackJob(
        args.store_dir, args.datasets_dir, args.device, args.worker_num
        )
    job.random_search(search_spec, args.process_num, args.max_wait, args.tolerance)
