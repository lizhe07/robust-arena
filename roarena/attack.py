# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:57:18 2020

@author: Zhe
"""

import os, argparse, pickle, random, time, torch
import numpy as np
import foolbox as fb

from jarvis import BaseJob
from jarvis.vision import prepare_datasets
from jarvis.utils import job_parser, get_seed, set_seed, time_str

from . import DEVICE, WORKER_NUM

EPS_LEVELS = list(range(1, 101))
METRICS = ['L2', 'LI']
NAMES = ['PGD', 'BI', 'DF', 'BB']
ATTACKS = {
    'L2': {
        'PGD': fb.attacks.L2ProjectedGradientDescentAttack(),
        'BI': fb.attacks.L2BasicIterativeAttack(),
        'DF': fb.attacks.L2DeepFoolAttack(),
        'BB': fb.attacks.L2BrendelBethgeAttack(),
        },
    'LI': {
        'PGD': fb.attacks.LinfProjectedGradientDescentAttack(),
        'BI': fb.attacks.LinfBasicIterativeAttack(),
        'DF': fb.attacks.LinfDeepFoolAttack(),
        'BB': fb.attacks.LinfinityBrendelBethgeAttack(),
        },
    }
IMG_SIZES = {
    'MNIST': 1*28*28,
    'CIFAR10': 3*32*32,
    'CIFAR100': 3*32*32,
    'ImageNet': 3*224*224,
    }


class AttackJob(BaseJob):
    r"""Performs adversarial attacks.

    Args
    ----
    store_dir: str
        The directory for storing results. When `store_dir` is ``None``, no
        external storage is used.
    datasets_dir: str
        The directory for vision datasets.
    device: str
        The device for computation.
    worker_num: int
        The worker number for data loader.

    """
    BATCH_SIZE = 20

    def __init__(self, store_dir, datasets_dir, device=DEVICE, worker_num=WORKER_NUM, **kwargs):
        if store_dir is None:
            super(AttackJob, self).__init__(**kwargs)
        else:
            super(AttackJob, self).__init__(os.path.join(store_dir, 'attacks'), **kwargs)
        self.datasets_dir = datasets_dir
        self.device = 'cuda' if device=='cuda' and torch.cuda.is_available() else 'cpu'
        self.worker_num = worker_num

    def get_config(self, arg_strs):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model_pth', help="path to the model")
        parser.add_argument('--metric', default='LI', choices=METRICS,
                            help="perturbation metric")
        parser.add_argument('--eps_level', type=int, choices=EPS_LEVELS,
                            help="integer for epsilon [0, 100)")
        parser.add_argument('--targeted', action='store_true',
                            help="whether the attack is targeted")
        parser.add_argument('--shuffle_mode', default='elm', choices=['elm', 'cls'],
                            help="shuffle mode of targeted attack labels")
        parser.add_argument('--shuffle_tag', default=0, type=int,
                            help="shuffule tag of targeted attack labels")
        parser.add_argument('--overshoot', default=0.01, type=float,
                            help="overshoot parameter for logits")
        parser.add_argument('--name', default='BB', choices=NAMES,
                            help="name of attack method")
        parser.add_argument('--sample_idx', default=0, type=int,
                            help="index of sample to attack")
        parser.add_argument('--seed', default=0, type=int,
                            help="random seed")

        args, _ = parser.parse_known_args(arg_strs)

        assert args.model_pth is not None
        if args.name in ['BI', 'DF']:
            args.seed = 0
        if args.name in ['PGD', 'BI'] and args.eps_level is None:
            args.eps_level = 1
        if args.name in ['DF', 'BB']:
            args.eps_level = None
        config = {
            'model_pth': args.model_pth,
            'metric': args.metric,
            'eps_level': args.eps_level,
            'targeted': args.targeted,
            'shuffle_mode': args.shuffle_mode if args.targeted else None,
            'shuffle_tag': args.shuffle_tag if args.targeted else None,
            'overshoot': args.overshoot,
            'name': args.name,
            'sample_idx': args.sample_idx,
            'seed': get_seed(args.seed),
            }
        return config

    def shuffled_targets(self, targets, shuffle_mode, shuffle_tag):
        set_seed(shuffle_tag)
        labels_all = np.array(targets)
        if shuffle_mode=='elm':
            last = labels_all.size
            targets = np.random.permutation(labels_all)
            while True:
                idxs, = np.nonzero(targets==labels_all)
                if idxs.size>0:
                    if idxs.size<last:
                        last = idxs.size
                        targets[np.random.permutation(idxs)] = targets[idxs]
                    else:
                        last = labels_all.size
                        targets = np.random.permutation(labels_all)
                else:
                    break
        if shuffle_mode=='cls':
            _labels = np.unique(labels_all)
            while True:
                _targets = np.random.permutation(_labels)
                if np.all(_targets!=_labels):
                    break
            targets = labels_all.copy()
            for _t, _l in zip(_targets, _labels):
                targets[labels_all==_l] = _t
        assert not np.any(targets==labels_all)
        return targets

    def dataset_attack(self, model, dataset, criterion):
        success = False
        for s_idx in random.sample(range(len(dataset)), len(dataset)):
            image, _ = dataset[s_idx]
            with torch.no_grad():
                logit = model(image[None].to(self.device))[0]
            if criterion(image[None], logit[None])[0]:
                success = True
                break
        assert success
        return image[None]

    def main(self, config, verbose=True):
        if verbose:
            print(config)

        # load model
        saved = torch.load(config['model_pth'])
        task, model = saved['task'], saved['model']
        model.eval().to(self.device)
        fmodel = fb.PyTorchModel(model, bounds=(0, 1))

        # prepare testing dataset
        dataset = prepare_datasets(task, self.datasets_dir)

        # prepare single-sample batch for attack
        sample_idx = config['sample_idx']
        image, label = dataset[sample_idx]
        images = image[None].to(self.device)
        labels = torch.tensor([label], dtype=torch.long, device=self.device)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(model(images), dim=1)
            prob, pred = probs.max(dim=1)
        prob_raw, pred_raw = prob.item(), pred.item()

        # prepare criterion
        if config['targeted']:
            targets = self.shuffled_targets(
                dataset.targets, config['shuffle_mode'], config['shuffle_tag'],
                )
            targets = torch.tensor([targets[sample_idx]], dtype=torch.long, device=self.device)
            criterion = fb.criteria.TargetedMisclassification(targets, config['overshoot'])
        else:
            criterion = fb.criteria.Misclassification(labels, config['overshoot'])

        # initialize attack
        set_seed(config['seed'])
        attack = ATTACKS[config['metric']][config['name']]
        run_kwargs = {}
        if config['name']=='BB':
            starting_points = self.dataset_attack(
                model, dataset, criterion,
                ).to(self.device)
            run_kwargs = {'starting_points': starting_points}

        # attack model with foolbox
        if verbose:
            tic = time.time()
        if config['eps_level'] is None:
            eps = None
        else:
            if config['metric']=='L2':
                eps_max = IMG_SIZES[task]**0.5/10
            if config['metric']=='LI':
                eps_max = 0.5
            eps = config['eps_level']/100*eps_max
        _, advs, successes = attack(fmodel, images, criterion, epsilons=eps, **run_kwargs)
        dist = attack.distance(images, advs).item()
        success = successes.item()

        with torch.no_grad():
            probs = torch.nn.functional.softmax(model(advs), dim=1)
            prob, pred = probs.max(dim=1)
        prob_adv, pred_adv = prob.item(), pred.item()
        adv = advs[0].cpu().numpy()

        if verbose:
            toc = time.time()
            print('eps: {:.4f}, attack {} ({})'.format(
                dist, 'successful' if success else 'failed',
                time_str(toc-tic),
                ))

        result = {
            'label': label,
            'prob_raw': prob_raw, 'pred_raw': pred_raw,
            'adv': adv,
            'dist': dist, 'success': success,
            'prob_adv': prob_adv, 'pred_adv': pred_adv,
            }
        preview = {
            'label': label,
            'prob_raw': prob_raw, 'pred_raw': pred_raw,
            'dist': dist, 'success': success,
            'prob_adv': prob_adv, 'pred_adv': pred_adv,
            }
        return result, preview

    def best_attack(self, model_pth, sample_idx, metric, targeted, *,
                    min_prob=None, max_dist=None, shuffle_mode='elm', shuffle_tag=0):
        if min_prob is not None:
            assert max_dist is None
            min_dist = None
        if max_dist is not None:
            assert min_prob is None
            max_prob = None
        cond = {
            'model_pth': model_pth,
            'sample_idx': sample_idx,
            'metric': metric,
            'targeted': targeted,
            }
        if targeted:
            cond.update({
                'shuffle_mode': shuffle_mode,
                'shuffle_tag': shuffle_tag,
                })
        best_key, counts = None, {}
        for key, config in self.conditioned(cond):
            name = config['name']
            if name in counts:
                counts[name] += 1
            else:
                counts[name] = 1
            preview = self.previews[key]
            if preview['success']:
                if min_prob is not None and preview['prob_adv']>=min_prob and (min_dist is None or min_dist>preview['dist']):
                    min_dist = preview['dist']
                    best_key = key
                if max_dist is not None and preview['dist']<=max_dist and (max_prob is None or max_prob<preview['prob_adv']):
                    max_prob = preview['prob_adv']
                    best_key = key
        adv = self.results[best_key]['adv']
        if min_prob is not None:
            return adv, min_dist, best_key, counts
        if max_dist is not None:
            return adv, max_prob, best_key, counts


if __name__=='__main__':
    parser = job_parser()
    parser.add_argument('--store_dir', default='store')
    parser.add_argument('--datasets_dir', default='datasets')
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--worker_num', default=WORKER_NUM, type=int)
    parser.add_argument('--max_seed', default=50, type=int)
    parser.add_argument('--sample_num', default=1000, type=int)
    args = parser.parse_args()

    if args.spec_pth is None:
        search_spec = {}
    else:
        with open(args.spec_pth, 'rb') as f:
            search_spec = pickle.load(f)

    if 'model_pth' not in search_spec:
        export_dir = '/'.join([args.store_dir, 'models'])
        assert os.path.exists(export_dir), "directory of models not found"
        search_spec['model_pth'] = [
            '/'.join([export_dir, f]) for f in os.listdir(export_dir) if f.endswith('.pt')
            ]
    if 'metric' not in search_spec:
        search_spec['metric'] = METRICS
    if 'eps_level' not in search_spec:
        search_spec['eps_level'] = [None]+EPS_LEVELS
    if 'targeted' not in search_spec:
        search_spec['targeted'] = [False, True]
    if 'name' not in search_spec:
        search_spec['name'] = NAMES
    if 'sample_idx' not in search_spec:
        search_spec['sample_idx'] = list(range(args.sample_num))
    if 'seed' not in search_spec:
        search_spec['seed'] = list(range(args.max_seed))

    job = AttackJob(
        args.store_dir, args.datasets_dir, args.device, args.worker_num,
        )
    job.random_search(search_spec, args.process_num, args.max_wait, args.tolerance)
