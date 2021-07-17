# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:57:18 2020

@author: Zhe
"""

import os, argparse, pickle, random, time, torch
from datetime import datetime
import pytz
import numpy as np
import foolbox as fb

from jarvis import BaseJob, Archive
from jarvis.vision import prepare_datasets
from jarvis.utils import job_parser, get_seed, set_seed, time_str

from . import DEVICE, WORKER_NUM

METRICS = ['L2', 'LI']
IMG_SIZES = {
    'MNIST': 1*28*28,
    'CIFAR10': 3*32*32,
    'CIFAR100': 3*32*32,
    'ImageNet': 3*224*224,
    }
EPS_NUM = 100
OVERSHOOTS = np.linspace(0, 4, 9)


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
        parser.add_argument('--targeted', action='store_true',
                            help="whether the attack is targeted")
        parser.add_argument('--shuffle_mode', default='elm', choices=['elm', 'cls'],
                            help="shuffle mode of targeted attack labels")
        parser.add_argument('--shuffle_tag', default=0, type=int,
                            help="shuffule tag of targeted attack labels")
        parser.add_argument('--sample_idx', default=0, type=int,
                            help="index of sample to attack")
        parser.add_argument('--seed', default=0, type=int,
                            help="random seed")

        args, _ = parser.parse_known_args(arg_strs)

        assert args.model_pth is not None
        config = {
            'model_pth': args.model_pth,
            'metric': args.metric,
            'targeted': args.targeted,
            'shuffle_mode': args.shuffle_mode if args.targeted else None,
            'shuffle_tag': args.shuffle_tag if args.targeted else None,
            'sample_idx': args.sample_idx,
            'seed': get_seed(args.seed),
            }
        return config

    def shuffled_targets(self, targets, shuffle_mode, shuffle_tag):
        r"""Returns shuffled targets for attack.

        Original image labels are shuffled so that wrong labels are assigned as
        attack targets.

        Args
        ----
        targets: list of int
            The original labels of dataset, obtained by `targets = dataset.targets`.
        shuffle_mode: str
            The shuffle mode of targeted attack labels. ``'elm'`` means
            element-wise shuffling, and ``'cls'`` means class-wise shuffling.
        shuffle_tag: int
            The shuffle tag of targeted attack labels, will be used as a random
            seed.

        Returns
        -------
        targets: ndarray
            The target class label for all images in the dataset.

        """
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
        r"""Dataset attack for single image.

        Args
        ----
        model: PyTorch model
            The model to be attacked.
        dataset: Dataset
            The dataset used for attack, only the images will be used. Usually
            it is the testing set as it is available to the attacker.
        criterion: Foolbox criterion
            A criterion containing only one label, e.g. `criterion.labels` or
            `criterion.target_classes` is of length one.

        Returns
        -------
        images: tensor, (1, C, H, W)
            A single-image batch as the 'adversarial' example.

        """
        success = False
        for s_idx in random.sample(range(len(dataset)), len(dataset)):
            image, _ = dataset[s_idx]
            image = image[None].to(self.device)
            with torch.no_grad():
                logit = model(image)
            if criterion(image, logit).item():
                success = True
                break
        assert success, "no image from the dataset meets criterion"
        return image

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
            criterion = fb.criteria.TargetedMisclassification(targets)
        else:
            criterion = fb.criteria.Misclassification(labels)

        # PGD attacks
        if verbose:
            tic = time.time()
        if config['metric']=='L2':
            attack = fb.attacks.L2ProjectedGradientDescentAttack()
            eps_max = IMG_SIZES[task]**0.5/10
        if config['metric']=='LI':
            attack = fb.attacks.LinfProjectedGradientDescentAttack()
            eps_max = 0.5
        advs_pgd, successes_pgd = [], []
        for eps in np.arange(1, EPS_NUM+1)/EPS_NUM*eps_max:
            _, advs, successes = attack(fmodel, images, criterion, epsilons=eps)
            advs_pgd.append(advs)
            successes_pgd.append(successes)
        advs_pgd = torch.cat(advs_pgd)
        successes_pgd = torch.cat(successes_pgd)
        dists_pgd = attack.distance(images.expand(EPS_NUM, -1, -1, -1), advs_pgd)
        with torch.no_grad():
            logits_pgd = model(advs_pgd)
        probs_pgd = torch.nn.functional.softmax(logits_pgd, dim=1)
        probs_pgd, preds_pgd = probs_pgd.max(dim=1)
        if verbose:
            toc = time.time()
            print('PGD attacks for {} epsilons performed ({})'.format(
                EPS_NUM, time_str(toc-tic),
                ))

        # BB attack
        if verbose:
            tic = time.time()
        if config['metric']=='L2':
            attack = fb.attacks.L2BrendelBethgeAttack()
        if config['metric']=='LI':
            attack = fb.attacks.LinfinityBrendelBethgeAttack()
        advs_bb, successes_bb = [], []
        for overshoot in OVERSHOOTS:
            criterion.overshoot = overshoot
            starting_points = self.dataset_attack(
                model, dataset, criterion,
                ).to(self.device)
            _, advs, successes = attack(fmodel, images, criterion, epsilons=None, starting_points=starting_points)
            advs_bb.append(advs)
            successes_bb.append(successes)
        advs_bb = torch.cat(advs_bb)
        successes_bb = torch.cat(successes_bb)
        dists_bb = attack.distance(images.expand(len(OVERSHOOTS), -1, -1, -1), advs_bb)
        with torch.no_grad():
            logits_bb = model(advs_bb)
        probs_bb = torch.nn.functional.softmax(logits_bb, dim=1)
        probs_bb, preds_bb = probs_bb.max(dim=1)
        if verbose:
            toc = time.time()
            print('BB attacks for {} overshoots performed ({})'.format(
                len(OVERSHOOTS), time_str(toc-tic),
                ))

        result = {
            'label_raw': label, 'prob_raw': prob_raw, 'pred_raw': pred_raw,
            'target': targets.item() if config['targeted'] else None,
            'advs_pgd': advs_pgd.cpu().numpy(), # (EPS_NUM, C, H, W)
            'successes_pgd': successes_pgd.cpu().numpy(), # (EPS_NUM,)
            'dists_pgd': dists_pgd.cpu().numpy(), # (EPS_NUM,)
            'probs_pgd': probs_pgd.cpu().numpy(), # (EPS_NUM,)
            'preds_pgd': preds_pgd.cpu().numpy(), # (EPS_NUM,)
            'advs_bb': advs_bb.cpu().numpy(), # (OVERSHOOT_NUM, C, H, W)
            'successes_bb': successes_bb.cpu().numpy(), # (OVERSHOOT_NUM,)
            'dists_bb': dists_bb.cpu().numpy(), # (OVERSHOOT_NUM,)
            'probs_bb': probs_bb.cpu().numpy(), # (OVERSHOOT_NUM,)
            'preds_bb': preds_bb.cpu().numpy(), # (OVERSHOOT_NUM,)
            }
        preview = {
            'label_raw': label, 'prob_raw': prob_raw, 'pred_raw': pred_raw,
            'target': targets.item() if config['targeted'] else None,
            'successes_pgd': successes_pgd.cpu().numpy(), # (EPS_NUM,)
            'dists_pgd': dists_pgd.cpu().numpy(), # (EPS_NUM,)
            'probs_pgd': probs_pgd.cpu().numpy(), # (EPS_NUM,)
            'preds_pgd': preds_pgd.cpu().numpy(), # (EPS_NUM,)
            'successes_bb': successes_bb.cpu().numpy(), # (OVERSHOOT_NUM,)
            'dists_bb': dists_bb.cpu().numpy(), # (OVERSHOOT_NUM,)
            'probs_bb': probs_bb.cpu().numpy(), # (OVERSHOOT_NUM,)
            'preds_bb': preds_bb.cpu().numpy(), # (OVERSHOOT_NUM,)
            }
        return result, preview

    def _get_default_arg(self, min_probs, max_dists):
        if min_probs is not None:
            assert max_dists is None
            if isinstance(min_probs, float):
                min_probs = [min_probs]
            assert isinstance(min_probs, list)
        if max_dists is not None:
            assert min_probs is None
            if isinstance(max_dists, (int, float)):
                max_dists = [1.*max_dists]
            assert isinstance(max_dists, list)
        return min_probs, max_dists

    def _best_keys(
            self, model_pth, metric, targeted, sample_idxs, *,
            min_probs=None, max_dists=None,
            shuffle_mode='elm', shuffle_tag=0,
            ):
        r"""Returns the best attacks found so far.

        Depending on which of `min_probs` and `max_dists` are provided, the
        best attack can either be the minimum perturbed example given a minimum
        probability requirement, or the most confident example given a maximum
        perturbation budget.

        Args
        ----
        model_pth: str
            The path to saved model.
        metric: str
            Perturbation metric, can be ``'LI'`` or ``'L2'``.
        targeted: bool
            Whether the attack is targeted or not.
        sample_idx: int
            The indices of samples to be attacked.
        min_probs: float or list of floats
            The minimum probability requirement of an adversarial example. If a
            single value is provided, it will be treated as a list of length 1.
        max_dists: float or list of floats
            The maximum perturbation budget of an adversarial example. If a
            single vlaue is provided, it will be treated as a list of length 1.
        names: list of str
            The attacks will be considered.
        shuffle_mode: str
            The shuffle mode of targeted attack labels.
        shuffle_tag: int
            The shuffle tag of targeted attack labels.

        Returns
        -------
        try_counts: dict
            The number of attacks tried so far. The keys of `try_counts` are
            `names`, and each value is an array of the same length as
            `sample_idxs`.
        best_keys: ndarray, (*, len(sample_idxs))
            The best key for each sample and each `min_prob` or `max_dist`.
        min_dists or max_probs: ndarray, (*, len(sample_idxs))
            The perturbation size or the prediction probability of the best
            attacks found, for each `min_prob` or `max_dist`.

        Example
        -------
        >>> try_counts, best_keys, min_dists = job.best_attack(
                model_pth, metric, targeted, sample_idxs,
                min_probs=[0.1, 0.5, 0.9],
                )
        >>> try_counts, best_keys, max_probs = job.best_attack(
                model_pth, metric, targeted, sample_idxs,
                max_dists=0.03, names=['PGD', 'BB'],
                )

        """
        assert self.readonly, "the job needs to be read-only"
        try_counts = np.zeros((len(sample_idxs),), dtype=float)
        if min_probs is not None:
            best_keys = np.full((len(min_probs), len(sample_idxs)), None, dtype=object)
            min_dists = np.full((len(min_probs), len(sample_idxs)), np.inf, dtype=float)
        if max_dists is not None:
            best_keys = np.full((len(max_dists), len(sample_idxs)), None, dtype=object)
            max_probs = np.full((len(max_dists), len(sample_idxs)), 0, dtype=float)

        cond = {
            'model_pth': model_pth,
            'metric': metric,
            'targeted': targeted,
            }
        if targeted:
            cond.update({
                'shuffle_mode': shuffle_mode,
                'shuffle_tag': shuffle_tag,
                })
        for key, config in self.conditioned(cond):
            sample_idx = config['sample_idx']
            if sample_idx not in sample_idxs:
                continue
            j = sample_idxs.index(sample_idx)
            try_counts[j] += 1
            preview = self.previews[key]
            if min_probs is not None:
                for i, min_prob in enumerate(min_probs):
                    idxs = preview['probs_pgd']>min_prob





    #     try_counts = dict((name, np.zeros((len(sample_idxs),), dtype=float)) for name in names)
    #     if min_probs is not None:
    #         best_keys = np.full((len(min_probs), len(sample_idxs)), None, dtype=object)
    #         min_dists = np.full((len(min_probs), len(sample_idxs)), np.inf, dtype=float)
    #     if max_dists is not None:
    #         best_keys = np.full((len(max_dists), len(sample_idxs)), None, dtype=object)
    #         max_probs = np.full((len(max_dists), len(sample_idxs)), 0, dtype=float)

    #     cond = {
    #         'model_pth': model_pth,
    #         'metric': metric,
    #         'targeted': targeted,
    #         }
    #     if targeted:
    #         cond.update({
    #             'shuffle_mode': shuffle_mode,
    #             'shuffle_tag': shuffle_tag,
    #             })
    #     for key, config in self.conditioned(cond):
    #         sample_idx, name = config['sample_idx'], config['name']
    #         if not((sample_idx in sample_idxs) and (name in names)):
    #             continue
    #         j = sample_idxs.index(sample_idx)
    #         try_counts[name][j] += 1
    #         preview = self.previews[key]
    #         if not preview['success']:
    #             continue
    #         if min_probs is not None:
    #             for i, min_prob in enumerate(min_probs):
    #                 if preview['prob_adv']>=min_prob and preview['dist']<min_dists[i, j]:
    #                     min_dists[i, j] = preview['dist']
    #                     best_keys[i, j] = key
    #         if max_dists is not None:
    #             for i, max_dist in enumerate(max_dists):
    #                 if preview['dist']<=max_dist and preview['prob_adv']>max_probs[i, j]:
    #                     max_probs[i, j] = preview['prob_adv']
    #                     best_keys[i, j] = key
    #     if min_probs is not None:
    #         return try_counts, best_keys, min_dists
    #     if max_dists is not None:
    #         return try_counts, best_keys, max_probs

    def export_digest(self, model_pth, metric, targeted, sample_idxs,
                      min_probs=None, max_dists=None,
                      update_cache=False, digest_pth=None):
        if self.store_dir is None:
            self.cache_configs = Archive(hashable=True)
            self.cache_results = Archive()
        else:
            self.cache_configs = Archive(os.path.join(self.store_dir, 'cache', 'configs'), hashable=True)
            self.cache_results = Archive(os.path.join(self.store_dir, 'cache', 'results'))
        min_probs, max_dists = self._get_default_arg(min_probs, max_dists)

        config = {
            'model_pth': model_pth, 'metric': metric, 'targeted': targeted,
            'sample_idxs': list(sample_idxs),
            'min_probs': min_probs, 'max_dists': max_dists,
            }
        key = self.cache_configs.add(config)
        if key in self.cache_results and not update_cache:
            result = self.cache_results[key]
        else:
            try_counts, best_keys, vals = self.fetch_best_attacks(**config)
            result = {
                'update_time': datetime.now(pytz.timezone('US/Central')).strftime('%m/%d/%Y %H:%M:%S %Z %z'),
                'try_counts': try_counts,
                'best_keys': best_keys,
                }
            if min_probs is not None:
                result['min_dists'] = vals
            if max_dists is not None:
                result['max_probs'] = vals
            result['configs'] = np.full(best_keys.shape, None, dtype=object)
            result['advs'] = np.full(best_keys.shape, None, dtype=object)
            result['probs'] = np.full(best_keys.shape, None, dtype=object)
            result['preds'] = np.full(best_keys.shape, None, dtype=object)
            for i in range(best_keys.shape[0]):
                for j in range(best_keys.shape[1]):
                    if best_keys[i, j] is not None:
                        result['configs'][i, j] = self.configs[best_keys[i, j]]
                        result['advs'][i, j] = self.results[best_keys[i, j]]['adv']
                        result['probs'][i, j] = self.results[best_keys[i, j]]['prob_adv']
                        result['preds'][i, j] = self.results[best_keys[i, j]]['pred_adv']
            self.cache_results[key] = result
        digest = dict(**config, **result)
        if digest_pth is not None:
            with open(digest_pth, 'wb') as f:
                pickle.dump(digest, f)
        return digest

    # def check_counts(self, try_counts, min_trys=None):
    #     if min_trys is None:
    #         min_trys = dict((name, 1 if name in ['PGD', 'BB'] else 0) for name in try_counts)
    #     is_enough = None
    #     for name in try_counts:
    #         _is_enough = try_counts[name]>=(min_trys[name] if name in min_trys else 0)
    #         if is_enough is None:
    #             is_enough = _is_enough
    #         else:
    #             is_enough &= _is_enough
    #     return is_enough

    # def get_succ_rates(self, digest, eps_ticks, min_prob=0.5, min_trys=None):
    #     assert digest['min_probs'] is not None and min_prob in digest['min_probs']
    #     is_enough = self.check_counts(digest['try_counts'], min_trys)
    #     dists = digest['min_dists'][digest['min_probs'].index(min_prob), is_enough]
    #     succ_rates = np.array([np.mean(dists<=eps) for eps in eps_ticks])
    #     return succ_rates


if __name__=='__main__':
    parser = job_parser()
    parser.add_argument('--store_dir', default='store')
    parser.add_argument('--datasets_dir', default='datasets')
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--worker_num', default=WORKER_NUM, type=int)
    parser.add_argument('--models_dir')
    parser.add_argument('--sample_num', default=1000, type=int)
    parser.add_argument('--max_seed', default=50, type=int)
    args = parser.parse_args()

    if args.spec_pth is None:
        search_spec = {}
    else:
        with open(args.spec_pth, 'rb') as f:
            search_spec = pickle.load(f)

    if 'model_pth' not in search_spec:
        export_dir = '/'.join([args.store_dir, 'models']) if args.models_dir is None else args.models_dir
        assert os.path.exists(export_dir), "directory of models not found"
        search_spec['model_pth'] = [
            '/'.join([export_dir, f]) for f in os.listdir(export_dir) if f.endswith('.pt')
            ]
    if 'metric' not in search_spec:
        search_spec['metric'] = METRICS
    if 'targeted' not in search_spec:
        search_spec['targeted'] = [False, True]
    if 'sample_idx' not in search_spec:
        search_spec['sample_idx'] = list(range(args.sample_num))
    if 'seed' not in search_spec:
        search_spec['seed'] = list(range(args.max_seed))

    job = AttackJob(
        args.store_dir, args.datasets_dir, args.device, args.worker_num,
        )
    job.random_search(search_spec, args.process_num, args.max_wait, args.tolerance)
