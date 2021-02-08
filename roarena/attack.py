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
EPS_RESOL = {'L2': 0.01, 'LI': 1/255}


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

    def __init__(self, store_dir, datasets_dir, device=DEVICE, worker_num=WORKER_NUM):
        if store_dir is None:
            super(AttackJob, self).__init__()
        else:
            super(AttackJob, self).__init__(os.path.join(store_dir, 'attacks'))
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
        if args.eps is None: # find minimum eps that gives successful attack
            args.name = 'BB' # only use boundary attack
            eps_level = None
        else:
            eps_level = int(args.eps/EPS_RESOL[args.metric])
            assert eps_level>0
        return {
            'model_pth': args.model_pth,
            'seed': get_seed(args.seed),
            'metric': args.metric,
            'name': args.name,
            'targeted': args.targeted,
            'eps_level': eps_level,
            'batch_idx': args.batch_idx,
            }

    def prepare_batch(self, dataset, batch_idx, targeted, targets_pth=None):
        r"""Prepares an image batch and the attack criterion.

        Args
        ----
        dataset: Dataset
            The testing set.
        batch_idx: int
            The index of the batch to attack. Batch size is `BATCH_SIZE`.
        targeted: bool
            Whether the attack is targeted.
        targets_pth: str
            The incorrect targets, which is precomputed and saved in a file.
            It will be ignored when `targeted==False`.

        Returns
        -------
        images: ep tensor,
            A batch of images.
        criterion: fb criteria
            The criterion used for foolbox attack.

        """
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
                assert np.all(successes)
                print('mean distance: {:.3f} ({})'.format(dists.mean(), time_str(toc-tic)))
            else:
                assert np.all(dists<=eps*1.1)
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

    def pool_results(self, model_pth, metric='L2', targeted=False, eps=None, *,
                     max_batch_num=None, preview_only=False):
        r"""Pools the results for one model.

        Args
        ----
        model_pth: str
            The path of the model file, which can be loaded by `torch.load`.
        metric: str
            The adversarial metric.
        targeted: bool
            Whether the attack is targeted.
        eps: float
            The attack size.
        max_batch_num: int
            The maximum number of batches to gather. Gather all available
            results when `max_batch_num` is ``None``.
        preview: bool
            Whether to pool previews only.

        Returns
        -------
        batch_idxs: list
            The batch indices that adversarial examples are calculated with at
            least one attack.
        advs: (N, C, H, W), array_like
            The adversarial examples.
        successes: (N,), array_like
            Whether attack is successful for each example.
        dists: (N,), array_like
            The distance between adversarial examples and the original images.

        """
        if eps is None:
            eps_level = None
        else:
            eps_level = int(eps/EPS_RESOL[metric])
        cond = {
            'model_pth': model_pth,
            'metric': metric,
            'targeted': targeted,
            'eps_level': eps_level
            }

        successes, dists = {}, {}
        if not preview_only:
            advs = {}
        for key, config in self.conditioned(cond):
            batch_idx = config['batch_idx']
            if preview_only:
                preview = self.previews[key]
                _successes, _dists = preview['successes'], preview['dists']
            else:
                result = self.results[key]
                _successes, _dists, _advs = result['successes'], result['dists'], result['advs']
            if batch_idx in advs:
                if eps is None:
                    idxs, = np.nonzero(_dists<dists[batch_idx])
                else:
                    idxs, = np.nonzero(_successes.astype(np.float)>successes[batch_idx].astype(np.float))
                successes[batch_idx][idxs] = _successes[idxs]
                dists[batch_idx][idxs] = _dists[idxs]
                if not preview_only:
                    advs[batch_idx][idxs] = _advs[idxs]
            else:
                successes[batch_idx] = _successes
                dists[batch_idx] = _dists
                if not preview_only:
                    advs[batch_idx] = _advs
            if max_batch_num is not None and len(advs)==max_batch_num:
                break
        batch_idxs = sorted(list(advs.keys()))
        if batch_idxs:
            successes = np.concatenate([successes[batch_idx] for batch_idx in batch_idxs])
            dists = np.concatenate([dists[batch_idx] for batch_idx in batch_idxs])
            if preview_only:
                advs = None
            else:
                advs = np.concatenate([advs[batch_idx] for batch_idx in batch_idxs])
            return batch_idxs, successes, dists, advs
        else:
            raise RuntimeError(f"no results found for {model_pth}")

    def summarize(self, model_pths, metric='L2', targeted=False, eps=None, max_batch_num=None):
        r"""Summarizes a list of models.

        Args
        ----
        model_pths: list
            A list of model paths, each of which can be loaded by `torch.load`.
        metric: str
            The adversarial metric.
        targeted: bool
            Whether the attack is targeted.
        eps: float
            The attack size.
        max_batch_num: int
            The maximum number of batches to gather. Gather all available
            results when `max_batch_num` is ``None``.

        Returns
        -------
        success_rates: (model_num,), array_like
            The success rates of attacking each model, only valid when `eps` is
            not ``None``.
        dist_percentiles: (model_num, 101), array_like
            The distance percentiles of each model, only valid when `eps` is
            ``None``.

        Examples
        --------
        >>> success_rates, _ = job.summarize(model_pths, 'L2', False, 0.5)

        >>> _, dist_percentiles = job.summarize(model_pths, 'LI', True, None)

        """
        success_rates, dist_percentiles = [], []
        for model_pth in model_pths:
            tic = time.time()
            _, successes, dists, _ = self.pool_results(
                model_pth, metric, targeted, eps,
                max_batch_num=max_batch_num, preview_only=True,
                )
            success_rates.append(successes.mean())
            dist_percentiles.append(np.percentile(dists, np.arange(101)))
            toc = time.time()
            print('results pooled for {} ({})'.format(model_pth, time_str(toc-tic)))
        success_rates = np.array(success_rates)
        dist_percentiles = np.array(dist_percentiles)
        return success_rates, dist_percentiles


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
            'model_pth': [os.path.join(export_dir, f) for f in os.listdir(export_dir) if f.endswith('.pt')],
            'seed': list(range(args.max_seed)),
            'metric': [args.metric],
            'name': ['BB'],
            'targeted': [False, True],
            'eps': [None],
            'batch_idx': list(range(args.batch_num)),
            }
    else:
        with open(args.spec_pth, 'rb') as f:
            search_spec = pickle.load(f)

    job = AttackJob(
        args.store_dir, args.datasets_dir, args.device, args.worker_num
        )
    job.random_search(search_spec, args.process_num, args.max_wait, args.tolerance)
