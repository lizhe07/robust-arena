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
    BATCH_SIZE = 20

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
        parser.add_argument('--shuffle_mode', default='elm', choices=['elm', 'cls'])
        parser.add_argument('--shuffle_tag', default=0, type=int)
        parser.add_argument('--eps', type=float)
        parser.add_argument('--overshoot', default=0.01, type=float)
        parser.add_argument('--batch_idx', default=0, type=int)

        args, _ = parser.parse_known_args(arg_strs)

        assert args.model_pth is not None
        if args.eps is None: # find minimum eps that gives successful attack
            args.name = 'BB' # only use boundary attack
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
            'overshoot': args.overshoot,
            'batch_idx': args.batch_idx,
            }
        if args.targeted:
            config.update({
                'shuffle_mode': args.shuffle_mode,
                'shuffle_tag': args.shuffle_tag,
                })
        return config

    def prepare_batch(self, dataset, batch_idx, targeted, overshoot,
                      shuffle_mode='elm', shuffle_tag=0, **kwargs):
        r"""Prepares an image batch and the attack criterion.

        Args
        ----
        dataset: Dataset
            The testing set.
        batch_idx: int
            The index of the batch to attack. Batch size is `BATCH_SIZE`.
        targeted: bool
            Whether the attack is targeted.
        overshoot: float
            The overshoot parameter.
        shuffle_mode: str
            The mode of shuffled targets for targeted attack.
        shuffle_tag: int
            The integer tag for shuffled targets.

        Returns
        -------
        images: tensor
            A batch of images.
        criterion: fb criteria
            The criterion used for foolbox attack.

        """
        images, labels = [], []
        idx_min = AttackJob.BATCH_SIZE*batch_idx
        idx_max = min(AttackJob.BATCH_SIZE*(batch_idx+1), len(dataset))
        for idx in range(idx_min, idx_max):
            image, label = dataset[idx]
            images.append(image)
            labels.append(torch.tensor(label, dtype=torch.long))
        images = torch.stack(images).to(self.device)
        labels = torch.stack(labels).to(self.device)
        if targeted:
            set_seed(shuffle_tag)
            labels_all = np.array(dataset.targets)
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
            targets = torch.tensor(
                targets[idx_min:idx_max], dtype=torch.long, device=self.device,
                )
            criterion = fb.criteria.TargetedMisclassification(targets, overshoot)
        else:
            criterion = fb.criteria.Misclassification(labels, overshoot)
        return images, labels, criterion

    def dataset_attack(self, model, dataset, labels, targeted, overshoot):
        r"""Generates dataset attack.

        Images are randomly selected from the whole dataset as attacks, which
        serve as the initial point for boundary attack.

        model: nn.Module
            The model to attack.
        dataset: Dataset
            The full dataset.
        labels: tensor
            The batch of labels used for generating attacks. If the attack is
            targeted, `labels` is the targeted labels. If the attack is
            untargeted, `labels` is the true labels.
        targeted: bool
            Whether the attack is targeted.
        overshoot: float
            The overshoot parameter.

        """
        model.eval().to(self.device)

        advs = []
        for label in labels:
            if targeted:
                idxs, = np.nonzero(np.array(dataset.targets)==label.item())
                criterion = fb.criteria.TargetedMisclassification(label[None], overshoot)
            else:
                idxs, = np.nonzero(np.array(dataset.targets)!=label.item())
                criterion = fb.criteria.Misclassification(label[None], overshoot)
            while True:
                image, _ = dataset[random.choice(idxs)]
                with torch.no_grad():
                    logit = model(image[None].to(self.device))[0]
                is_adv = criterion(image[None], logit[None])[0]
                if is_adv:
                    break
            advs.append(image)
        return torch.stack(advs)

    def main(self, config, verbose=True):
        if verbose:
            print(config)

        # load model
        saved = torch.load(config['model_pth'])
        model = saved['model']
        model.eval().to(self.device)
        fmodel = fb.PyTorchModel(model, bounds=(0, 1))

        # prepare testing dataset
        kwargs = {'task': saved['task'], 'datasets_dir': self.datasets_dir}
        if 'grayscale' in saved:
            kwargs['grayscale'] = saved['grayscale']
        dataset = prepare_datasets(**kwargs)

        # prepare batch for attack
        images, labels, criterion = self.prepare_batch(dataset, **config)
        with torch.no_grad():
            logits = model(images)
            _, predicts = logits.max(dim=1)

        # initialize attack
        set_seed(config['seed'])
        attack = ATTACKS[config['metric']][config['name']]
        run_kwargs = {}
        if config['name']=='BB':
            starting_points = self.dataset_attack(
                model, dataset,
                criterion.target_classes if config['targeted'] else criterion.labels,
                config['targeted'], config['overshoot'],
                ).to(self.device)
            run_kwargs = {'starting_points': starting_points}

        # attack model with foolbox
        if verbose:
            tic = time.time()
        if config['eps_level'] is None:
            eps = None
        else:
            eps = config['eps_level']*EPS_RESOL[config['metric']]
        _, advs, successes = attack(fmodel, images, criterion, epsilons=eps, **run_kwargs)
        dists = attack.distance(images, advs)
        advs = advs.cpu().numpy().copy()
        successes = successes.cpu().numpy().copy()
        dists = dists.cpu().numpy().copy()

        images = images.cpu().numpy()
        predicts, labels = predicts.cpu().numpy(), labels.cpu().numpy()

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
            'predicts': predicts,
            'labels': labels,
            'successes': successes,
            'dists': dists,
            }
        preview = {
            'predicts': predicts,
            'labels': labels,
            'successes': successes,
            'dists': dists,
            }
        return result, preview

    def best_attack(self, model_pth, sample_idx, metric='L2', targeted=False,
                    overshoot=0.01, shuffle_mode='elm', shuffle_tag=0):
        batch_idx = sample_idx//AttackJob.BATCH_SIZE
        sample_idx = sample_idx%AttackJob.BATCH_SIZE
        cond = {
            'model_pth': model_pth,
            'metric': metric,
            'targeted': targeted,
            'eps_level': None,
            'overshoot': overshoot,
            'batch_idx': batch_idx,
            }
        if targeted:
            cond.update({
                'shuffle_mode': shuffle_mode,
                'shuffle_tag': shuffle_tag,
                })
        min_dist, best_key = None, None
        for key, config in self.conditioned(cond):
            preview = self.previews[key]
            _dists = preview['dists']
            if min_dist is None or min_dist>_dists[sample_idx]:
                min_dist = _dists[sample_idx]
                best_key = key
        result = self.results[best_key]
        best_adv = result['advs'][sample_idx]
        return min_dist, best_adv

    def pool_results(self, model_pth, metric='L2', targeted=False, eps=None,
                     overshoot=0.01, *, max_batch_num=None, preview_only=False):
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
        overshoot: float
            The overshoot parameter.
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
            'eps_level': eps_level,
            'overshoot': overshoot,
            }

        successes, dists = {}, {}
        if not preview_only:
            advs = {}
        batch_idxs = []
        for key, config in self.conditioned(cond):
            batch_idx = config['batch_idx']
            if preview_only:
                preview = self.previews[key]
                _successes, _dists = preview['successes'], preview['dists']
            else:
                result = self.results[key]
                _successes, _dists, _advs = result['successes'], result['dists'], result['advs']
            if batch_idx in batch_idxs:
                if eps is None:
                    idxs, = np.nonzero(_dists<dists[batch_idx])
                else:
                    idxs, = np.nonzero(_successes.astype(float)>successes[batch_idx].astype(float))
                successes[batch_idx][idxs] = _successes[idxs]
                dists[batch_idx][idxs] = _dists[idxs]
                if not preview_only:
                    advs[batch_idx][idxs] = _advs[idxs]
            else:
                successes[batch_idx] = _successes
                dists[batch_idx] = _dists
                if not preview_only:
                    advs[batch_idx] = _advs
                batch_idxs.append(batch_idx)
            if max_batch_num is not None and len(batch_idxs)==max_batch_num:
                break
        if batch_idxs:
            sample_idxs = []
            for batch_idx in batch_idxs:
                idx_min = AttackJob.BATCH_SIZE*batch_idx
                idx_max = idx_min+len(successes[batch_idx])
                sample_idxs += list(range(idx_min, idx_max))
            successes = np.concatenate([successes[batch_idx] for batch_idx in batch_idxs])
            dists = np.concatenate([dists[batch_idx] for batch_idx in batch_idxs])
            if preview_only:
                advs = None
            else:
                advs = np.concatenate([advs[batch_idx] for batch_idx in batch_idxs])
            return sample_idxs, successes, dists, advs
        else:
            raise RuntimeError(f"no results found for {model_pth}")

    def shared_idxs(self, model_pths, margin=0.1, **kwargs):
        s_idxs = None
        for model_pth in model_pths:
            _s_idxs, _, _dists, _ = self.pool_results(model_pth, preview_only=True, **kwargs)
            _s_idxs = set(np.array(_s_idxs)[(_dists>np.quantile(_dists, margin))&(_dists<np.quantile(_dists, 1-margin))])
            if s_idxs is None:
                s_idxs = _s_idxs
            else:
                s_idxs = s_idxs&_s_idxs
        s_idxs = list(s_idxs)
        return s_idxs

    def gather_images(self, dataset, model_pths, s_idxs, **kwargs):
        images, labels = [], []
        for s_idx in s_idxs:
            image, label = dataset[s_idx]
            images.append(image.numpy())
            labels.append(label)
        images, labels = np.stack(images), np.stack(labels)

        advs, predicts = [], []
        for model_pth in model_pths:
            _advs = np.stack([self.best_attack(model_pth, s_idx, **kwargs)[1] for s_idx in s_idxs])
            advs.append(_advs)

            model = torch.load(model_pth)['model']
            model.eval().to(self.device)
            with torch.no_grad():
                logits = model(torch.tensor(_advs, dtype=torch.float, device=self.device)).cpu()
                _, _predicts = logits.max(dim=1)
            predicts.append(_predicts.numpy())
        advs, predicts = np.stack(advs), np.stack(predicts)

        diffs = advs-images
        return images, labels, advs, predicts, diffs

    def summarize(self, model_pths, metric='L2', targeted=False, eps=None, overshoot=0.01, max_batch_num=None):
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
        overshoot: float
            The overshoot parameter.
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
            try:
                _, successes, dists, _ = self.pool_results(
                    model_pth, metric, targeted, eps, overshoot,
                    max_batch_num=max_batch_num, preview_only=True,
                    )
            except:
                continue
            success_rates.append(successes.mean())
            dist_percentiles.append(np.percentile(dists, np.arange(101)))
        success_rates = np.array(success_rates)
        dist_percentiles = np.array(dist_percentiles).reshape(-1, 101) # reshape for empty array
        return success_rates, dist_percentiles

    def plot_perturbation_distribution(self, ax, groups, **kwargs):
        dists = {}
        for tag, model_pths, _ in groups:
            dists[tag] = []
            for model_pth in model_pths:
                _, _, _dists, _ = self.pool_results(model_pth, preview_only=True, **kwargs)
                dists[tag].append(_dists)
            dists[tag] = np.concatenate(dists[tag])
        tags = [tag for tag, *_ in groups]
        ax.boxplot([dists[tag] for tag in tags], vert=False, sym='', labels=tags)
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_xlabel('minimum perturbation')


    def plot_comparison(self, ax, groups, dist_percentiles):
        r"""Plots comparison of groups.

        Args
        ----
        ax: matplot axis
            The axis for plotting.
        groups: list
            Each item is a tuple of `(tag, model_pths, color)`. `tag` is the
            label for the group, `model_pths` is the list of model pths and
            `color` is a color tuple of shape `(3,)`.
        dist_percentiles: list
            Each item is a dictionary returned by `summarize`.

        """
        p_ticks = np.arange(101)
        lines, legends = [], []
        for i, (tag, _, color) in enumerate(groups):
            d_mean = np.mean(dist_percentiles[i], axis=0)
            d_std = np.std(dist_percentiles[i], axis=0)
            idxs = (d_mean>0)&(p_ticks<=98)
            h, = ax.plot(d_mean[idxs], p_ticks[idxs], color=color)
            ax.fill_betweenx(
                p_ticks[idxs], d_mean[idxs]-d_std[idxs], d_mean[idxs]+d_std[idxs],
                color=color, alpha=0.2,
                )
            lines.append(h)
            legends.append(tag)
        ax.legend(lines, legends)
        ax.set_ylabel('success rate (%)')


if __name__=='__main__':
    parser = job_parser()
    parser.add_argument('--store_dir', default='store')
    parser.add_argument('--datasets_dir', default='datasets')
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--worker_num', default=WORKER_NUM, type=int)
    parser.add_argument('--max_seed', default=100, type=int)
    parser.add_argument('--batch_num', default=50, type=int)
    args = parser.parse_args()

    if args.spec_pth is None:
        export_dir = os.path.join(args.store_dir, 'models', 'exported')
        assert os.path.exists(export_dir), "directory of exported models not found"
        search_spec = {
            'model_pth': [os.path.join(export_dir, f) for f in os.listdir(export_dir) if f.endswith('.pt')],
            'seed': list(range(args.max_seed)),
            'metric': METRICS,
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
