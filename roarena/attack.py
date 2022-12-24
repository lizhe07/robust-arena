import os, yaml, pickle, random, time, torch
from datetime import datetime
import pytz
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import foolbox as fb

from jarvis.archive import Archive
from jarvis.config import Config, from_cli
from jarvis.manager import Manager
from jarvis.vision import prepare_datasets
from jarvis.utils import time_str

from . import DEVICE

METRICS = ['L2', 'LI']
IMG_SIZES = {
    'MNIST': 1*28*28,
    'CIFAR10': 3*32*32,
    'CIFAR100': 3*32*32,
    'ImageNet': 3*224*224,
}
EPS_NUM = 100
OVERSHOOTS = np.linspace(0, 4, 9)

cli_args = Config({
    'store_dir': 'store',
    'datasets_dir': 'datasets',
    'model_path': [],
    'metric': ['LI', 'L2'],
    'targeted': [False, True],
    'shuffle_mode': ['elm', 'cls'],
    'shuffle_tag': list(range(6)),
    'num_samples': 1000,
    'num_epochs': 4,
})


class AttackManager(Manager):

    def __init__(self,
        store_dir: str, datasets_dir: str,
        device: str = DEVICE,
        **kwargs,
    ):
        super(AttackManager, self).__init__(f'{store_dir}/a-results', **kwargs)
        self.datasets_dir = datasets_dir
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.defaults = Config({
            'metric': 'LI',
            'targeted': False,
            'shuffle_mode': 'elm',
            'shuffle_tag': 0,
            'sample_idx': 0,
        })

    def get_config(self, config):
        config = super(AttackManager, self).get_config(config)
        if not config.targeted:
            config.shuffle_mode = None
            config.shuffle_tag = None
        return config

    def setup(self, config):
        super(AttackManager, self).setup(config)
        if self.verbose>0:
            print(self.config)

        # load model
        saved = torch.load(config['model_path'])
        self.task, self.model = saved['task'], saved['model']
        self.model.eval().to(self.device)
        self.fmodel = fb.PyTorchModel(self.model, bounds=(0, 1))

        # prepare testing dataset
        self.dataset = prepare_datasets(self.task, self.datasets_dir)

        # prepare single-sample batch for attack
        sample_idx = config['sample_idx']
        self.image, self.label = self.dataset[sample_idx]
        self.images = self.image[None].to(self.device)
        self.labels = torch.tensor([self.label], dtype=torch.long, device=self.device)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(self.model(self.images), dim=1)
            prob, pred = probs.max(dim=1)
        self.prob_raw, self.pred_raw = prob.item(), pred.item()

        # prepare criterion
        if config['targeted']:
            targets = self.shuffled_targets(
                self.dataset.targets, config['shuffle_mode'], config['shuffle_tag'],
            )
            self.targets = torch.tensor([targets[sample_idx]], dtype=torch.long, device=self.device)
            self.criterion = fb.criteria.TargetedMisclassification(self.targets)
        else:
            self.criterion = fb.criteria.Misclassification(self.labels)

    def init_ckpt(self):
        super(AttackManager, self).init_ckpt()
        self.ckpt.update({
            'label_raw': self.label, 'prob_raw': self.prob_raw, 'pred_raw': self.pred_raw,
            'target': self.targets.item() if self.config.targeted else None,
            'successes_pgd': np.full(EPS_NUM, fill_value=False, dtype=bool),
            'probs_pgd': np.full(EPS_NUM, fill_value=0., dtype=float),
            'successes_bb': np.full(len(OVERSHOOTS), fill_value=False, dtype=bool),
            'dists_bb': np.full(len(OVERSHOOTS), fill_value=np.inf, dtype=float),
        })

    def train(self):
        # PGD attacks
        tic = time.time()
        if self.config.metric=='L2':
            attack = fb.attacks.L2ProjectedGradientDescentAttack()
            eps_max = IMG_SIZES[self.task]**0.5/10
        if self.config.metric=='LI':
            attack = fb.attacks.LinfProjectedGradientDescentAttack()
            eps_max = 0.5
        self.criterion.overshoot = 0.
        advs_pgd, successes_pgd = [], []
        for eps in np.arange(1, EPS_NUM+1)/EPS_NUM*eps_max:
            _, advs, successes = attack(self.fmodel, self.images, self.criterion, epsilons=eps)
            advs_pgd.append(advs)
            successes_pgd.append(successes)
        advs_pgd = torch.cat(advs_pgd)
        successes_pgd = torch.cat(successes_pgd)
        dists_pgd = attack.distance(self.images.expand(EPS_NUM, -1, -1, -1), advs_pgd)
        with torch.no_grad():
            logits_pgd = self.model(advs_pgd)
        probs_pgd = torch.nn.functional.softmax(logits_pgd, dim=1)
        probs_pgd, preds_pgd = probs_pgd.max(dim=1)
        toc = time.time()
        if self.verbose>0:
            print('PGD attacks for {} epsilons performed ({})'.format(
                EPS_NUM, time_str(toc-tic),
            ))
        advs_pgd = advs_pgd.cpu().numpy() # (EPS_NUM, C, H, W)
        successes_pgd = successes_pgd.cpu().numpy() # (EPS_NUM,)
        dists_pgd = dists_pgd.cpu().numpy() # (EPS_NUM,)
        probs_pgd = probs_pgd.cpu().numpy() # (EPS_NUM,)
        preds_pgd = preds_pgd.cpu().numpy() # (EPS_NUM,)

        # BB attack
        tic = time.time()
        if self.config.metric=='L2':
            attack = fb.attacks.L2BrendelBethgeAttack()
        if self.config.metric=='LI':
            attack = fb.attacks.LinfinityBrendelBethgeAttack()
        advs_bb, successes_bb = [], []
        for overshoot in OVERSHOOTS:
            self.criterion.overshoot = overshoot
            if successes_pgd[-1]:
                starting_points = torch.tensor(advs_pgd[-1][None], dtype=torch.float, device=self.device)
            else:
                starting_points = self.dataset_attack(
                    self.model, self.dataset, self.criterion,
                ).to(self.device)
            _, advs, successes = attack(
                self.fmodel, self.images, self.criterion,
                epsilons=None, starting_points=starting_points,
            )
            advs_bb.append(advs)
            successes_bb.append(successes)
        advs_bb = torch.cat(advs_bb)
        successes_bb = torch.cat(successes_bb)
        dists_bb = attack.distance(self.images.expand(len(OVERSHOOTS), -1, -1, -1), advs_bb)
        with torch.no_grad():
            logits_bb = self.model(advs_bb)
        probs_bb = torch.nn.functional.softmax(logits_bb, dim=1)
        probs_bb, preds_bb = probs_bb.max(dim=1)
        toc = time.time()
        if self.verbose>0:
            print('BB attacks for {} overshoots performed ({})'.format(
                len(OVERSHOOTS), time_str(toc-tic),
            ))
        advs_bb = advs_bb.cpu().numpy() # (OVERSHOOT_NUM, C, H, W)
        successes_bb = successes_bb.cpu().numpy() # (OVERSHOOT_NUM,)
        dists_bb = dists_bb.cpu().numpy() # (OVERSHOOT_NUM,)
        probs_bb = probs_bb.cpu().numpy() # (OVERSHOOT_NUM,)
        preds_bb = preds_bb.cpu().numpy() # (OVERSHOOT_NUM,)

        if 'advs_pgd' not in self.ckpt:
            self.ckpt.update({
                'advs_pgd': advs_pgd,
                'successes_pgd': successes_pgd,
                'dists_pgd': dists_pgd,
                'probs_pgd': probs_pgd,
                'preds_pgd': preds_pgd,
            })
        else:
            # success attacks with higher confidence
            idxs = successes_pgd&(~self.ckpt['successes_pgd']|(probs_pgd>self.ckpt['probs_pgd']))
            self.ckpt['advs_pgd'][idxs] = advs_pgd[idxs]
            self.ckpt['successes_pgd'][idxs] = successes_pgd[idxs]
            self.ckpt['dists_pgd'][idxs] = dists_pgd[idxs]
            self.ckpt['probs_pgd'][idxs] = probs_pgd[idxs]
            self.ckpt['preds_pgd'][idxs] = preds_pgd[idxs]
        if 'advs_bb' not in self.ckpt:
            self.ckpt.update({
                'advs_bb': advs_bb,
                'successes_bb': successes_bb,
                'dists_bb': dists_bb,
                'probs_bb': probs_bb,
                'preds_bb': preds_bb,
            })
        else:
            # success attacks with smaller perturbation
            idxs = successes_bb&(~self.ckpt['successes_bb']|(dists_bb<self.ckpt['dists_bb']))
            self.ckpt['advs_bb'][idxs] = advs_bb[idxs]
            self.ckpt['successes_bb'][idxs] = successes_bb[idxs]
            self.ckpt['dists_bb'][idxs] = dists_bb[idxs]
            self.ckpt['probs_bb'][idxs] = probs_bb[idxs]
            self.ckpt['preds_bb'][idxs] = preds_bb[idxs]

    def eval(self):
        self.ckpt['eval_records'][self.epoch] = {
            'probs_pgd': self.ckpt['probs_pgd'].copy(),
            'dists_bb': self.ckpt['dists_bb'].copy(),
        }
        self.preview = dict((key, self.ckpt[key]) for key in [
            'label_raw', 'prob_raw', 'pred_raw', 'target',
            'successes_pgd', 'dists_pgd', 'probs_pgd', 'preds_pgd',
            'successes_bb', 'dists_bb', 'probs_bb', 'preds_bb',
        ] if key in self.ckpt)

    @staticmethod
    def shuffled_targets(targets, shuffle_mode, shuffle_tag):
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
        rng = np.random.default_rng(shuffle_tag)
        labels_all = np.array(targets)
        if shuffle_mode=='elm':
            last = labels_all.size
            targets = rng.permutation(labels_all)
            while True:
                idxs, = np.nonzero(targets==labels_all)
                if idxs.size>0:
                    if idxs.size<last:
                        last = idxs.size
                        targets[rng.permutation(idxs)] = targets[idxs]
                    else:
                        last = labels_all.size
                        targets = rng.permutation(labels_all)
                else:
                    break
        if shuffle_mode=='cls':
            _labels = np.unique(labels_all)
            while True:
                _targets = rng.permutation(_labels)
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
        rng = np.random.default_rng()
        success = False
        for s_idx in rng.permutation(len(dataset)):
            image, _ = dataset[s_idx]
            image = image[None].to(self.device)
            with torch.no_grad():
                logit = model(image)
            if criterion(image, logit).item():
                success = True
                break
        assert success, "no image from the dataset meets criterion"
        return image

    def best_attack(self,
        model_path, metric, targeted, shuffle_mode, shuffle_tag, sample_idx,
        min_epoch=1, min_probs=None, max_dists=None, return_advs=False,
    ):
        assert min_probs is None or max_dists is None
        best_idxs, min_dists, max_probs = [], [], []
        attack_types = [] # 0 for PGD, 1 for BB
        config = {
            'model_path': model_path,
            'metric': metric, 'targeted': targeted,
            'shuffle_mode': shuffle_mode, 'shuffle_tag': shuffle_tag,
            'sample_idx': sample_idx,
        }
        key = self.configs.add(config)
        assert self.stats[key]['epoch']>=min_epoch, "No completed attacks found."
        preview = self.previews[key]

        if min_probs is not None: # find minimal attack with at least min_prob output
            for min_prob in min_probs:
                best_idx, min_dist, attack_type = None, np.inf, None
                idxs, = (preview['successes_pgd']&(preview['probs_pgd']>=min_prob)).nonzero()
                if len(idxs)>0:
                    idx = idxs[np.argmin(preview['dists_pgd'][idxs])]
                    if preview['dists_pgd'][idx]<min_dist:
                        best_idx = idx
                        min_dist = preview['dists_pgd'][idx]
                        attack_type = 0
                idxs, = (preview['successes_bb']&(preview['probs_bb']>=min_prob)).nonzero()
                if len(idxs)>0:
                    idx = idxs[np.argmin(preview['dists_bb'][idxs])]
                    if preview['dists_bb'][idx]<min_dist:
                        best_idx = idx
                        min_dist = preview['dists_bb'][idx]
                        attack_type = 1
                best_idxs.append(best_idx)
                min_dists.append(min_dist)
                attack_types.append(attack_type)
        if max_dists is not None: # find strongest attack with given attack budget
            for max_dist in max_dists:
                best_idx, max_prob, attack_type = None, 0, None
                idxs, = (preview['successes_pgd']&(preview['dists_pgd']<=max_dist)).nonzero()
                if len(idxs)>0:
                    idx = idxs[np.argmax(preview['probs_pgd'][idxs])]
                    if preview['probs_pgd'][idx]>max_prob:
                        best_idx = idx
                        max_prob = preview['probs_pgd'][idx]
                        attack_type = 0
                idxs, = (preview['successes_bb']&(preview['dists_bb']<=max_dist)).nonzero()
                if len(idxs)>0:
                    idx = idxs[np.argmax(preview['probs_bb'][idxs])]
                    if preview['probs_bb'][idx]>max_prob:
                        best_idx = idx
                        max_prob = preview['probs_bb'][idx]
                        attack_type = 1
                best_idxs.append(best_idx)
                max_probs.append(max_probs)
                attack_types.append(attack_type)

        if return_advs:
            ckpt = self.ckpts[key]
            advs = []
            for best_idx, attack_type in zip(best_idxs, attack_types):
                if best_idx is None:
                    advs.append(None)
                else:
                    if attack_type==0:
                        advs.append(ckpt['advs_pgd'][best_idx])
                    if attack_type==1:
                        advs.append(ckpt['advs_bb'][best_idx])
        else:
            advs = None
        return min_dists, max_probs, advs

    def summarize(
        self, model_paths, metric, targeted, shuffle_mode, shuffle_tag,
        min_epoch=1, min_prob=0.1, num_advs=0,
    ):
        q_ticks = 1-np.logspace(0, -3, 50)
        d_ticks = []
        shared_idxs = None
        for model_path in model_paths:
            cond = {
                'model_path': model_path,
                'metric': metric, 'targeted': targeted,
                'shuffle_mode': shuffle_mode, 'shuffle_tag': shuffle_tag,
            }
            min_dists, s_idxs = [], set()
            for key in self.completed(min_epoch=min_epoch, cond=cond):
                sample_idx = self.configs[key]['sample_idx']
                _min_dist, _, _ = self.best_attack(
                    model_path, metric, targeted, shuffle_mode, shuffle_tag, sample_idx,
                    min_epoch=min_epoch, min_probs=[min_prob], return_advs=False,
                )
                min_dists.append(_min_dist[0])
                s_idxs.add(sample_idx)
            if shared_idxs is None:
                shared_idxs = s_idxs
            else:
                shared_idxs &= s_idxs
            d_ticks.append(np.quantile(min_dists, q_ticks))
        if len(shared_idxs)<num_advs:
            raise RuntimeError("Not enough shared images ({}) successfully attacked for all models.".format(len(shared_idxs)))
        sample_idxs = random.sample(list(shared_idxs), num_advs)
        advs = []
        for model_path in model_paths:
            _advs = []
            for sample_idx in sample_idxs:
                _, _, _adv = self.best_attack(
                    model_path, metric, targeted, shuffle_mode, shuffle_tag, sample_idx,
                    min_epoch=min_epoch, min_probs=[min_prob], return_advs=True,
                )
                _advs.append(_adv[0])
            advs.append(_advs)
        advs = np.array(advs)
        return q_ticks, d_ticks, sample_idxs, advs

    def plot_digest(self, q_ticks, d_ticks, ax=None, colors=None, legends=None):
        num_groups = len(d_ticks)
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 3))
        if colors is None:
            colors = [matplotlib.cm.get_cmap('tab10')(x)[:3] for x in np.linspace(0, 1, num_groups)]
        if legends is None:
            legends = [f'Group {i}' for i in range(num_groups)]

        lines = []
        for i in range(num_groups):
            color = colors[i]
            h, = ax.plot(d_ticks[i], q_ticks*100., color=color)
            lines.append(h)
        ax.legend(lines, legends, fontsize=12)
        d_max = max(min(d_ticks[i][q_ticks>0.95]) for i in range(num_groups))
        ax.set_xlim([-0.1*d_max, 1.1*d_max])
        ax.set_xlabel(r'Attack size $\epsilon$')
        ax.set_ylim([0, 105])
        ax.set_ylabel('Success rate (%)')
        ax.grid(axis='y')
        return ax

    def _best_attacks(
            self, model_pth, metric, targeted, sample_idxs,
            min_probs, max_dists, shuffle_mode, shuffle_tag,
            ):
        r"""Finds the best attacks.

        See `export_digest` for more details.

        """
        assert self.readonly, "the job needs to be read-only"
        counts = np.zeros((len(sample_idxs),), dtype=float)
        if min_probs is not None:
            best_keys = np.full((len(min_probs), len(sample_idxs)), None, dtype=object)
        if max_dists is not None:
            best_keys = np.full((len(max_dists), len(sample_idxs)), None, dtype=object)
        best_advs = np.full(best_keys.shape, None, dtype=object)
        adv_dists = np.full(best_keys.shape, np.inf, dtype=float)
        adv_probs = np.full(best_keys.shape, 0, dtype=float)
        adv_preds = np.full(best_keys.shape, -1, dtype=int)

        cond = {
            'model_pth': model_pth,
            'metric': metric,
            'targeted': targeted,
            'shuffle_mode': shuffle_mode if targeted else None,
            'shuffle_tag': shuffle_tag if targeted else None,
            }
        for key, config in self.conditioned(cond):
            sample_idx = config['sample_idx']
            if sample_idx not in sample_idxs:
                continue
            j = sample_idxs.index(sample_idx)
            counts[j] += 1
            preview = self.previews[key]
            dists = np.concatenate([preview['dists_pgd'], preview['dists_bb']])
            probs = np.concatenate([preview['probs_pgd'], preview['probs_bb']])
            preds = np.concatenate([preview['preds_pgd'], preview['preds_bb']])
            if config['targeted']:
                successes = preds==preview['target']
            else:
                successes = preds!=preview['label_raw']
            if min_probs is not None:
                for i, min_prob in enumerate(min_probs):
                    idxs, = np.nonzero(successes&(probs>=min_prob))
                    if len(idxs)==0:
                        continue
                    idx = idxs[dists[idxs].argmin()]
                    if dists[idx]<adv_dists[i, j]:
                        best_keys[i, j] = (key, idx)
                        adv_dists[i, j] = dists[idx]
                        adv_probs[i, j] = probs[idx]
                        adv_preds[i, j] = preds[idx]
            if max_dists is not None:
                for i, max_dist in enumerate(max_dists):
                    idxs, = np.nonzero(successes&(dists<=max_dist))
                    if len(idxs)==0:
                        continue
                    idx = idxs[probs[idxs].argmax()]
                    if probs[idx]>adv_probs[i, j]:
                        best_keys[i, j] = (key, idx)
                        adv_dists[i, j] = dists[idx]
                        adv_probs[i, j] = probs[idx]
                        adv_preds[i, j] = preds[idx]
        for i in range(best_keys.shape[0]):
            for j in range(len(sample_idxs)):
                if best_keys[i, j] is not None:
                    key, idx = best_keys[i, j]
                    result = self.results[key]
                    best_advs[i, j] = np.concatenate([result['advs_pgd'], result['advs_bb']])[idx]
        return counts, adv_dists, adv_probs, adv_preds, best_advs

    def export_digest(
            self, model_pth, metric, targeted, sample_idxs, *,
            min_probs=None, max_dists=None, shuffle_mode='elm', shuffle_tag=0,
            update_cache=False, digest_pth=None,
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
        sample_idxs: iterable
            The indices of samples to be attacked.
        min_probs: float or list of floats
            The minimum probability requirement of an adversarial example. If a
            single value is provided, it will be treated as a list of length 1.
        max_dists: float or list of floats
            The maximum perturbation budget of an adversarial example. If a
            single vlaue is provided, it will be treated as a list of length 1.
        shuffle_mode: str
            The shuffle mode of targeted attack labels.
        shuffle_tag: int
            The shuffle tag of targeted attack labels.
        update_cache: bool
            Whether to update digest cache.
        digest_pth: str
            The export path of digest if not ``None``.

        Returns
        -------
        digest: dict
            A dictionary contianing input arguments and the following keys:

            update_time: str
                The update time.
            counts: ndarray of float, (len(sample_idxs),)
                Number of attacks tried for each sample.
            best_advs: ndarray of object, (len(min_dists), len(sample_idxs)) or
            (len(max_probs), len(sample_idxs))
                The best adversarial examples found, each row corresponds to
                each value in `min_probs` or `max_dists`.
            advs_dist: ndarray of float, same shape as `best_advs`
                The adversarial perturbation size.
            advs_prob: ndarray of float, same shape as `best_advs`
                The classification probabilities of adversarial examples.
            advs_pred: ndarray of int, same shape of `best_advs`
                The prediction label of adversarial examples.

        """
        if self.store_dir is None:
            self.cache_configs = Archive(hashable=True)
            self.cache_results = Archive()
        else:
            self.cache_configs = Archive(os.path.join(self.store_dir, 'cache', 'configs'), hashable=True)
            self.cache_results = Archive(os.path.join(self.store_dir, 'cache', 'results'))
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

        config = {
            'model_pth': model_pth, 'metric': metric, 'targeted': targeted,
            'sample_idxs': list(sample_idxs),
            'min_probs': min_probs, 'max_dists': max_dists,
            'shuffle_mode': shuffle_mode if targeted else None,
            'shuffle_tag': shuffle_tag if targeted else None,
        }
        key = self.cache_configs.add(config)
        if key in self.cache_results and not update_cache:
            result = self.cache_results[key]
        else:
            counts, adv_dists, adv_probs, adv_preds, best_advs = self._best_attacks(**config)
            result = {
                'update_time': datetime.now(pytz.timezone('US/Central')).strftime('%m/%d/%Y %H:%M:%S %Z %z'),
                'counts': counts,
                'adv_dists': adv_dists,
                'adv_probs': adv_probs,
                'adv_preds': adv_preds,
                'best_advs': best_advs,
                }
            self.cache_results[key] = result
        digest = dict(**config, **result)
        if digest_pth is not None:
            with open(digest_pth, 'wb') as f:
                pickle.dump(digest, f)
        return digest

    def get_succ_rates(self, digest, epsilons, min_prob=0.5, min_trys=1, verbose=True):
        r"""Gets attack successful rates.

        digest: dict
            A digest returned by `export_digest`.
        epsilons: list
            A list of perturbation sizes.
        min_prob: float
            The minimum reporting probability of adversarial attacks.
        min_trys: int
            The minimum number of attack trys for a sample.
        verbose: bool
            Whether to display information.

        succ_rates: ndarray
            The attack successful rates for each epsilon value.

        """
        assert digest['min_probs'] is not None and min_prob in digest['min_probs']
        is_enough = digest['counts']>=min_trys
        dists = digest['adv_dists'][digest['min_probs'].index(min_prob), is_enough]
        succ_rates = np.array([np.mean(dists<=eps) for eps in epsilons])
        if verbose:
            print('{} samples attacked with at least {} trys, mean distance {:.3f} with minimum probability {:.2f}'.format(
                is_enough.sum(), min_trys, dists.mean(), min_prob,
                ))
        return succ_rates


if __name__=='__main__':
    cli_args.update(from_cli())
    store_dir = cli_args.pop('store_dir')
    datasets_dir = cli_args.pop('datasets_dir')
    manager = AttackManager(store_dir, datasets_dir)

    choices = {}
    choices['model_path'] = cli_args.pop('model_path')
    choices['metric'] = cli_args.pop('metric')
    choices['targeted'] = cli_args.pop('targeted')
    choices['shuffle_mode'] = cli_args.pop('shuffle_mode')
    choices['shuffle_tag'] = cli_args.pop('shuffle_tag')
    choices['sample_idx'] = list(range(cli_args.pop('num_samples')))
    if isinstance(choices['model_path'], str):
        with open(choices['model_path']) as f:
            choices['model_path'] = yaml.safe_load(f)
    elif len(choices['model_path'])==0:
        choices['model_path'] = [
            f'{store_dir}/exported/{f}' for f in os.listdir(f'{store_dir}/exported') if f.endswith('.pt')
        ]

    manager.sweep(choices, **cli_args)
