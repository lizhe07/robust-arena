import os, argparse, pickle, random, time, torch, json
from datetime import datetime
import pytz
import numpy as np
import foolbox as fb

from jarvis import BaseJob, Archive
from jarvis.vision import prepare_datasets
from jarvis.utils import job_parser, get_seed, set_seed, time_str

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


class AttackJob(BaseJob):
    r"""Performs adversarial attacks."""

    def __init__(self,
        store_dir, datasets_dir, device=DEVICE,
        **kwargs,
    ):
        super(AttackJob, self).__init__(store_dir=store_dir, **kwargs)
        self.datasets_dir = datasets_dir
        self.device = device if torch.cuda.is_available() else 'cpu'

    def strs2config(self, arg_strs):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model-path', help="path to the model")
        parser.add_argument('--metric', default='LI', choices=METRICS,
                            help="perturbation metric")
        parser.add_argument('--targeted', action='store_true',
                            help="whether the attack is targeted")
        parser.add_argument('--shuffle-mode', default='elm', choices=['elm', 'cls'],
                            help="shuffle mode of targeted attack labels")
        parser.add_argument('--shuffle-tag', default=0, type=int,
                            help="shuffule tag of targeted attack labels")
        parser.add_argument('--sample-idx', default=0, type=int,
                            help="index of sample to attack")

        args, _ = parser.parse_known_args(arg_strs)

        assert args.model_path is not None
        config = {
            'model_path': args.model_path,
            'metric': args.metric,
            'targeted': args.targeted,
            'shuffle_mode': args.shuffle_mode if args.targeted else None,
            'shuffle_tag': args.shuffle_tag if args.targeted else None,
            'sample_idx': args.sample_idx,
        }
        return config

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

    def main(self, config, num_epochs=1, verbose=1):
        if verbose>0:
            print(config)

        # load model
        saved = torch.load(config['model_path'])
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

        # load checkpoint
        try:
            epoch, ckpt, _ = self.load_ckpt(config)
            if verbose>0:
                print(f"Checkpoint (epoch {epoch}) loaded successfully.")
        except:
            epoch = 0
            ckpt = {
                'label_raw': label, 'prob_raw': prob_raw, 'pred_raw': pred_raw,
                'target': targets.item() if config['targeted'] else None,
            }
        while epoch<num_epochs:
            if verbose>0:
                print(f"Epoch {epoch}")
            # PGD attacks
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
            toc = time.time()
            if verbose>0:
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
            if config['metric']=='L2':
                attack = fb.attacks.L2BrendelBethgeAttack()
            if config['metric']=='LI':
                attack = fb.attacks.LinfinityBrendelBethgeAttack()
            advs_bb, successes_bb = [], []
            for overshoot in OVERSHOOTS:
                criterion.overshoot = overshoot
                if successes_pgd[-1]:
                    starting_points = torch.tensor(advs_pgd[-1][None], dtype=torch.float, device=self.device)
                else:
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
            toc = time.time()
            if verbose>0:
                print('BB attacks for {} overshoots performed ({})'.format(
                    len(OVERSHOOTS), time_str(toc-tic),
                ))
            advs_bb = advs_bb.cpu().numpy() # (OVERSHOOT_NUM, C, H, W)
            successes_bb = successes_bb.cpu().numpy() # (OVERSHOOT_NUM,)
            dists_bb = dists_bb.cpu().numpy() # (OVERSHOOT_NUM,)
            probs_bb = probs_bb.cpu().numpy() # (OVERSHOOT_NUM,)
            preds_bb = preds_bb.cpu().numpy() # (OVERSHOOT_NUM,)

            # update best attacks
            epoch += 1
            if 'advs_pgd' not in ckpt:
                ckpt.update({
                    'advs_pgd': advs_pgd,
                    'successes_pgd': successes_pgd,
                    'dists_pgd': dists_pgd,
                    'probs_pgd': probs_pgd,
                    'preds_pgd': preds_pgd,
                })
            else:
                # success attacks with higher confidence
                idxs = successes_pgd&(~ckpt['successes_pgd']|(probs_pgd>ckpt['probs_pgd']))
                ckpt['advs_pgd'][idxs] = advs_pgd[idxs]
                ckpt['successes_pgd'][idxs] = successes_pgd[idxs]
                ckpt['dists_pgd'][idxs] = dists_pgd[idxs]
                ckpt['probs_pgd'][idxs] = probs_pgd[idxs]
                ckpt['preds_pgd'][idxs] = preds_pgd[idxs]
            if 'advs_bb' not in ckpt:
                ckpt.update({
                    'advs_bb': advs_bb,
                    'successes_bb': successes_bb,
                    'dists_bb': dists_bb,
                    'probs_bb': probs_bb,
                    'preds_bb': preds_bb,
                })
            else:
                # success attacks with smaller perturbation
                idxs = successes_bb&(~ckpt['successes_bb']|(dists_bb<ckpt['dists_bb']))
                ckpt['advs_bb'][idxs] = advs_bb[idxs]
                ckpt['successes_bb'][idxs] = successes_bb[idxs]
                ckpt['dists_bb'][idxs] = dists_bb[idxs]
                ckpt['probs_bb'][idxs] = probs_bb[idxs]
                ckpt['preds_bb'][idxs] = preds_bb[idxs]
        preview = dict((key, ckpt[key]) for key in [
            'label_raw', 'prob_raw', 'pred_raw', 'target',
            'successes_pgd', 'dists_pgd', 'probs_pgd', 'preds_pgd',
            'successes_bb', 'dists_bb', 'probs_bb', 'preds_bb',
        ])
        return ckpt, preview

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
    parser = job_parser()
    parser.add_argument('--store-dir', default='store')
    parser.add_argument('--datasets-dir', default='datasets')
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--spec-path', default='a-tests/spec.json')
    parser.add_argument('--num-epochs', default=1, type=int)
    args, _ = parser.parse_known_args()

    time.sleep(random.random()*args.max_wait)
    job = AttackJob(
        f'{args.store_dir}/a-tests', args.datasets_dir, args.device,
    )

    try:
        with open(f'{args.store_dir}/{args.spec_path}', 'r') as f:
            search_spec = json.load(f)
    except:
        search_spec = {}
    if search_spec.get('model-path') is None:
        search_spec['model-path'] = [
            f'{args.store_dir}/exported/{f}' for f in os.listdir(f'{args.store_dir}/exported') if f.endswith('.pt')
        ]
    if search_spec.get('metric') is None:
        search_spec['metric'] = METRICS
    if search_spec.get('targeted') is None:
        search_spec['targeted'] = [False, True]
    if search_spec.get('shuffle-mode') is None:
        search_spec['shuffle-mode'] = ['elm', 'cls']
    if search_spec.get('shuffle-tag') is None:
        search_spec['shuffle-tag'] = list(range(6))
    if search_spec.get('sample-idx') is None:
        search_spec['sample-idx'] = list(range(1000))
    job.grid_search(search_spec, num_epochs=args.num_epochs, num_works=args.num_works, patience=args.patience)
