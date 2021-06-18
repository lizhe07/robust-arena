# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 00:01:54 2021

@author: Zhe
"""

import os, argparse, pickle, torch
import numpy as np
from scipy.fft import fft2, ifft2

from jarvis import BaseJob
from jarvis.vision import prepare_datasets
from jarvis.utils import get_seed, set_seed, job_parser

from . import DEVICE, BATCH_SIZE, WORKER_NUM

ALPHAS = [0, 3, 4, 5, 6, 8, 9, 11, 13, 15, 17, 19, 22, 27, 33, 42, 57, 100]


class EinMonDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, alpha=5, seed=0):
        self.dataset = dataset

        set_seed(seed)
        labels_low = np.array(dataset.targets)
        labels_high = labels_low.copy()
        last = len(dataset)
        while True:
            idxs, = np.nonzero(labels_high==labels_low)
            if idxs.size>0:
                if idxs.size<last:
                    labels_high[idxs[np.random.permutation(idxs.size)]] = labels_high[idxs]
                    last = idxs.size
                else:
                    labels_high = labels_low.copy()[np.random.permutation(len(dataset))]
                    last = len(dataset)
            else:
                break

        idxs_low, idxs_high = np.arange(len(dataset)), np.arange(len(dataset))
        for c in range(len(dataset.classes)):
            _idxs, = np.nonzero(labels_low==c)
            idxs_high[labels_high==c] = np.random.permutation(_idxs)

        self.idxs_low, self.idxs_high = idxs_low, idxs_high

        img, _ = dataset[0]
        img_size = img.shape[1]
        assert img.shape[2]==img_size
        dx, dy = np.meshgrid(np.arange(img_size)/img_size, np.arange(img_size)/img_size)
        dx = np.mod(dx+0.5, 1)-0.5
        dy = np.mod(dy+0.5, 1)-0.5
        self.mask = ((dx**2+dy**2)**0.5<=alpha/100*0.5).astype(float)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_low, label_low = self.dataset[self.idxs_low[idx]]
        img_high, label_high = self.dataset[self.idxs_high[idx]]

        f_low = fft2(img_low.numpy())
        f_high = fft2(img_high.numpy())
        f_mix = f_low*self.mask+f_high*(1-self.mask)
        img_mix = np.real(ifft2(f_mix))
        img_mix = np.clip(img_mix, 0, 1)

        return torch.tensor(img_mix, dtype=torch.float), label_low, label_high


class EinMonJob(BaseJob):

    def __init__(self, store_dir, datasets_dir, device=DEVICE,
                 batch_size=BATCH_SIZE, worker_num=WORKER_NUM, **kwargs):
        if store_dir is None:
            super(EinMonJob, self).__init__(**kwargs)
        else:
            super(EinMonJob, self).__init__(os.path.join(store_dir, 'em-results'), **kwargs)
        self.datasets_dir = datasets_dir
        self.device = 'cuda' if device=='cuda' and torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.worker_num = worker_num

    def prepare_dataset(self, task, alpha, seed, grayscale=False):
        r"""Prepares the Einstein-Monroe dataset.

        Existing datasets that mix CIFAR images with different mixing
        frequencies are loaded.

        Args
        ----
        task: str
            The name of the dataset, only supports ``'CIFAR10'`` and
            ``'CIFAR100'`` for now.
        alpha: int
            The normalized mixing frequency, an integeger from 0 to 100.
        grayscale: bool
            Whether use grayscale CIFAR images.

        Returns
        -------
        dataset: Dataset
            The Einstein-Monroe dataset. Each item is a tuple
            `(image, label_low, label_high)`, with class labels for
            low-frequency and high-frequency component.

        """
        dataset = EinMonDataset(
            prepare_datasets(task, self.datasets_dir, grayscale=grayscale),
            alpha=alpha, seed=seed,
            )
        return dataset

    def evaluate(self, model, dataset):
        r"""Evaluates model.

        Args
        ----
        model: nn.Module
            The pytorch model.
        dataset: Dataset
            The dataset of Einstein-Monroe experiment.

        Returns
        -------
        loss_low, acc_low, loss_high, acc_high: float
            Cross entropy loss and accuracy for low and high frequency
            component respectively.

        """
        model.eval().to(self.device)
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.worker_num
            )
        loss_low, count_low = 0., 0.
        loss_high, count_high = 0., 0.
        for images, labels_low, labels_high in loader:
            images = images.to(self.device)
            labels_low, labels_high = labels_low.to(self.device), labels_high.to(self.device)
            with torch.no_grad():
                logits = model(images)
            _, predicts = logits.max(dim=1)

            loss_low += criterion(logits, labels_low).item()
            count_low += (predicts==labels_low).to(torch.float).sum().item()
            loss_high += criterion(logits, labels_high).item()
            count_high += (predicts==labels_high).to(torch.float).sum().item()
        loss_low, acc_low = loss_low/len(dataset), count_low/len(dataset)
        loss_high, acc_high = loss_high/len(dataset), count_high/len(dataset)
        return loss_low, acc_low, loss_high, acc_high

    def get_config(self, arg_strs):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model_pth')
        parser.add_argument('--seed', default=0, type=int,
                            help="random seed")
        parser.add_argument('--alpha', default=5, type=int,
                            help="mixing ratio, an integer from 0 to 100")

        args = parser.parse_args(arg_strs)

        assert args.model_pth is not None
        return {
            'model_pth': args.model_pth,
            'seed': get_seed(args.seed),
            'alpha': args.alpha,
            }

    def main(self, config, verbose=True):
        if verbose:
            print(config)

        # load model
        saved = torch.load(config['model_pth'])
        model = saved['model']

        # prepare Einstein-Monroe dataset
        dataset = self.prepare_dataset(
            saved['task'], config['alpha'], config['seed'],
            grayscale=saved['grayscale'] if 'grayscale' in saved else False,
            )

        # evaluate model
        loss_low, acc_low, loss_high, acc_high = self.evaluate(model, dataset)
        if verbose:
            print('low-frequency component\nloss: {:.4f}, acc: {:.2%}'.format(loss_low, acc_low))
            print('high-frequency component\nloss: {:.4f}, acc: {:.2%}'.format(loss_high, acc_high))

        result = {
            'loss_low': loss_low, 'acc_low': acc_low,
            'loss_high': loss_high, 'acc_high': acc_high,
            }
        preview = {}
        return result, preview

    def summarize(self, model_pths, alphas):
        r"""Summarizes a list of models.

        Args
        ----
        model_pths: list
            A list of model paths, each of which can be loaded by `torch.load`.
        alphas: list
            The normalized mixing frequencies.

        Returns
        -------
        accs_low, accs_high: dict
            A dictionary with alpha values as keys. Each item is a numpy array,
            containing testing accuracies of each model.

        """
        accs_low, accs_high = {}, {}
        for alpha in alphas:
            accs_low[alpha] = []
            accs_high[alpha] = []
            for model_pth in model_pths:
                cond = {
                    'model_pth': model_pth,
                    'alpha': alpha,
                    }
                for key, _ in self.conditioned(cond):
                    result = self.results[key]
                    accs_low[alpha].append(result['acc_low'])
                    accs_high[alpha].append(result['acc_high'])
            accs_low[alpha] = np.array(accs_low[alpha])
            accs_high[alpha] = np.array(accs_high[alpha])
        return accs_low, accs_high

    def plot_comparison(self, ax, groups, accs_low, accs_high, alphas):
        r"""Plots comparison of groups.

        Args
        ----
        ax: matplot axis
            The axis for plotting.
        groups: list
            Each item is a tuple of `(tag, model_pths, color)`. `tag` is the
            label for the group, `model_pths` is the list of model pths and
            `color` is a color tuple of shape `(3,)`.
        accs_low, accs_high: list
            Each item is a dictionary returned by `summarize`.
        alphas: list
            The normalized mixing frequencies.

        """
        bin_width = 0.8/len(groups)
        bars, legends = [], []
        for i, (tag, _, color) in enumerate(groups):
            acc_mean = np.array([np.mean(accs_low[i][alpha]) for alpha in alphas])*100
            acc_std = np.array([np.std(accs_low[i][alpha]) for alpha in alphas])*100
            h = ax.bar(
                np.arange(len(alphas))+(i-0.5*(len(groups)-1))*bin_width,
                acc_mean, width=bin_width, yerr=acc_std, zorder=2, facecolor=color,
                )
            h.errorbar.get_children()[0].set_edgecolor(np.array(color)*0.6)
            acc_mean = -np.array([np.mean(accs_high[i][alpha]) for alpha in alphas])*100
            acc_std = np.array([np.std(accs_high[i][alpha]) for alpha in alphas])*100
            h = ax.bar(
                np.arange(len(alphas))+(i-0.5*(len(groups)-1))*bin_width,
                acc_mean, width=bin_width, yerr=acc_std, zorder=2, facecolor=color,
                )
            h.errorbar.get_children()[0].set_edgecolor(np.array(color)*0.6)
            bars.append(h)
            legends.append(tag)
        ax.legend(bars, legends)
        ax.set_xlabel('normalized cutoff frequency')
        xticks = np.arange(len(alphas), step=2)
        ax.set_xticks(xticks)
        ax.set_xticklabels(['{:.1f}'.format(alphas[xtick]) for xtick in xticks], rotation=90)
        ax.set_ylabel(r'high-freq $\longleftrightarrow$ low-freq')
        ax.set_ylim([-100, 100])
        ax.set_yticks([-100, -50, 0, 50, 100])
        ax.set_yticklabels(['100%', '50%', '0%', '50%', '100%'])
        ax.grid(axis='y')


if __name__=='__main__':
    parser = job_parser()
    parser.add_argument('--store_dir', default='store')
    parser.add_argument('--datasets_dir', default='vision_datasets')
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
    parser.add_argument('--worker_num', default=WORKER_NUM, type=int)
    parser.add_argument('--max_seed', default=4, type=int)
    args = parser.parse_args()

    if args.spec_pth is None:
        export_dir = os.path.join(args.store_dir, 'models', 'exported')
        assert os.path.exists(export_dir), "directory of exported models not found"
        search_spec = {
            'model_pth': [os.path.join(export_dir, f) for f in os.listdir(export_dir) if f.endswith('.pt')],
            'seed': list(range(args.max_seed)),
            'alpha': ALPHAS,
            }
    else:
        with open(args.spec_pth, 'rb') as f:
            search_spec = pickle.load(f)

    job = EinMonJob(
        args.store_dir, args.datasets_dir, args.device, args.batch_size, args.worker_num
        )
    job.random_search(search_spec, args.process_num, args.max_wait, args.tolerance)
