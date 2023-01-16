import os, yaml, torch
import numpy as np
from scipy.fft import fft2, ifft2

from jarvis.utils import get_seed, set_seed
from jarvis.config import Config, from_cli
from jarvis.manager import Manager
from jarvis.vision import prepare_datasets

from . import DEVICE, BATCH_SIZE, NUM_WORKERS
ALPHAS = [0, 3, 4, 5, 6, 8, 9, 11, 13, 15, 17, 19, 22, 27, 33, 42, 57, 100]
MAX_SEED = 6

cli_args = Config({
    'store_dir': 'store',
    'datasets_dir': 'datasets',
    'model_path': [],
    'alpha': ALPHAS,
    'seed': list(range(MAX_SEED)),
})


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
        for c in range(len(dataset.class_names)):
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


class EinMonManager(Manager):

    def __init__(self,
        store_dir: str, datasets_dir: str,
        device: str = DEVICE,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        **kwargs,
    ):
        super(EinMonManager, self).__init__(f'{store_dir}/e-results', **kwargs)
        self.datasets_dir = datasets_dir
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_config(self, config):
        config = super(EinMonManager, self).get_config(config)
        assert config.seed is not None
        config.seed = get_seed(config.seed)
        return config

    def prepare_dataset(self, task, alpha, seed):
        r"""Prepares the Einstein-Monroe dataset."""
        dataset = EinMonDataset(
            prepare_datasets(task, self.datasets_dir), alpha=alpha, seed=seed,
        )
        return dataset

    def setup(self, config):
        super(EinMonManager, self).setup(config)

        # load model
        saved = torch.load(config.model_path)
        self.model = saved['model']
        if torch.cuda.device_count()>1:
            if self.verbose>0:
                print("Using {} GPUs".format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)

        # prepare Einstein-Monroe dataset
        self.dataset = self.prepare_dataset(
            saved['task'], config.alpha, config.seed,
        )

    def eval(self):
        self.model.eval().to(self.device)
        criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(self.device)

        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
        loss_low, count_low = 0., 0.
        loss_high, count_high = 0., 0.
        for images, labels_low, labels_high in loader:
            images = images.to(self.device)
            labels_low, labels_high = labels_low.to(self.device), labels_high.to(self.device)
            with torch.no_grad():
                logits = self.model(images)
            _, predicts = logits.max(dim=1)

            loss_low += criterion(logits, labels_low).item()
            count_low += (predicts==labels_low).to(torch.float).sum().item()
            loss_high += criterion(logits, labels_high).item()
            count_high += (predicts==labels_high).to(torch.float).sum().item()
        num_samples = len(self.dataset)
        loss_low, acc_low = loss_low/num_samples, count_low/num_samples
        loss_high, acc_high = loss_high/num_samples, count_high/num_samples

        if self.verbose>0:
            print('low-frequency component\nloss: {:.4f}, acc: {:.2%}'.format(loss_low, acc_low))
            print('high-frequency component\nloss: {:.4f}, acc: {:.2%}'.format(loss_high, acc_high))

        self.ckpt = {
            'loss_low': loss_low, 'acc_low': acc_low,
            'loss_high': loss_high, 'acc_high': acc_high,
        }
        self.preview = self.ckpt

    def summarize(self, model_paths, alphas):
        r"""Summarizes a list of models.

        Args
        ----
        model_paths:
            A list of model paths, each of which can be loaded by `torch.load`.
        alphas:
            The normalized mixing frequencies.

        Returns
        -------
        accs_low, accs_high: dict
            A dictionary with alpha values as keys. Each item is a numpy array,
            containing testing accuracies of each model.

        """
        accs_low_mean, accs_low_std = [], []
        accs_high_mean, accs_high_std = [], []
        for model_path in model_paths:
            _accs_low_mean, _accs_low_std = [], []
            _accs_high_mean, _accs_high_std = [], []
            for alpha in alphas:
                accs_low, accs_high = [], []
                cond = {
                    'model_path': model_path, 'alpha': alpha,
                }
                for key in self.completed(cond=cond):
                    ckpt = self.ckpts[key]
                    accs_low.append(ckpt['acc_low'])
                    accs_high.append(ckpt['acc_high'])
                _accs_low_mean.append(np.mean(accs_low))
                _accs_low_std.append(np.std(accs_low))
                _accs_high_mean.append(np.mean(accs_high))
                _accs_high_std.append(np.std(accs_high))
            accs_low_mean.append(np.array(_accs_low_mean))
            accs_low_std.append(np.array(_accs_low_std))
            accs_high_mean.append(np.array(_accs_high_mean))
            accs_high_std.append(np.array(_accs_high_std))
        return accs_low_mean, accs_low_std, accs_high_mean, accs_high_std

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
    cli_args.update(from_cli())
    store_dir = cli_args.pop('store_dir')
    datasets_dir = cli_args.pop('datasets_dir')
    manager = EinMonManager(store_dir, datasets_dir)

    choices = {}
    choices['model_path'] = cli_args.pop('model_path')
    choices['alpha'] = cli_args.pop('alpha')
    choices['seed'] = cli_args.pop('seed')
    if isinstance(choices['model_path'], str):
        with open(choices['model_path']) as f:
            choices['model_path'] = yaml.safe_load(f)
    elif len(choices['model_path'])==0:
        choices['model_path'] = [
            f'{store_dir}/exported/{f}' for f in os.listdir(f'{store_dir}/exported') if f.endswith('.pt')
        ]

    manager.sweep(choices, num_epochs=0, **cli_args)
