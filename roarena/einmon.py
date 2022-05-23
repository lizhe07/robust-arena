import os, argparse, pickle, torch, time, random, json
import numpy as np
from scipy.fft import fft2, ifft2

from jarvis import BaseJob
from jarvis.vision import prepare_datasets
from jarvis.utils import get_seed, set_seed, job_parser

from . import DEVICE, BATCH_SIZE, NUM_WORKERS
ALPHAS = [0, 3, 4, 5, 6, 8, 9, 11, 13, 15, 17, 19, 22, 27, 33, 42, 57, 100]
MAX_SEED = 6


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


class EinMonJob(BaseJob):

    def __init__(self,
        store_dir, datasets_dir,
        device=DEVICE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        **kwargs,
    ):
        super(EinMonJob, self).__init__(store_dir=store_dir, **kwargs)
        self.datasets_dir = datasets_dir
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_dataset(self, task, alpha, seed):
        r"""Prepares the Einstein-Monroe dataset."""
        dataset = EinMonDataset(
            prepare_datasets(task, self.datasets_dir), alpha=alpha, seed=seed,
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
            dataset, batch_size=self.batch_size, num_workers=self.num_workers
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

    def strs2config(self, arg_strs):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-path')
        parser.add_argument('--seed', default=0, type=int,
                            help="random seed")
        parser.add_argument('--alpha', default=5, type=int,
                            help="mixing ratio, an integer from 0 to 100")
        args = parser.parse_args(arg_strs)

        assert args.model_path is not None
        return {
            'model_path': args.model_path,
            'seed': get_seed(args.seed),
            'alpha': args.alpha,
        }

    def main(self, config, epoch=1, verbose=1):
        if verbose>0:
            print(config)

        # load model
        saved = torch.load(config['model_path'])
        model = saved['model']
        if torch.cuda.device_count()>1:
            if verbose>0:
                print("Using {} GPUs".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)

        # prepare Einstein-Monroe dataset
        dataset = self.prepare_dataset(
            saved['task'], config['alpha'], config['seed'],
        )

        # evaluate model
        loss_low, acc_low, loss_high, acc_high = self.evaluate(model, dataset)
        if verbose>0:
            print('low-frequency component\nloss: {:.4f}, acc: {:.2%}'.format(loss_low, acc_low))
            print('high-frequency component\nloss: {:.4f}, acc: {:.2%}'.format(loss_high, acc_high))

        ckpt = {
            'loss_low': loss_low, 'acc_low': acc_low,
            'loss_high': loss_high, 'acc_high': acc_high,
        }
        preview = {}
        return ckpt, preview

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
    parser.add_argument('--store-dir', default='store')
    parser.add_argument('--datasets-dir', default='datasets')
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int)
    parser.add_argument('--num-workers', default=NUM_WORKERS, type=int)
    parser.add_argument('--spec-path', default='e-tests/spec.json')
    args, _ = parser.parse_known_args()

    time.sleep(random.random()*args.max_wait)
    job = EinMonJob(
        f'{args.store_dir}/e-tests', args.datasets_dir,
        args.device, args.batch_size, args.num_workers,
    )

    try:
        with open(f'{args.store_dir}/{args.spec.path}', 'r') as f:
            search_spec = json.load(f)
    except:
        search_spec = {}
    if search_spec.get('model-path') is None:
        search_spec['model-path'] = [
            f'{args.store_dir}/exported/{f}' for f in os.listdir(f'{args.store_dir}/exported') if f.endswith('.pt')
        ]
    if search_spec.get('seed') is None:
        search_spec['seed'] = list(range(MAX_SEED))
    if search_spec.get('alpha') is None:
        search_spec['alpha'] = ALPHAS
    job.grid_search(
        search_spec, num_works=args.num_works, patience=args.patience,
    )
