import os, yaml, torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms.functional import rgb_to_grayscale

from jarvis.config import Config, from_cli
from jarvis.manager import Manager
from jarvis.vision import evaluate, prepare_datasets, IMAGENET_TEST

from . import DEVICE, BATCH_SIZE

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]
SEVERITIES = [1, 2, 3, 4, 5]

cli_args = Config({
    'store_dir': 'store',
    'datasets_dir': 'datasets',
    'model_path': [],
    'corruption': CORRUPTIONS,
    'severity': SEVERITIES,
})


class CorruptionManager(Manager):
    r"""Tests model robustness on common corruptions.

    Only 'CIFAR10' and 'CIFAR100' are implemented now.

    Args
    ----
    store_dir:
        The directory for storing results. When `store_dir` is ``None``, no
        external storage is used.
    datasets_dir:
        The directory for vision datasets, must have 'CIFAR-10-C' and
        'CIFAR-100-C' as subdirectories.
    device:
        The device for computation.
    batch_size:
        The batch size used during testing.

    """

    def __init__(self,
        store_dir: str, datasets_dir: str,
        device: str = DEVICE, batch_size: int = BATCH_SIZE,
        **kwargs,
    ):
        super(CorruptionManager, self).__init__(f'{store_dir}/c-results', **kwargs)
        self.datasets_dir = datasets_dir
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size

    def get_config(self, config):
        config = super(CorruptionManager, self).get_config(config)
        if config.severity==0:
            config.corruption = None
        return config

    def prepare_dataset(self, task: str, corruption: str, severity: int):
        r"""Returns common corruption testing set.

        Args
        ----
        task:
            The task name.
        corruption:
            The corruption name, can only be one of `CORRUPTIONS`.
        severity:
            The severity level, can only be 1 to 5.

        Returns
        -------
        dataset:
            The dataset containing corrupted images and class labels.

        """
        if severity==0:
            dataset = prepare_datasets(task, self.datasets_dir)
            return dataset

        if task.endswith('-Gray'):
            task = task[:-5]
            to_grayscale = True
        else:
            to_grayscale = False
        if task in ['CIFAR10', 'CIFAR100']:
            if task=='CIFAR10':
                npy_dir = os.path.join(self.datasets_dir, 'CIFAR-10-C')
            if task=='CIFAR100':
                npy_dir = os.path.join(self.datasets_dir, 'CIFAR-100-C')

            images = np.load(os.path.join(npy_dir, f'{corruption}.npy'))/255.
            images = torch.tensor(
                images[(severity-1)*10000:severity*10000], dtype=torch.float
            ).permute(0, 3, 1, 2)
            if to_grayscale:
                images = rgb_to_grayscale(images)
            labels = torch.tensor(
                np.load(os.path.join(npy_dir, 'labels.npy'))[:10000], dtype=torch.long
            )
            dataset = torch.utils.data.TensorDataset(images, labels)
        if task=='ImageNet':
            t_test = IMAGENET_TEST
            if to_grayscale:
                t_test = transforms.Compose([t_test, transforms.Grayscale()])
            dataset = ImageFolder(
                os.path.join(self.datasets_dir, 'ImageNet-C', corruption, str(severity)),
                transform=t_test,
            )
        if task=='TinyImageNet':
            t_test = transforms.ToTensor()
            if to_grayscale:
                t_test = transforms.Compose([t_test, transforms.Grayscale()])
            dataset = ImageFolder(
                os.path.join(self.datasets_dir, 'Tiny-ImageNet-C', corruption, str(severity)),
                transform=t_test,
            )
        return dataset

    def setup(self, config):
        super(CorruptionManager, self).setup(config)

        # load model
        saved = torch.load(config.model_path)
        self.model = saved['model']
        if torch.cuda.device_count()>1:
            if self.verbose>0:
                print("Using {} GPUs".format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)

        # prepare common corruption dataset
        self.dataset = self.prepare_dataset(
            saved['task'], self.config.corruption, self.config.severity,
        )

    def eval(self):
        loss, acc = evaluate(
            self.model, self.dataset, self.batch_size,
            device=self.device, verbose=self.verbose,
        )

        self.ckpt = {'loss': loss, 'acc': acc}
        self.preview = {'loss': loss, 'acc': acc}

    def summarize(self, model_paths: list[str], severity: int = 5) -> list[dict[str, list[float]]]:
        r"""Summarizes a list of models.

        Args
        ----
        model_paths:
            A list of model paths, each of which can be loaded by `torch.load`.
        severity:
            The severity level.

        Returns
        -------
        accs:
            A dictionary with corruption names as keys. Each item is a numpy
            array, containing testing accuracies of each model.

        """
        accs = []
        for model_path in model_paths:
            print(f"Fetching corruption robustness results for {model_path}...")
            _accs = {}
            for corruption in CORRUPTIONS:
                config = self.get_config({
                    'model_path': model_path,
                    'corruption': corruption,
                    'severity': severity,
                })
                key = self.configs.get_key(config)
                _accs[corruption] = self.previews[key]['acc']
            _accs['clean'] = torch.load(model_path)['acc']
            accs.append(_accs)
        return accs

    def plot_full(self,
        accs: list[dict[str, list[float]]],
        ax=None, bar_width=0.8, colors=None, legends=None,
    ):
        r"""Plots full comparison of groups of models.

        Classification accuracy for each type of corruption is compared. If one
        group contains more than one model, standard deviation across models are
        shown as errorbars.

        Args
        ----
        accs:
            Model(s) performance for each corruption at a severity level.
        ax:
            Matplotlib axis for plotting.
        bar_width:
            Width parameter for bar plots, value in (0, 1).
        colors:
            A list of RGB values, specifying colors for each group.
        legends:
            Legends for each group.

        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3))
        num_groups = len(accs)
        bar_width /= num_groups
        _CORRUPTIONS = CORRUPTIONS+['clean']
        xticks = np.arange(len(_CORRUPTIONS))
        if colors is None:
            colors = [matplotlib.cm.get_cmap('tab10')(x)[:3] for x in np.linspace(0, 1, num_groups)]
        if legends is None:
            legends = [f'Group {i}' for i in range(num_groups)]
        bars = []
        for i in range(num_groups):
            color = colors[i]
            acc_mean = np.array([np.mean(accs[i][c]) for c in _CORRUPTIONS])*100
            acc_std = np.array([np.std(accs[i][c]) for c in _CORRUPTIONS])*100
            h = ax.bar(
                xticks+(i-0.5*(num_groups-1))*bar_width,
                acc_mean, width=bar_width, yerr=acc_std, zorder=2, facecolor=color,
            )
            h.errorbar.get_children()[0].set_edgecolor(np.array(color)*0.8+0.2)
            bars.append(h)
        ax.legend(bars, legends, fontsize=12)
        ylim = max(max(accs[i].values()) for i in range(num_groups))*100
        ylim = -(-ylim//10)*10
        ax.set_xticks(xticks)
        ax.set_xticklabels(_CORRUPTIONS, rotation=90)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim([0, ylim])
        ax.grid(axis='y')

    def plot_digest(self,
        accs: list[dict[str, list[float]]],
        ax=None, bar_width=0.8, colors=None, legends=None,
    ):
        r"""Plots digest comparison of groups of models.

        Classification accuracy for 'low', 'medium' and 'high' corruptions is
        compared. Standard deviation computed over in-category corruptions and
        in-group models are shown as error bars.

        Args
        ----
        accs:
            Model(s) performance for each corruption at a severity level.
        ax:
            Matplotlib axis for plotting.
        bar_width:
            Width parameter for bar plots, value in (0, 1).
        colors:
            A list of RGB values, specifying colors for each group.
        legends:
            Legends for each group.

        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        num_groups = len(accs)
        bar_width /= num_groups
        CORRUPTIONS = {
            'low': ['snow', 'frost', 'fog', 'brightness', 'contrast'],
            'medium': ['motion_blur', 'zoom_blur', 'defocus_blur', 'glass_blur', 'elastic_transform', 'jpeg_compression', 'pixelate'],
            'high': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
            'clean': ['clean'],
        }
        xticks = np.arange(4)
        if colors is None:
            colors = [matplotlib.cm.get_cmap('tab10')(x)[:3] for x in np.linspace(0, 1, num_groups)]
        if legends is None:
            legends = [f'Group {i}' for i in range(num_groups)]
        bars = []
        for i in range(num_groups):
            color = colors[i]
            acc_mean, acc_sem = np.zeros(4), np.zeros(4)
            for j, category in enumerate(['low', 'medium', 'high', 'clean']):
                _accs = []
                for c in CORRUPTIONS[category]:
                    if isinstance(accs[i][c], float):
                        _accs += [accs[i][c]]
                    else:
                        _accs += list(accs[i][c])
                acc_mean[j] = np.mean(_accs)*100.
                acc_sem[j] = np.std(_accs)*100./len(CORRUPTIONS[category])**0.5
            h = ax.bar(
                xticks+(i-0.5*(num_groups-1))*bar_width,
                acc_mean, width=bar_width, yerr=acc_sem, zorder=2, facecolor=color,
            )
            h.errorbar.get_children()[0].set_edgecolor(np.array(color)*0.8+0.2)
            bars.append(h)
        ax.legend(bars, legends, fontsize=12)
        ylim = max(max(accs[i].values()) for i in range(num_groups))*100
        ylim = -(-ylim//10)*10
        ax.set_xticks(xticks)
        ax.set_xticklabels(['Low', 'Medium', 'High', 'Clean'])
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim([0, ylim])
        ax.grid(axis='y')


if __name__=='__main__':
    cli_args.update(from_cli())
    store_dir = cli_args.pop('store_dir')
    datasets_dir = cli_args.pop('datasets_dir')
    manager = CorruptionManager(store_dir, datasets_dir)

    choices = {}
    choices['model_path'] = cli_args.pop('model_path')
    choices['corruption'] = cli_args.pop('corruption')
    choices['severity'] = cli_args.pop('severity')
    if isinstance(choices['model_path'], str):
        with open(choices['model_path']) as f:
            choices['model_path'] = yaml.safe_load(f)
    elif len(choices['model_path'])==0:
        choices['model_path'] = [
            f'{store_dir}/exported/{f}' for f in os.listdir(f'{store_dir}/exported') if f.endswith('.pt')
        ]

    manager.sweep(choices, num_epochs=0, **cli_args)
