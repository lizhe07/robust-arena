import os, argparse, torch, time, random, json
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms.functional import rgb_to_grayscale

import jarvis
from jarvis import BaseJob
from jarvis.vision import evaluate, IMAGENET_TEST
from jarvis.utils import job_parser

from . import DEVICE, BATCH_SIZE

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    ]
SEVERITIES = [1, 2, 3, 4, 5]


class CorruptionJob(BaseJob):
    r"""Tests model robustness on common corruptions.

    Only 'CIFAR10' and 'CIFAR100' are implemented now.

    Args
    ----
    store_dir: str
        The directory for storing results. When `store_dir` is ``None``, no
        external storage is used.
    datasets_dir: str
        The directory for vision datasets, must have 'CIFAR-10-C' and
        'CIFAR-100-C' as subdirectories.
    device: str
        The device for computation.
    batch_size: int
        The batch size used during testing.
    worker_num: int
        The worker number for data loader.

    """

    def __init__(self,
        store_dir, datasets_dir,
        device=DEVICE, batch_size=BATCH_SIZE,
        **kwargs,
    ):
        super(CorruptionJob, self).__init__(store_dir=store_dir, **kwargs)
        self.datasets_dir = datasets_dir
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size

    def strs2config(self, arg_strs):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-path')
        parser.add_argument('--corruption', choices=CORRUPTIONS)
        parser.add_argument('--severity', default=5, type=int)
        args, _ = parser.parse_known_args(arg_strs)

        assert args.model_path is not None
        assert args.severity in SEVERITIES
        return {
            'model_path': args.model_path,
            'corruption': args.corruption,
            'severity': args.severity,
        }

    def prepare_dataset(self, task, corruption, severity):
        r"""Returns common corruption testing set.

        Args
        ----
        task: str
            The task name.
        corruption: str
            The corruption name, can only be one of `CORRUPTIONS`.
        severity: int
            The severity level, can only be 1 to 5.

        Returns
        -------
        dataset: TensorDataset
            The dataset containing corrupted images and class labels.

        """
        if severity==0:
            dataset = jarvis.vision.prepare_datasets(task, self.datasets_dir)
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

        # evaluate on common corruption dataset
        dataset = self.prepare_dataset(
            saved['task'], config['corruption'], config['severity'],
        )
        loss, acc = evaluate(
            model, dataset, self.batch_size, self.device, verbose,
        )

        ckpt = {'loss': loss, 'acc': acc}
        preview = {'loss': loss, 'acc': acc}
        return ckpt, preview

    def summarize(self, model_pths, severity=5):
        r"""Summarizes a list of models.

        Args
        ----
        model_pths: list
            A list of model paths, each of which can be loaded by `torch.load`.
        severity: int
            The severity level.

        Returns
        -------
        accs: dict
            A dictionary with corruption names as keys. Each item is a numpy
            array, containing testing accuracies of each model.

        """
        accs = {}
        for corruption in CORRUPTIONS:
            accs[corruption] = []
            for model_pth in model_pths:
                config = {
                    'model_pth': model_pth,
                    'corruption': corruption,
                    'severity': severity,
                    }
                key = self.configs.get_key(config)
                if key is not None and self.is_completed(key):
                    accs[corruption].append(self.results[key]['acc'])
            accs[corruption] = np.array(accs[corruption])
        accs['clean'] = []
        for model_pth in model_pths:
            config = {
                'model_pth': model_pth,
                'corruption': None,
                'severity': 0,
                }
            result, _ = self.process(config, verbose=False)
            accs['clean'].append(result['acc'])
        accs['clean'] = np.array(accs['clean'])
        return accs

    def plot_comparison(self, ax, groups, accs):
        r"""Plots comparison of groups.

        Args
        ----
        ax: matplot axis
            The axis for plotting.
        groups: list
            Each item is a tuple of `(tag, model_pths, color)`. `tag` is the
            label for the group, `model_pths` is the list of model pths and
            `color` is a color tuple of shape `(3,)`.
        accs: list
            Each item is a dictionary returned by `summarize`.

        """
        bin_width = 0.8/len(groups)
        bars, legends = [], []
        _CORRUPTIONS = CORRUPTIONS+['clean']
        xticks = np.array(list(range(len(CORRUPTIONS)))+[len(CORRUPTIONS)+1])
        for i, (tag, _, color) in enumerate(groups):
            acc_mean = np.array([np.mean(accs[i][c]) for c in _CORRUPTIONS])*100
            acc_std = np.array([np.std(accs[i][c]) for c in _CORRUPTIONS])*100
            h = ax.bar(
                xticks+(i-0.5*(len(groups)-1))*bin_width,
                acc_mean, width=bin_width, yerr=acc_std, zorder=2, facecolor=color,
                )
            h.errorbar.get_children()[0].set_edgecolor(np.array(color)*0.6)
            bars.append(h)
            legends.append(tag)
        ax.legend(bars, legends)
        ax.set_xticks(xticks)
        ax.set_xticklabels(_CORRUPTIONS, rotation=90)
        ax.set_ylabel('accuracy (%)')
        ax.set_ylim([0, 100])
        ax.grid(axis='y')


if __name__=='__main__':
    parser = job_parser()
    parser.add_argument('--store-dir', default='store')
    parser.add_argument('--datasets-dir', default='datasets')
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int)
    parser.add_argument('--spec-path', default='c-tests/spec.json')
    args, _ = parser.parse_known_args()

    time.sleep(random.random()*args.max_wait)
    job = CorruptionJob(
        f'{args.store_dir}/c-tests', args.datasets_dir, args.device, args.batch_size,
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
    if search_spec.get('corruption') is None:
        search_spec['corruption'] = CORRUPTIONS
    if search_spec.get('severity') is None:
        search_spec['severity'] = SEVERITIES
    job.grid_search(
        search_spec, num_works=args.num_works, patience=args.patience,
    )
