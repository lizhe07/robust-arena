# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:29:37 2020

@author: Zhe
"""

import os, argparse, time, torch, torchvision
import numpy as np
import eagerpy as ep
import foolbox as fb
from foolbox import PyTorchModel
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

from jarvis import BaseJob
from jarvis.vision import prepare_datasets, evaluate
from jarvis.utils import time_str

DEVICE = 'cuda'
BATCH_SIZE = 64
WORKER_NUM = 4

ATTACKS = {
    'L2': {
        'PGD': fb.attacks.L2ProjectedGradientDescentAttack(),
        'BI': fb.attacks.L2BasicIterativeAttack(),
        'DF': fb.attacks.L2DeepFoolAttack(),
        'BB': fb.attacks.L2BrendelBethgeAttack(),
        },
    'Linf': {
        'PGD': fb.attacks.LinfProjectedGradientDescentAttack(),
        'BI': fb.attacks.LinfBasicIterativeAttack(),
        'DF': fb.attacks.LinfDeepFoolAttack(),
        'BB': fb.attacks.LinfinityBrendelBethgeAttack(),
        },
    }


class AttackJob(BaseJob):

    def __init__(self, save_dir, benchmarks_dir):
        super(AttackJob, self).__init__(save_dir)
        self.benchmarks_dir = benchmarks_dir

    def get_work_config(self, arg_strs):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model_pth')
        parser.add_argument('--norm', default='Linf', choices=['L2', 'Linf'])
        parser.add_argument('--attack_tags', default=['PGD', 'BI', 'DF'], nargs='+',
                            choices=['PGD', 'BI', 'DF', 'BB'])
        parser.add_argument('--success_rate', default=0.999, type=float)
        parser.add_argument('--eps_step', default=1e-3, type=float)

        args = parser.parse_args(arg_strs)

        attack_config = {
            'model_pth': args.model_pth,
            'norm': args.norm,
            'attack_tags': args.attack_tags,
            'success_rate': args.success_rate,
            'eps_step': args.eps_step,
            }
        return attack_config

    def main(self, attack_config):
        saved = torch.load(attack_config['model_pth'])
        model = saved['model']
        task = saved['config']['model_config']['task']
        attacks = [ATTACKS[attack_config['norm']][a_tag] \
                   for a_tag in attack_config['attack_tags']]

        advs, successes, dist = test_adversarial_attack(
            model, task, attacks, self.benchmarks_dir
            )
        output = {
            'advs': advs.numpy(),
            'successes': successes.numpy(),
            'dist': dist,
            }
        preview = {
            'dist': dist,
            }
        return output, preview

class CorruptionJob(BaseJob):

    def __init__(self, save_dir, benchmarks_dir, device=DEVICE,
                 batch_size=BATCH_SIZE, worker_num=WORKER_NUM):
        super(BaseJob, self).__init__(save_dir)
        self.benchmarks_dir = benchmarks_dir
        self.device = device
        self.batch_size = batch_size
        self.worker_num = worker_num

    def get_work_config(self, arg_strs):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model_pth')
        parser.add_argument('--corruption', choices=[
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'frog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
            ])
        parser.add_argument('severity', default=5, type=int)

        args = parser.parse_args(arg_strs)

        corrupt_config = {
            'model_pth': args.model_pth,
            'corruption': args.corruption,
            'severity': args.severity,
            }
        return corrupt_config

    def main(self, corrupt_config):
        saved = torch.load(corrupt_config['model_pth'])
        model = saved['model']
        task = saved['config']['model_config']['task']

        if task=='CIFAR10':
            images = np.load(os.path.join(
                self.benchmarks_dir, 'CIFAR-10-C',
                '{}.npy'.format(corrupt_config['corruption']),
                ))
            s = corrupt_config['severity']
            images = images[(s-1)*10000:s*10000].transpose(0, 3, 1, 2)
            labels = np.load(os.path.join(
                self.benchmarks_dir, 'CIFAR-10-C', 'labels.npy'
                ))[:10000]

            dataset = TensorDataset(
                torch.tensor(images, dtype=torch.float),
                torch.tensor(labels, dtype=torch.long),
                )

        loss, acc = evaluate(model, dataset,
                             self.device, self.batch_size, self.worker_num)
        output = {'loss': loss, 'acc': acc}
        preview = {'loss': loss, 'acc': acc}
        return output, preview


def distorted_images(i_original, d_type, d_val):
    r"""Returns distorted images.

    Args
    ----
    i_original: (N, C, H, W), tensor
        A tensor of the original images, with values in [0, 1].
    d_type: str
        The distortion type.
    d_val: float
        The value of distortion level.

    Returns
    -------
    i_distorted: (N, C, H, W), tensor
        A tensor of the distorted images, with the same shape and data type as
        `i_original`.

    """
    # min and max values for each image in the batch
    i_mins = [i.min() for i in i_original]
    i_maxs = [i.max() for i in i_original]

    if d_type=='Gaussian':
        i_distorted = i_original+torch.randn_like(i_original)*d_val
    if d_type=='Uniform':
        i_distorted = i_original+(torch.rand_like(i_original)-0.5)*2*d_val
    if d_type=='SaltPepper':
        saltpepper = (torch.rand_like(i_original)<d_val).to(i_original)\
            *(2*torch.randint_like(i_original, 2)-1)
        i_distorted = i_original+saltpepper
    if d_type=='LowPass':
        i_lowpassed = gaussian_filter(
            i_original.cpu().numpy(), [0, 0, d_val, d_val],
            mode='constant', cval=0.5
            )
        i_distorted = torch.tensor(i_lowpassed).to(i_original)
    if d_type=='HighPass':
        i_lowpassed = gaussian_filter(
            i_original.cpu().numpy(), [0, 0, d_val, d_val],
            mode='constant', cval=0.5
            )
        i_highpassed = i_original-torch.tensor(i_lowpassed, dtype=torch.float)
        i_diff = 0.5-i_highpassed.mean(dim=(1, 2, 3), keepdim=True)
        i_distorted = i_highpassed+i_diff

    i_distorted = torch.stack([
        i.clamp(i_min, i_max) for i, i_min, i_max in zip(i_distorted, i_mins, i_maxs)
        ])
    return i_distorted


def get_dataset(task, benchmarks_dir):
    if task=='CIFAR10':
        dataset = torchvision.datasets.CIFAR10(
            benchmarks_dir, train=False, transform=transforms.ToTensor(),
            )
    if task=='CIFAR100':
        dataset = torchvision.datasets.CIFAR100(
            benchmarks_dir, train=False, transform=transforms.ToTensor(),
            )
    if task=='16ImageNet':
        dataset = torchvision.datasets.ImageFolder(
            f'{benchmarks_dir}/16imagenet_split/test',
            transform=transforms.ToTensor(),
            )
    return dataset


def random_distortion_dataset(task, d_type, d_val, benchmarks_dir, cache_dir,
                              batch_size=BATCH_SIZE, worker_num=WORKER_NUM):
    dataset_cache = os.path.join(cache_dir, f'{task}_{d_type}_{d_val}.pt')
    if os.path.exists(dataset_cache):
        dataset = torch.load(dataset_cache)
    else:
        dataset = get_dataset(task, benchmarks_dir)

        images, labels = [], []
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=worker_num)
        for _images, _labels in loader:
            _images = distorted_images(_images, d_type, d_val)
            images.append(_images)
            labels.append(_labels)
        dataset = TensorDataset(torch.cat(images), torch.cat(labels))
        torch.save(dataset, dataset_cache)
    return dataset


def evaluate(model, dataset, device=DEVICE, batch_size=BATCH_SIZE, worker_num=WORKER_NUM):
    if device=='cuda' and not torch.cuda.is_available():
        device = 'cpu'
    model.eval().to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=worker_num)
    loss, count = 0., 0.
    for images, labels in loader:
        with torch.no_grad():
            logits = model(images.to(device))
            loss += criterion(logits, labels.to(device)).item()
            _, predicts = logits.max(dim=1)
            count += (predicts.cpu()==labels).to(torch.float).sum().item()
    loss = loss/len(dataset)
    acc = count/len(dataset)
    return loss, acc


def dataset_forwardpass(model, dataset, device=DEVICE,
                        batch_size=BATCH_SIZE):
    if device=='cuda' and not torch.cuda.is_available():
        device = 'cpu'
    model.eval().to(device)

    loader = DataLoader(dataset, batch_size=batch_size)
    images, labels, predicts = [], [], []
    for _images, _labels in loader:
        with torch.no_grad():
            _, _predicts = model(_images.to(device)).max(dim=1)
        images.append(_images)
        labels.append(_labels)
        predicts.append(_predicts.cpu())
    images = torch.cat(images)
    labels = torch.cat(labels)
    predicts = torch.cat(predicts)
    return images, labels, predicts


def attack_model(model, dataset, attack, eps,
                 device=DEVICE, batch_size=BATCH_SIZE, worker_num=WORKER_NUM):
    if device=='cuda' and not torch.cuda.is_available():
        device = 'cpu'
    model.eval().to(device)
    fmodel = PyTorchModel(model, bounds=(0, 1))

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=worker_num)
    advs, successes = [], []
    for images, labels in loader:
        _images, _labels = ep.astensors(images.to(device), labels.to(device))
        _, _advs, _successes = attack(fmodel, _images, _labels, epsilons=eps)
        advs.append(_advs.raw.cpu())
        successes.append(_successes.raw.cpu())
    advs = torch.cat(advs)
    successes = torch.cat(successes)
    return advs, successes


def test_random_distortions(model, task, d_type, d_val, benchmarks_dir, cache_dir):
    dataset = random_distortion_dataset(task, d_type, d_val, benchmarks_dir, cache_dir)
    loss, acc = evaluate(model, dataset)
    return loss, acc


def test_adversarial_attack(model, task, attacks, benchmarks_dir,
                            success_rate=0.999, eps_step=1e-3):
    lp = None
    for attack in attacks:
        if lp is None:
            lp = attack.distance
        else:
            assert lp==attack.distance

    dataset = get_dataset(task, benchmarks_dir)
    images, labels, predicts = dataset_forwardpass(model, dataset)

    advs = images.clone()
    successes = labels!=predicts

    eps = 0
    while successes.to(torch.float).mean()<success_rate:
        eps += eps_step
        print('attacking with eps {:.3f}...'.format(eps))
        tic = time.time()
        for attack in attacks:
            idxs, = (successes!=True).numpy().nonzero()
            _dataset = TensorDataset(images[idxs], labels[idxs])
            _advs, _successes = attack_model(model, _dataset, attack, eps)
            advs[idxs], successes[idxs] = _advs, _successes
        toc = time.time()
        print('success rate {:7.2%} ({})'.format(
            successes.to(torch.float).mean(),
            time_str(toc-tic),
            ))
    dist = lp(images, advs).mean().item()
    return advs, successes, dist
