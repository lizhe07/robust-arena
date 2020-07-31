# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:29:37 2020

@author: Zhe
"""

import os, torch, torchvision
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

DEVICE = 'cuda'
BATCH_SIZE = 64
WORKER_NUM = 4


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


def random_distortion_dataset(task, d_type, d_val, benchmarks_dir,
                              batch_size=BATCH_SIZE, worker_num=WORKER_NUM):
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

    images, labels = [], []
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=worker_num)
    for _images, _labels in loader:
        _images = distorted_images(_images, d_type, d_val)
        images.append(_images)
        labels.append(_labels)
    dataset = TensorDataset(torch.cat(images), torch.cat(labels))
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


def test_random_distortions(task, model, d_type, d_val,
                            cache_dir, benchmarks_dir=None):
    dataset_cache = os.path.join(cache_dir, f'{task}_{d_type}_{d_val}.pt')
    if os.path.exists(dataset_cache):
        dataset = torch.load(dataset_cache)
    else:
        dataset = random_distortion_dataset(task, d_type, d_val, benchmarks_dir)
        torch.save(dataset, dataset_cache)

    loss, acc = evaluate(model, dataset)
    return loss, acc
