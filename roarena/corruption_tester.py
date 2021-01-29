# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:39:55 2020

@author: Zhe
"""

import os, argparse, torch
import numpy as np

from jarvis import BaseJob
from jarvis.vision import evaluate

DEVICE = 'cuda'
BATCH_SIZE = 160
WORKER_NUM = 0

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    ]
SEVERITIES = [1, 2, 3, 4, 5]

parser = argparse.ArgumentParser()
parser.add_argument('--store_dir')
parser.add_arguemtn('--datasets_dir', default='vision_datasets')
parser.add_argument('--device', default=DEVICE)
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
parser.add_argument('--worker_num', default=WORKER_NUM, type=int)
parser.add_argument('--process_num', default=0, type=int)
parser.add_argument('--max_wait', default=1, type=float)
parser.add_argument('--tolerance', default=float('inf'), type=float)
args = parser.parse_args()


class CorruptionTest(BaseJob):

    def __init__(self, store_dir, datasets_dir, device=DEVICE,
                 batch_size=BATCH_SIZE, worker_num=WORKER_NUM):
        if store_dir is None:
            super(CorruptionTest, self).__init__()
        else:
            super(CorruptionTest, self).__init__(os.path.join(store_dir, 'corruption_tests'))
        self.device = device
        self.batch_size = batch_size
        self.worker_num = worker_num

    def get_config(self, arg_strs):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model_pth')
        parser.add_argument('--corruption', choices=CORRUPTIONS)
        parser.add_argument('--severity', default=5, type=int)

        args = parser.parse_args(arg_strs)
        assert args.model_pth is not None
        assert args.severity in SEVERITIES
        return {
            'model_pth': args.model_pth,
            'corruption': args.corruption,
            'severity': args.severity,
            }

    def prepare_dataset(self, task, grayscale, corruption, severity):
        if task=='CIFAR10':
            npy_dir = os.path.join(self.datasets_dir, 'CIFAR-10-C')
        if task=='CIFAR100':
            npy_dir = os.path.join(self.datasets_dir, 'CIFAR-100-C')

        images = np.load(os.path.join(npy_dir, f'{corruption}.npy'))/255.
        images = torch.tensor(
            images[(severity-1)*10000:severity*10000], dtype=torch.float
            ).permute(0, 3, 1, 2)
        labels = torch.tensor(
            np.load(os.path.join(npy_dir, 'labels.npy'))[:10000], dtype=torch.long
            )
        dataset = torch.utils.data.TensorDataset(images, labels)
        return dataset


    def main(self, config, verbose=True):
        if verbose:
            print(config)

        # load model
        saved = torch.load(config['model_pth'])
        model = saved['model']

        # evaluate on common corruption dataset
        dataset = self.prepare_dataset(
            saved['task'], saved['grayscale'],
            config['corruption'], config['severity'],
            )
        loss, acc = evaluate(
            model, dataset, self.device, self.batch_size, self.worker_num,
            )
        result = {'loss': loss, 'acc': acc}
        preview = {}
        return result, preview

if __name__=='__main__':
    export_dir = os.path.join(args.store_dir, 'exported')
    assert os.path.exists(export_dir), "directory of exported models not found"

    job = CorruptionTest(args.store_dir, args.device, args.batch_size, args.worker_num)
    search_spec = {
        'model_pth': [os.path.join(export_dir, f) for f in os.listdir(export_dir)],
        'corruption': CORRUPTIONS,
        'severity': SEVERITIES,
        }

    job.random_search(search_spec, args.process_num, args.max_wait, args.tolerance)
