# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:39:55 2020

@author: Zhe
"""

import os, argparse, torch
import numpy as np

from jarvis import BaseJob
from jarvis.vision import evaluate
from jarvis.utils import update_default

DEVICE = 'cuda'
EVAL_BATCH_SIZE = 160
WORKER_NUM = 0


class CorruptionTestJob(BaseJob):

    def __init__(self, save_dir, datasets_dir, **kwargs):
        super(CorruptionTestJob, self).__init__(save_dir)
        self.run_config = dict(
            datasets_dir=datasets_dir, **kwargs
            )

    def get_work_config(self, arg_strs):
        model_pth, corrupt_config = get_configs(arg_strs)
        work_config = {
            'model_pth': model_pth,
            'corrupt_config': corrupt_config,
            }
        return work_config

    def main(self, work_config):
        loss, acc = main(**work_config, **self.run_config)
        output = {
            'loss': loss,
            'acc': acc,
            }
        preview = output
        return output, preview


def prepare_dataset(task, corruption, severity, datasets_dir):
    if task=='CIFAR10':
        images = np.load(os.path.join(
            datasets_dir, 'CIFAR-10-C', f'{corruption}.npy'
            ))/255.
        images = images[(severity-1)*10000:severity*10000].transpose(0, 3, 1, 2)
        labels = np.load(os.path.join(
            datasets_dir, 'CIFAR-10-C', 'labels.npy'
            ))[:10000]

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(images, dtype=torch.float),
            torch.tensor(labels, dtype=torch.long),
            )
    return dataset


def get_configs(arg_strs=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_pth')
    parser.add_argument('--corruption', choices=[
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        ])
    parser.add_argument('--severity', default=5, type=int)

    args = parser.parse_args(arg_strs)

    model_pth = args.model_pth
    corrupt_config = {
        'corruption': args.corruption,
        'severity': args.severity,
        }
    return model_pth, corrupt_config


def main(model_pth, corrupt_config, **kwargs):
    print('model path: {}'.format(model_pth))
    print('corruption config: {}'.format(corrupt_config))
    run_config = update_default({
        'datasets_dir': 'vision_datasets',
        'device': DEVICE,
        'eval_batch_size': EVAL_BATCH_SIZE,
        'worker_num': WORKER_NUM,
        }, kwargs)

    saved = torch.load(model_pth)
    model = saved['model']

    dataset = prepare_dataset(
        saved['config']['model_config']['task'],
        corrupt_config['corruption'],
        corrupt_config['severity'],
        run_config['datasets_dir'],
        )

    loss, acc = evaluate(
        model, dataset, run_config['device'],
        run_config['eval_batch_size'], run_config['worker_num']
        )
    return loss, acc
