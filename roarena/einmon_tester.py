# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:08:59 2020

@author: Zhe
"""

import argparse, pickle, torch

from jarvis import BaseJob
from jarvis.utils import update_default

DEVICE = 'cuda'
BATCH_SIZE = 160
WORKER_NUM = 0


class EinMonJob(BaseJob):

    def __init__(self, save_dir, datasets_dir, **kwargs):
        super(EinMonJob, self).__init__(save_dir)
        self.run_config = dict(
            datasets_dir=datasets_dir, **kwargs
            )

    def get_work_config(self, arg_strs):
        model_pth, alpha = get_configs(arg_strs)
        work_config = {
            'model_pth': model_pth,
            'alpha': alpha,
            }
        return work_config

    def main(self, work_config):
        loss_low, acc_low, loss_high, acc_high = main(
            **work_config, **self.run_config
            )
        output = {
            'loss_low': loss_low,
            'acc_low': acc_low,
            'loss_high': loss_high,
            'acc_high': acc_high,
            }
        preview = output
        return output, preview


def prepare_datasets(task, alpha, datasets_dir):
    with open('{}/{}-EM/alpha_{:02d}.pickle'.format(
            datasets_dir, task, int(100*alpha),
            ), 'wb') as f:
        saved = pickle.load(f)
    images = saved['images_mix']
    labels_low = saved['labels_low']
    labels_high = saved['labels_high']
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(images, dtype=torch.float),
        torch.tensor(labels_low, dtype=torch.long),
        torch.tensor(labels_high, dtype=torch.long),
        )
    return dataset


def evaluate(model, dataset, device=DEVICE, batch_size=BATCH_SIZE, worker_num=WORKER_NUM):
    if device=='cuda' and not torch.cuda.is_available():
        device = 'cpu'
    model.eval().to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=worker_num
        )
    loss_low, count_low = 0., 0.
    loss_high, count_high = 0., 0.
    for images, labels_low, labels_high in loader:
        with torch.no_grad():
            logits = model(images.to(device))
            loss_low += criterion(logits, labels_low.to(device)).item()
            loss_high += criterion(logits, labels_high.to(device)).item()
            _, predicts = logits.max(dim=1)
            count_low += (predicts.cpu()==labels_low).to(torch.float).sum().item()
            count_high += (predicts.cpu()==labels_high).to(torch.float).sum().item()
    loss_low, loss_high = loss_low/len(dataset), loss_high/len(dataset)
    acc_low, acc_high = count_low/len(dataset), count_high/len(dataset)
    return loss_low, acc_low, loss_high, acc_high


def get_configs(arg_strs=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_pth')
    parser.add_argument('--alpha', default=0.05, type=float)

    args = parser.parse_args(arg_strs)

    model_pth = args.model_pth
    alpha = args.alpha
    return model_pth, alpha


def main(model_pth, alpha, **kwargs):
    print('model path:\n{}'.format(model_pth))
    print('alpha:\n{}'.format(alpha))
    run_config = update_default({
        'datasets_dir': 'vision_datasets',
        'device': DEVICE,
        'batch_size': BATCH_SIZE,
        'worker_num': WORKER_NUM,
        }, kwargs)

    # load model
    saved = torch.load(model_pth)
    model = saved['model']

    # prepare Einstein-Monroe dataset
    dataset = prepare_datasets(
        saved['config']['model_config']['task'], alpha,
        run_config['datasets_dir'],
        )

    # evaluate model
    loss_low, acc_low, loss_high, acc_high = evaluate(
        model, dataset, run_config['device'], run_config['batch_size'],
        run_config['worker_num']
        )
    print('low-frequency component\nloss: {:.4f}, acc: {:.2%}'.format(loss_low, acc_low))
    print('high-frequency component\nloss: {:.4f}, acc: {:.2%}'.format(loss_high, acc_high))

    return loss_low, acc_low, loss_high, acc_high
