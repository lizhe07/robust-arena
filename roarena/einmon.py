# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 00:01:54 2021

@author: Zhe
"""

import os, argparse, pickle, torch
from torchvision.transforms.functional import rgb_to_grayscale

from jarvis import BaseJob
from jarvis.utils import job_parser

from . import DEVICE, BATCH_SIZE, WORKER_NUM

ALPHAS = [0.05*i for i in range(20)]


class EinMonJob(BaseJob):

    def __init__(self, store_dir, datasets_dir, device=DEVICE,
                 batch_size=BATCH_SIZE, worker_num=WORKER_NUM):
        if store_dir is None:
            super(EinMonJob, self).__init__()
        else:
            super(EinMonJob, self).__init__(os.path.join(store_dir, 'em-results'))
        self.datasets_dir = datasets_dir
        self.device = 'cuda' if device=='cuda' and torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.worker_num = worker_num

    def prepare_dataset(self, task, alpha, grayscale=False):
        with open('{}/{}-EM/alpha_{:02d}.pickle'.format(
                self.datasets_dir, task, int(100*alpha),
                ), 'rb') as f:
            saved = pickle.load(f)
        images = torch.tensor(saved['images_mix'], dtype=torch.float)
        if grayscale:
            images = rgb_to_grayscale(images)
        labels_low = torch.tensor(saved['labels_low'], dtype=torch.long)
        labels_high = torch.tensor(saved['labels_high'], dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(images, labels_low, labels_high)
        return dataset

    def evaluate(self, model, dataset):
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
        parser.add_argument('--alpha', default=0.05, type=float, choices=ALPHAS)

        args = parser.parse_args(arg_strs)

        assert args.model_pth is not None
        return {
            'model_pth': args.model_pth,
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
            saved['task'], config['alpha'],
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


if __name__=='__main__':
    parser = job_parser()
    parser.add_argument('--store_dir', default='store')
    parser.add_argument('--datasets_dir', default='vision_datasets')
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
    parser.add_argument('--worker_num', default=WORKER_NUM, type=int)
    args = parser.parse_args()

    if args.spec_pth is None:
        export_dir = os.path.join(args.store_dir, 'models', 'exported')
        assert os.path.exists(export_dir), "directory of exported models not found"
        search_spec = {
            'model_pth': [os.path.join(export_dir, f) for f in os.listdir(export_dir) if f.endswith('.pt')],
            'alpha': ALPHAS,
            }
    else:
        with open(args.spec_pth, 'rb') as f:
            search_spec = pickle.load(f)

    job = EinMonJob(
        args.store_dir, args.datasets_dir, args.device, args.batch_size, args.worker_num
        )
    job.random_search(search_spec, args.process_num, args.max_wait, args.tolerance)
