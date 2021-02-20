# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:18:14 2021

@author: Zhe
"""

import os, argparse, pickle, torch
from torch.utils.data import DataLoader
import numpy as np

from jarvis import BaseJob
from jarvis.utils import job_parser
from jarvis.vision import prepare_datasets

from . import DEVICE, BATCH_SIZE, WORKER_NUM

NOISE_TYPES = ['Gaussian', 'Uniform', 'SaltPepper']
DEFALT_VALS = {
    'Gaussian': [0, 0.02, 0.04, 0.06, 0.08, 0.1],
    'Uniform': [0, 0.02, 0.04, 0.06, 0.08],
    'SaltPepper': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    }


class NoiseJob(BaseJob):

    def __init__(self, store_dir, datasets_dir, device=DEVICE,
                 batch_size=BATCH_SIZE, worker_num=WORKER_NUM):
        if store_dir is None:
            super(NoiseJob, self).__init__()
        else:
            super(NoiseJob, self).__init__(os.path.join(store_dir, 'n-tests'))
        self.datasets_dir = datasets_dir
        self.device = 'cuda' if device=='cuda' and torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.worker_num = worker_num

    def get_config(self, arg_strs):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model_pth')
        parser.add_argument('--n_type', choices=NOISE_TYPES)
        parser.add_argument('--n_val', default=0., type=float)

        args, _ = parser.parse_known_args(arg_strs)

        assert args.model_pth is not None
        return {
            'model_pth': args.model_pth,
            'n_type': args.n_type,
            'n_val': args.n_val,
            }

    def distort(self, images, n_type, n_val):
        if n_type=='Gaussian':
            images += torch.randn_like(images)*n_val
        if n_type=='Uniform':
            images += (torch.rand_like(images)-0.5)*2*n_val
        if n_type=='SaltPepper':
            saltpepper = (torch.rand_like(images)<n_val).to(torch.float)*(2*torch.randint_like(images, 2)-1)
            images = images+saltpepper
        images = torch.clamp(images, 0, 1)
        return images

    def evaluate(self, model, dataset, n_type, n_val):
        model.eval().to(self.device)
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')

        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.worker_num)
        loss, count = 0., 0.
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            images = self.distort(images, n_type, n_val)

            with torch.no_grad():
                logits = model(images)
                loss += criterion(logits, labels).item()
                _, predicts = logits.max(dim=1)
                count += (predicts==labels).to(torch.float).sum().item()
        loss = loss/len(dataset)
        acc = count/len(dataset)
        return loss, acc

    def main(self, config, verbose=True):
        if verbose:
            print(config)

        # load model
        saved = torch.load(config['model_pth'])
        model = saved['model']

        # prepare testing dataset
        kwargs = {'task': saved['task'], 'datasets_dir': self.datasets_dir}
        if 'grayscale' in saved:
            kwargs['grayscale'] = saved['grayscale']
        dataset = prepare_datasets(**kwargs)

        loss, acc = self.evaluate(model, dataset, config['n_type'], config['n_val'])
        if verbose:
            print('accuracy: {:.2%}'.format(acc))

        result = {'loss': loss, 'acc': acc}
        preview = {}
        return result, preview

    def summarize(self, model_pths, n_type, n_vals):
        r"""Summarizes a list of models.

        Args
        ----
        model_pths: list
            A list of model paths, each of which can be loaded by `torch.load`.

        Returns
        -------
        accs: dict
            A dictionary with corruption names as keys. Each item is a numpy
            array, containing testing accuracies of each model.

        """
        accs = {}
        for n_val in n_vals:
            accs[n_val] = []
            for model_pth in model_pths:
                config = {
                    'model_pth': model_pth,
                    'n_type': n_type,
                    'n_val': n_val,
                    }
                key = self.configs.add(config)
                if self.is_completed(key):
                    accs[n_val].append(self.results[key]['acc'])
            accs[n_val] = np.array(accs[n_val])
        return accs


if __name__=='__main__':
    parser = job_parser()
    parser.add_argument('--store_dir', default='store')
    parser.add_argument('--datasets_dir', default='vision_datasets')
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
    parser.add_argument('--worker_num', default=WORKER_NUM, type=int)
    parser.add_argument('--n_type', default='Gaussian', choices=NOISE_TYPES)
    args = parser.parse_args()

    if args.spec_pth is None:
        export_dir = os.path.join(args.store_dir, 'models', 'exported')
        assert os.path.exists(export_dir), "directory of exported models not found"
        search_spec = {
            'model_pth': [os.path.join(export_dir, f) for f in os.listdir(export_dir) if f.endswith('.pt')],
            'n_type': [args.n_type],
            'n_val': DEFALT_VALS[args.n_type],
            }
    else:
        with open(args.spec_pth, 'rb') as f:
            search_spec = pickle.load(f)

    job = NoiseJob(
        args.store_dir, args.datasets_dir, args.device, args.batch_size, args.worker_num
        )
    job.random_search(search_spec, args.process_num, args.max_wait, args.tolerance)
