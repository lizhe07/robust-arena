# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 00:29:35 2020

@author: Zhe
"""

import os, argparse, time, torch, torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from jarvis import BaseJob
from jarvis.vision import prepare_datasets, prepare_model, evaluate
from jarvis.utils import get_seed, set_seed, update_default, \
    numpy_dict, tensor_dict, progress_str, time_str

from . import augmentations

from . import __version__ as VERSION

EVAL_DEVICE = 'cuda'
EVAL_BATCH_SIZE = 160
TRAIN_DISP_NUM = 6
WORKER_NUM = 0


class AugmixJob(BaseJob):

    def __init__(self, save_dir, benchmarks_dir,
                 eval_device=EVAL_DEVICE, eval_batch_size=EVAL_BATCH_SIZE,
                 train_disp_num=TRAIN_DISP_NUM, worker_num=WORKER_NUM):
        super(AugmixJob, self).__init__(save_dir)
        self.benchmarks_dir = benchmarks_dir
        self.eval_device = eval_device
        self.eval_batch_size = eval_batch_size
        self.train_disp_num = train_disp_num
        self.worker_num = worker_num

    def get_work_config(self, arg_strs):
        model_config, train_config, aug_config = get_configs(arg_strs)

        work_config = {
            'model_config': model_config,
            'train_config': train_config,
            'aug_config': aug_config,
            }
        return work_config

    def main(self, work_config):
        run_config = {
            'benchmarks_dir': self.benchmarks_dir,
            }

        losses, accs, states, best_epoch = main(
            **work_config, run_config=run_config
            )

        output = {
            'losses': losses,
            'accs': accs,
            'best_state': states[best_epoch],
            'best_epoch': best_epoch,
            }
        preview = {
            'loss_valid': losses['valid'][best_epoch],
            'loss_test': losses['test'][best_epoch],
            'acc_valid': accs['valid'][best_epoch],
            'acc_test': accs['test'][best_epoch],
            }
        return output, preview

    def export_best(self, model_config, top_k=5):
        matched_ids, losses = [], []
        for w_id in self.completed_ids():
            config = self.configs.fetch_record(w_id)
            if config['model_config']==model_config:
                matched_ids.append(w_id)
                losses.append(self.previews.fetch_record(w_id)['loss_test'])
        best_ids = [w_id for i, (w_id, _) in enumerate(sorted(
            zip(matched_ids, losses), key=lambda item:item[1], reverse=True)) if i<top_k]

        export_dir = os.path.join(self.save_dir, 'exported')
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        for w_id in best_ids:
            config = self.configs.fetch_record(w_id)
            output = self.outputs.fetch_record(w_id)
            model = prepare_model(**config['model_config'])
            model.load_state_dict(tensor_dict(output['best_state']))

            saved = {
                'version': VERSION,
                'config': config,
                'model': model,
                'losses': output['losses'],
                'accs': output['accs'],
                'best_epoch': output['best_epoch'],
                }
            torch.save(saved, os.path.join(export_dir, '{}.pt'.format(w_id)))
        return matched_ids


def aug(image, aug_config, preprocess):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations
  if aug_config['all_ops']:
    aug_list = augmentations.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * aug_config['mixture_width']))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(aug_config['mixture_width']):
    image_aug = image.copy()
    depth = aug_config['mixture_depth'] if aug_config['mixture_depth'] > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, aug_config['aug_severity'])
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


class AugMixDataset(Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, aug_config):
    self.dataset = dataset
    self.aug_config = aug_config
    self.preprocess = torchvision.transforms.ToTensor()

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.aug_config['no_jsd']:
      return aug(x, self.aug_config, self.preprocess), y
    else:
      im_tuple = (self.preprocess(x), aug(x, self.aug_config, self.preprocess),
                  aug(x, self.aug_config, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)


def train(model, optimizer, dataset, weight, no_jsd, batch_size, device,
          disp_num=TRAIN_DISP_NUM, worker_num=WORKER_NUM):
    r"""Trains the model for one epoch.

    Args
    ----
    model: nn.Module
        The model to be trained.
    optimizer: Optimizer
        The optimizer for `model`.
    task_dataset: Dataset
        The classification task dataset.
    weight: (class_num,), tensor
        The class weight for unbalanced training set.
    train_config: dict
        The training configuration dictionary.
    reg_config: dict
        The regularization configuration dictionary.
    beta: float
        The damping coefficient for updating mean activation.
    eps: float
        The small positive number used in similarity loss.
    disp_num: int
        The display number for one training epoch.
    worker_num: int
        The number of workers of the data loader.

    """
    model.train().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight).to(device)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, drop_last=True, num_workers=worker_num
        )

    batch_num = len(loader)
    for batch_idx, batch in enumerate(loader, 1):
        if no_jsd:
            images, labels = batch
            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))
        else:
            (images, images_aug1, images_aug2), labels = batch
            logits = model(images.to(device))
            logits_aug1 = model(images_aug1.to(device))
            logits_aug2 = model(images_aug2.to(device))

            loss = criterion(logits, labels.to(device))

            p_clean = F.softmax(logits, dim=1)
            p_aug1 = F.softmax(logits_aug1, dim=1)
            p_aug2 = F.softmax(logits_aug2, dim=1)
            p_mixture = torch.clamp((p_clean+p_aug1+p_aug2)/3., 1e-7, 1)
            loss += 12*(F.kl_div(p_mixture, p_clean, reduction='batchmean')+
                        F.kl_div(p_mixture, p_aug1, reduction='batchmean')+
                        F.kl_div(p_mixture, p_aug2, reduction='batchmean'))/3.

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx%(-(-batch_num//disp_num))==0 or batch_idx==batch_num:
            with torch.no_grad():
                loss = criterion(logits, labels.to(device))
                _, predicts = logits.max(dim=1)
                flags = (predicts.cpu()==labels).to(torch.float)
                if weight is None:
                    acc = flags.mean()
                else:
                    acc = (flags*weight[labels]).sum()/weight[labels].sum()
            print('{}: [loss: {:4.2f}] [acc:{:7.2%}]'.format(
                progress_str(batch_idx, batch_num, True),
                loss.item(), acc.item(),
                ))


def get_configs(arg_strs=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', '16ImageNet', 'ImageNet'])
    parser.add_argument('--arch', default='ResNet18', choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'])

    parser.add_argument('--train_device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--train_seed', type=int)
    parser.add_argument('--valid_num', type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--epoch_num', default=50, type=int)

    parser.add_argument('--mixture_width', default=3, type=int)
    parser.add_argument('--mixture_depth', default=-1, type=int)
    parser.add_argument('--aug_severity', default=1, type=int)
    parser.add_argument('--no_jsd', action='store_true')
    parser.add_argument('--all_ops', action='store_true')

    args, arg_strs = parser.parse_known_args(arg_strs)

    model_config = {
        'task': args.task,
        'arch': args.arch,
        }

    if args.valid_num is None:
        if model_config['task'].startswith('CIFAR'):
            args.valid_num = 5000
        if model_config['task']=='16ImageNet':
            args.valid_num = 100
        if model_config['task']=='ImageNet':
            args.valid_num = 50
    train_config = {
        'device': args.train_device,
        'seed': get_seed(args.train_seed),
        'valid_num': args.valid_num,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'epoch_num': args.epoch_num,
        }
    aug_config = {
        'mixture_width': args.mixture_width,
        'mixture_depth': args.mixture_depth,
        'aug_severity': args.aug_severity,
        'no_jsd': args.no_jsd,
        'all_ops': args.all_ops,
        }

    return model_config, train_config, aug_config


def main(model_config, train_config, aug_config, run_config=None):
    print('model config:\n{}'.format(model_config))
    print('train config:\n{}'.format(train_config))
    print('aug config:\n{}'.format(aug_config))
    run_config = update_default({
        'benchmarks_dir': 'benchmarks',
        'eval_device': EVAL_DEVICE,
        'eval_batch_size': EVAL_BATCH_SIZE,
        'train_disp_num': TRAIN_DISP_NUM,
        'worker_num': WORKER_NUM,
        }, run_config)
    set_seed(train_config['seed'])

    # prepare task datasets
    dataset_train, dataset_valid, dataset_test, weight = \
        prepare_datasets(model_config['task'], run_config['benchmarks_dir'],
                         train_config['valid_num'], train_tensor=False)
    dataset_train = AugMixDataset(dataset_train, aug_config)

    # prepare model
    model = prepare_model(model_config['task'], model_config['arch'])
    print('\n{} model for {} initialized'.format(
        model_config['arch'], model_config['task'],
        ))

    # prepare optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(), lr=train_config['lr'],
        momentum=train_config['momentum'],
        weight_decay=train_config['weight_decay']
        )

    # train until completed
    epoch_idx = 0
    losses = {'valid': [], 'test': []}
    accs = {'valid': [], 'test': []}
    states = []
    while True:
        tic = time.time()
        # evaluate task performance on validation and testing set
        print('evaluating task performance...')
        for key, dataset in zip(['valid', 'test'], [dataset_valid, dataset_test]):
            if key=='valid':
                print('validation set:')
            if key=='test':
                print('testing set:')
            loss, acc = evaluate(
                model, dataset,
                device=run_config['eval_device'],
                batch_size=run_config['eval_batch_size'],
                worker_num=run_config['worker_num'],
                )
            print('loss: {:4.2f}, acc:{:7.2%}'.format(loss, acc))
            losses[key].append(loss)
            accs[key].append(acc)
        # save model parameters
        states.append(numpy_dict(model.state_dict()))
        best_epoch = losses['valid'].index(min(losses['valid']))
        toc = time.time()
        print('elapsed time for evaluation: {}'.format(time_str(toc-tic)))

        epoch_idx += 1
        if epoch_idx>train_config['epoch_num']:
            break
        print('\nepoch {}'.format(epoch_idx))

        # adjust learning rate and reload from checkpoints
        if epoch_idx in [int(0.5*train_config['epoch_num'])+1,
                         int(0.8*train_config['epoch_num'])+1,
                         train_config['epoch_num']]:
            optimizer.param_groups[0]['lr'] *= 0.1
            model.load_state_dict(tensor_dict(states[best_epoch]))
            print('learning rate decreased, and best model so far reloaded')

        tic = time.time()
        print('training...')
        print('lr: {:.4g}'.format(optimizer.param_groups[0]['lr']))
        train(
            model, optimizer, dataset_train, weight, aug_config['no_jsd'],
            train_config['batch_size'], train_config['device'],
            disp_num=run_config['train_disp_num'],
            worker_num=run_config['worker_num']
            )
        toc = time.time()
        print('elapsed time for one epoch: {}'.format(time_str(toc-tic)))

    print('\ntest acc at best epoch ({}) {:.2%}'.format(best_epoch, accs['test'][best_epoch]))
    return losses, accs, states, best_epoch
