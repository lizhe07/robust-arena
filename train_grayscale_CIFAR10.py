# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:51:49 2021

@author: Zhe
"""

from argparse import ArgumentParser
import os
import git
import torch as ch
from robustness import data_augmentation as da
import torchvision.transforms as trans

import cox
import cox.utils
import cox.store

from jarvis.models.resnet import ResNet

class RobustResNet(ResNet):
    r"""Wrappter class for using robustness package.

    """

    def __init__(self, **kwargs):
        super(RobustResNet, self).__init__(**kwargs)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        pre_acts, post_acts, logits = self.layer_activations(self.normalizer(x))
        if with_latent:
            if no_relu:
                return logits, pre_acts[-1]
            else:
                return logits, post_acts[-1]
        else:
            return logits

try:
    from robustness.model_utils import make_and_restore_model
    from robustness.datasets import DATASETS
    from robustness.train import train_model, eval_model
    from robustness.tools import constants, helpers
    from robustness import defaults, __version__
    from robustness.defaults import check_and_fill_args
except:
    raise ValueError("Make sure to run with python -m (see README.md)")

def main(args, store=None):
    '''Given arguments from `setup_args` and a store from `setup_store`,
    trains as a model. Check out the argparse object in this file for
    argument options.
    '''
    # MAKE DATASET AND LOADERS
    trans_train = da.TRAIN_TRANSFORMS_DEFAULT(32)
    trans_train = trans.Compose(trans_train.transforms[:-1]+[trans.Grayscale(), trans.ToTensor()])
    trans_test = da.TEST_TRANSFORMS_DEFAULT(32)
    trans_test = trans.Compose(trans_test.transforms[:-1]+[trans.Grayscale(), trans.ToTensor()])
    data_path = os.path.expandvars(args.data)
    dataset = DATASETS[args.dataset](data_path, transform_train=trans_train, transform_test=trans_test)
    dataset.mean = ch.tensor([0.5], dtype=ch.float)
    dataset.std = ch.tensor([0.2], dtype=ch.float)

    train_loader, val_loader = dataset.make_loaders(args.workers,
                    args.batch_size, data_aug=bool(args.data_aug))

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    loaders = (train_loader, val_loader)

    # MAKE MODEL
    arch = RobustResNet(block_nums=[2, 2, 2, 2], block_type='Basic', conv0_kernel_size=3, in_channels=1, class_num=10)
    model, checkpoint = make_and_restore_model(arch=arch,
            dataset=dataset, resume_path=args.resume)
    if 'module' in dir(model): model = model.module

    print(args)
    if args.eval_only:
        return eval_model(args, model, val_loader, store=store)

    if not args.resume_optimizer: checkpoint = None
    model = train_model(args, model, loaders, store=store,
                                    checkpoint=checkpoint)
    return model

def setup_args(args):
    '''
    Fill the args object with reasonable defaults from
    :mod:`robustness.defaults`, and also perform a sanity check to make sure no
    args are missing.
    '''
    # override non-None values with optional config_path
    if args.config_path:
        args = cox.utils.override_json(args, args.config_path)

    ds_class = DATASETS[args.dataset]
    args = check_and_fill_args(args, defaults.CONFIG_ARGS, ds_class)

    if not args.eval_only:
        args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds_class)

    if args.adv_train or args.adv_eval:
        args = check_and_fill_args(args, defaults.PGD_ARGS, ds_class)

    args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds_class)
    if args.eval_only: assert args.resume is not None, \
            "Must provide a resume path if only evaluating"
    return args

def setup_store_with_metadata(args):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
    # Add git commit to args
    try:
        repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)),
                            search_parent_directories=True)
        version = repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        version = __version__
    args.version = version

    # Create the store
    store = cox.store.Store(args.out_dir, args.exp_name)
    args_dict = args.__dict__
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)

    return store

parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

if __name__ == "__main__":
    args = parser.parse_args()
    args = cox.utils.Parameters(args.__dict__)

    args = setup_args(args)
    store = setup_store_with_metadata(args)

    final_model = main(args, store=store)
