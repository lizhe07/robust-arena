# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 03:46:38 2020

@author: Zhe
"""

import os, argparse, pickle, random, time
from roarena.tester import AttackJob


parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', default='adv_attacks')
parser.add_argument('--benchmarks_dir', default='benchmarks')
parser.add_argument('--models_dir', default='cnn_models')
parser.add_argument('--group_key', default='none',
                    choices=['none', 'neural', 'shuffle'])
parser.add_argument('--max_wait', default=60, type=float, help='seconds of wait before each job')
parser.add_argument('--process_num', default=0, type=int, help='number of works to process')
parser.add_argument('--tolerance', default=float('inf'), type=float, help='hours since start')

args = parser.parse_args()


if __name__=='__main__':
    random_wait = random.random()*args.max_wait
    print('random wait {:.1f}s'.format(random_wait))
    time.sleep(random_wait)

    attack_job = AttackJob(args.save_dir, args.benchmarks_dir)

    with open(os.path.join(args.models_dir, 'model.ids.pickle'), 'rb') as f:
        catalog = pickle.load(f)
    search_spec = {
        'model_pth': [os.path.join(args.models_dir, 'exported', f'{model_id}.pt') \
                      for model_id in catalog[args.group_key]],
        'norm': ['L2', 'Linf'],
        }

    attack_job.random_search(search_spec, args.process_num, args.tolerance)
