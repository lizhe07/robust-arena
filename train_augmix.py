# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 14:25:33 2020

@author: Zhe
"""

import argparse, random, time
from roarena.augmix import AugmixJob


parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', default='augmix_models')
parser.add_argument('--benchmarks_dir', default='benchmarks')
parser.add_argument('--max_wait', default=60, type=float, help='seconds of wait before each job')
parser.add_argument('--process_num', default=0, type=int, help='number of works to process')
parser.add_argument('--tolerance', default=float('inf'), type=float, help='hours since start')

args = parser.parse_args()


if __name__=='__main__':
    random_wait = random.random()*args.max_wait
    print('random wait {:.1f}s'.format(random_wait))
    time.sleep(random_wait)

    augmix_job = AugmixJob(args.save_dir, args.benchmarks_dir)
    search_spec = {
        'train_seed': [41, 85, 29, 63, 7],
        }

    augmix_job.random_search(search_spec, args.process_num, args.tolerance)
