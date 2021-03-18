# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:58:51 2020

@author: Zhe
"""

import argparse, pickle, time, random
from roarena import corruption_tester

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', default='save/corruption_tests')
parser.add_argument('--datasets_dir', default='vision_datasets')
parser.add_argument('--device', default=corruption_tester.DEVICE)
parser.add_argument('--eval_batch_size', default=corruption_tester.EVAL_BATCH_SIZE, type=int)
parser.add_argument('--worker_num', default=corruption_tester.WORKER_NUM, type=int)

parser.add_argument('--spec_path', default='save/jobs/corruption_tests.pickle')
parser.add_argument('--max_wait', default=1, type=float, help='seconds of wait before each job')
parser.add_argument('--process_num', default=1, type=int, help='number of works to process')
parser.add_argument('--tolerance', default=float('inf'), type=float, help='hours since start')

args = parser.parse_args()

if __name__=='__main__':
    job = corruption_tester.CorruptionTestJob(
        args.save_dir, datasets_dir=args.datasets_dir, device=args.device,
        eval_batch_size=args.eval_batch_size, worker_num=args.worker_num,
        )

    with open(args.spec_path, 'rb') as f:
        search_spec = pickle.load(f)
    random_wait = random.random()*args.max_wait
    print('random wait {:.1f}s'.format(random_wait))
    time.sleep(random_wait)

    job.random_search(search_spec, args.process_num, args.tolerance)
