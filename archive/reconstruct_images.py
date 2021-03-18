# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 00:49:38 2020

@author: Zhe
"""

import argparse, pickle, time, random
from roarena import reconstructor

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', default='save/reconstructions')
parser.add_argument('--datasets_dir', default='vision_datasets')
parser.add_argument('--device', default=reconstructor.DEVICE)
parser.add_argument('--batch_size', default=reconstructor.BATCH_SIZE, type=int)
parser.add_argument('--disp_num', default=reconstructor.DISP_NUM, type=int)

parser.add_argument('--spec_path', default='save/jobs/image_reconstructions.pickle')
parser.add_argument('--max_wait', default=1, type=float, help='seconds of wait before each job')
parser.add_argument('--process_num', default=1, type=int, help='number of works to process')
parser.add_argument('--tolerance', default=float('inf'), type=float, help='hours since start')

args = parser.parse_args()

if __name__=='__main__':
    job = reconstructor.ReconsJob(
        args.save_dir, datasets_dir=args.datasets_dir, device=args.device,
        batch_size=args.batch_size, disp_num=args.disp_num,
        )

    with open(args.spec_path, 'rb') as f:
        search_spec = pickle.load(f)
    random_wait = random.random()*args.max_wait
    print('random wait {:.1f}s'.format(random_wait))
    time.sleep(random_wait)

    job.random_search(search_spec, args.process_num, args.tolerance)
