# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:23:25 2021

@author: Zhe
"""

import argparse

def job_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_pth')
    parser.add_argument('--process_num', default=0, type=int)
    parser.add_argument('--max_wait', default=1, type=float)
    parser.add_argument('--tolerance', default=float('inf'), type=float)
    return parser
