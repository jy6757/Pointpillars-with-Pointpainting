import argparse
import os
import torch
from tqdm import tqdm
import pdb

from utils import setup_seed
from dataset import Kitti


def main(args):
    setup_seed()
    pdb.set_trace()
    train_dataset = Kitti(data_root=args.data_root, split='train')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='./data/kitti',
                        help='your data root for kitti')
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.00025)
    parser.add_argument('--max_epoch', type=int, default=160)
    parser.add_argument('--log_freq', type=int, default=8)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=20)
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
