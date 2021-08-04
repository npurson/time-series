import os
import random
import argparse

from os import path as osp
from typing import List


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='root directory of dataset.')
    parser.add_argument('-s', '--split', default=0.7, type=float,
                        help='ratio of splitting train set and valid set.')
    return parser.parse_args()


def walkdir(dir, cls=None, suffix='.txt') -> List[List[str]]:
    """Walks through subdirectories and
    returns a list of file paths and annotations.
    """
    anns = [[]]
    for f in os.listdir(dir):
        path = osp.join(dir, f)
        if osp.isdir(path):
            anns += walkdir(path, f)
        elif path.endswith(suffix) and cls:
            anns[0].append(' '.join([path, cls]))
    return anns


def split_data(root, split=0.7):
    anns = walkdir(root)
    with open(osp.join(root, 'train.txt'), 'w') as traintxt, \
         open(osp.join(root, 'val.txt'), 'w') as valtxt:
        for ann in anns:
            random.shuffle(ann)
            spt = int(split * len(ann))
            if spt == 0 and len(ann) != 0:
                spt += 1
            if spt == len(ann) and len(ann) != 0:
                spt -= 1
            traintxt.writelines([line + '\n' for line in ann[:spt]])
            valtxt.writelines([line + '\n' for line in ann[spt:]])


if __name__ == '__main__':
    args = parse_args()
    split_data(args.root, args.split)
