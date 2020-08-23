'''
SurfGenTorch: surface generation package based on libtorch
'''

import argparse
from pathlib import Path
from typing import List
import numpy
import torch

import FortranLibrary as FL
import PythonLibrary.utility as PLut

import SSAIC
import pretrain

def parse_args() -> argparse.Namespace:
    PLut.EchoCommand()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('job', type=str, help='pretrain, train')
    parser.add_argument('format', type=str, help='Columbus7, default')
    parser.add_argument('IntCoordDef', type=Path, help='internal coordinate definition file')
    parser.add_argument('origin', type=Path, help='internal coordinate origin file')
    parser.add_argument('scale_symmetry', type=Path, help='scale and symmetry definition file')
    parser.add_argument('data_set', type=Path, nargs='+', help='data set list file or directory')
    parser.add_argument('-c','--checkpoint', type=Path, nargs='+', help="checkpoint file to continue from")
    parser.add_argument('-o','--optimizer', type=str, default='Adam', help='Adam, SGD, LBFGS (default = Adam)')
    parser.add_argument('-e','--epoch', type=int, default=1000, help='default = 1000')
    # pretrain only
    parser.add_argument('-i','--irreducible', type=int, help='the irreducible to pretrain')
    parser.add_argument('-m','--max_depth', type=int, default=0, help='max depth of the pretraining network (default = unlimited)')
    parser.add_argument('-f','--freeze', type=int, default=0, help='freeze leading layers (default = 0)')
    args = parser.parse_args()
    return args

def verify_data_set(data_set_file:List[Path]) -> List[Path]:
    data_set = []
    for item in data_set_file:
        if item.is_dir():
            data_set.append(item)
        else:
            with open(item, 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break
                    data_set.append(item.parent / Path(line.strip()))
    print("The training set will be read from: ")
    line = '    '; line_length = 4; last_log = False
    for data in data_set:
        data_string = str(data)
        line += data_string + ' '
        line_length += len(data_string) + 1
        last_log = True
        if len(line) > 75:
            print(line)
            line = '    '; line_length = 4; last_log = False
    if last_log: print(line)
    return data_set

if __name__ == "__main__":
    print(__doc__)
    print("Yifan Shen 2020\n")
    args = parse_args()
    print()
    PLut.ShowTime()
    print()
    print("Job type: " + args.job)
    print("File format: " + args.format)
    SSAIC.define_SSAIC(args.format, args.IntCoordDef, args.origin, args.scale_symmetry)
    data_set = verify_data_set(args.data_set)

    print()
    if args.job == 'pretrain':
        pretrain.pretrain(args.irreducible, args.max_depth, args.freeze,
            data_set,
            chk=args.checkpoint,
            opt=args.optimizer, epoch=args.epoch)

    print()
    PLut.ShowTime()
    print("Mission success")