from pathlib import Path
import numpy
import torch

import FortranLibrary as FL
import PythonLibrary.utility as PLut

import utility
import SSAIC
import pretrain

if __name__ == "__main__":
    print("SurfGenTorch: surface generation package based on libtorch")
    print("Yifan Shen 2020\n")
    args = utility.parse_args()
    print()
    PLut.ShowTime()
    print()
    print("Job type: " + args.job)
    print("File format: " + args.format)
    SSAIC.define_SSAIC(args.format, args.IntCoordDef, args.origin, args.scale_symmetry)
    data_set = utility.verify_data_set(args.data_set)

    print()
    if args.job == 'pretrain':
        pretrain.pretrain(args.irreducible, args.max_depth, args.freeze,
            data_set,
            chk=args.checkpoint,
            opt=args.optimizer, epoch=args.epoch)

    print()
    PLut.ShowTime()
    print("Mission success")