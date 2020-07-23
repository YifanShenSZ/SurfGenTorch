from pathlib import Path
import logging
import numpy
import torch
import PythonLibrary.utility as PLut
import FortranLibrary as FL
import utility
import symmetry

logger = logging.getLogger(Path(__file__).stem)
logging.basicConfig()
logger.setLevel("DEBUG")

if __name__ == "__main__":
    print("SurfGenTorch: surface generation package based on libtorch")
    print("Yifan Shen 2020\n")
    args = utility.parse_args()
    print()
    PLut.ShowTime()
    logger.info("Job type: " + args.job)
    logger.info("File format: " + args.format)
    uniqintdim, UniqIntCoordDef = FL.FetchInternalCoordinateDefinition(args.format, file=args.UniqIntCoord)
    FL.DefineInternalCoordinate(args.format, file=args.UniqIntCoord)
    logger.info("Number of Unique internal coordinate: %d", uniqintdim)
    cartdim, origin = utility.cartdim_origin(args.format, args.origin, uniqintdim)

    if args.job == 'pretrain':
        symmetry.define_symmetry(args.symmetry)
        data_set = utility.verify_data_set(args.data_set)
