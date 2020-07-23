import argparse
from pathlib import Path
import logging
from typing import List
import numpy
import PythonLibrary.utility as PLut
import PythonLibrary.io as PLio
import FortranLibrary as FL

logger = logging.getLogger(Path(__file__).stem)
logging.basicConfig()
logger.setLevel("DEBUG")

# Command line input
def parse_args() -> argparse.Namespace:
    PLut.EchoCommand()
    parser = argparse.ArgumentParser('Surface generation package based on pytorch')
    parser.add_argument('job', type=str, help='job type: pretrain, train')
    parser.add_argument('format', type=str, help='file format: Columbus7 or default')
    parser.add_argument('UniqIntCoord', type=Path, help='unique internal coordinate definition file')
    parser.add_argument('origin', type=Path, help='internal coordinate space origin file')
    # pretrain
    parser.add_argument('-s','--symmetry', type=Path, help='symmetry definition file')
    parser.add_argument('-d','--data_set', type=Path, nargs='+', help='data set list file or directory')
    parser.add_argument('-t','--data_type', type=str, default='double', help='data type: float, double, default = double')
    #
    args = parser.parse_args()
    return args

# Read the origin_file, return cartdim and origin
def cartdim_origin(format:str, origin_file:Path, intdim:int) -> (int, numpy.ndarray):
    if format == 'Columbus7':
        NAtoms, _, _, r, _ = PLio.read_geom_Columbus7(origin_file)
    else:
        NAtoms, _, r = PLio.read_geom_xyz(origin_file)
    cartdim = 3 * NAtoms
    origin = numpy.empty(intdim)
    FL.InternalCoordinate(r, origin)
    return cartdim, origin

def verify_data_set(original_data_set:List[Path]) -> List[Path]:
    data_set = []
    for item in original_data_set:
        if item.is_dir():
            data_set.append(item)
        else:
            with open(item, 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break
                    data_set.append(item.parent / Path(line.strip()))
    logger.info("The training set will be read from: ")
    line = '    '; line_length = 4; last_log = False
    for data in data_set:
        data_string = str(data)
        line += data_string + ' '
        line_length += len(data_string) + 1
        last_log = True
        if len(line) > 50:
            logger.info(line)
            line = '    '; line_length = 4; last_log = False
    if last_log: logger.info(line)
    return data_set