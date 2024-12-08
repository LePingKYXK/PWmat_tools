
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


parser = ap.ArgumentParser(add_help=True,
                           formatter_class=ap.ArgumentDefaultsHelpFormatter,
                           description="""
                           Author:   Dr. Huan Wang,
                           Email:    huan.wang@whut.edu.cn,
                           Version:  v1.0,
                           Date:     December 08, 2024,
                           Modified: September 20, 2024""")
parser.add_argument("-f1", "--inputf1",
                    #metavar="<itype_time>",
                    type=Path,
                    required=True,
                    help="The TDOS file",
                    default="TDOS.totalspin_projected_ShiftFermi"
                    )
parser.add_argument("-f2", "--inputf2",
                    #metavar="<wavelength>",
                    type=Path,
                    required=True,
                    help="The occ. electronDOS file",
                    default="occDOS.totalspin_projected_ShiftFermi"
                    )
parser.add_argument("-o", "--output",
                    #metavar="<wavelength>",
                    type=Path,
                    required=True,
                    help="The output DOS file, containing the total, electron, and hole",
                    )
args = parser.parse_args()


def read_DOS_file(filename: Path) -> np.ndarray:
    data = np.genfromtxt(filename, comments='#')
    return data


def substraction(TDOS: np.ndarray, occDOS: np.ndarray) -> np.ndarray:
    return TDOS[:, 1:] - occDOS[:, 1:]


def main():
    tdos_file   = args.inputf1
    occdos_file = args.inputf2
    TDOS_array = read_DOS_file(tdos_file)
    occDOS_array = read_DOS_file(occdos_file)
    results = np.concatenate((TDOS_array, occDOS_array), axis=1)




if __name__ == '__main__':
    main()
