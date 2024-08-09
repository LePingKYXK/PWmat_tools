

import argparse as ap
import re
import time
from pathlib import Path


parser = ap.ArgumentParser(add_help=True,
                           formatter_class=ap.ArgumentDefaultsHelpFormatter,
                           description="""
                           Author:   Dr. Huan Wang,
                           Email:    huan.wang@whut.edu.cn,
                           Version:  v2.0,
                           Date:     August 09, 2024""")
parser.add_argument("-f", "--filename",
                    type=Path,
                    help="The MOVEMENT file",
                    default=(Path.cwd() / "MOVEMENT"),
                    )
parser.add_argument("-s", "--steps",
                    metavar="<index>",
                    type=int,
                    nargs="+",
                    help="The step in MOVEMENT",
                    )
args = parser.parse_args()



