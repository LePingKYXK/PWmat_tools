
import argparse as ap
import numpy as np
import re


parser = ap.ArgumentParser(add_help=True,
                           formatter_class=ap.ArgumentDefaultsHelpFormatter,
                           description="""
                           Author:   Dr. Huan Wang,
                           Email:    huan.wang@whut.edu.cn,
                           Version:  v1.0,
                           Date:     Octber 13, 2024
                           """)
parser.add_argument("-f", "--filename",
                    metavar="<input filename>",
                    type=Path,
                    help="The MOVEMENT file",
                    default=Path.cwd() / "MOVEMENT"
                    )
parser.add_argument("-i", "--indices",
                    metavar="<indices of atoms>",
                    type=int,
                    nargs="+",
                    help="The indices of atoms to be analyzed",
                    default=0
                    )
args = parser.parse_args()


def number_of_atoms(filename: Path) -> int:
    """Return the number of atoms in the system."""
    
    with open(filename, 'r') as fo:
        for line in fo:
            match_atom = re.search(r"(\d+)\s*atoms", line)
            if match_atom:
                break
    return int(match_atom.group(1))


def parse_MOVEMENT_file(filename: Path, row_marks: np.ndarray) -> np.ndarray:
    """Parse the MOVEMENT file to obtain time and Force vector components (x, y, z)."""

    data_list = []
    time_list = []
    table_data = []
    
    with open(filename, 'r') as fo:
        collecting = False
        current_line = 0
        
        for line in fo:
            matche_time = re.search(r"Iteration\s*=\s+(\d+\.\d+E[-+]?\d+)", line)
            if matche_time:
                time_list.append(float(matche_time.group(1)))
                
            if "-Force" in line:
                collecting = True
                current_line = 0
                table_data = []
                
            elif "Velocity" in line:
                collecting = False
                data_list.append(table_data)
            
            if collecting and not "-Force" in line:
                table_data.append(line.strip().split()[1:])
                current_line += 1
                if isinstance(row_marks, int) and current_line == row_marks:
                    collecting = False
                elif isinstance(row_marks, np.ndarray) and current_line == row_marks.max():
                    collecting = False
  
    return time_list, data_list


def main():
    filename = args.filename
    indices = args.indices
    num_atom = number_of_atoms(filename)
    parse_MOVEMENT_file(filename, indices)


if "__main__" == __name__:
    main()
