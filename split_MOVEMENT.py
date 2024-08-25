

import argparse as ap
import re
import time
from pathlib import Path


parser = ap.ArgumentParser(add_help=True,
                           formatter_class=ap.ArgumentDefaultsHelpFormatter,
                           description="""
                           Author:   Dr. Huan Wang,
                           Email:    huan.wang@whut.edu.cn,
                           Version:  v1.1,
                           Date:     August 09, 2024
                           Modified: August 19, 2024""")
parser.add_argument("-f", "--filename",
                    type=Path,
                    help="The MOVEMENT file",
                    default=(Path.cwd() / "MOVEMENT"),
                    )
parser.add_argument("-s", "--steps",
                    metavar="<index>",
                    type=float,
                    nargs="+",
                    help="The step in MOVEMENT",
                    )
args = parser.parse_args()


def read_data(filename: Path, fs_list: list) -> list:
    """
    This function reads the MOVEMENT file and then splits and saves
    each block into a list of lists.

    Parameters
    ----------
    filename : Path
        The MOVEMENT file.
    fs_list : list
        The list contains the specific femtosecond.

    Returns
    -------
    configure : list of lists
        A list of lists contains specific configurations of the MD simulation.
    """

    structure = []
    configure = []
    femto_sec = [float(fs) for fs in fs_list]
    print("".join(("\n", "The input femtoseconds are:", " {:} " * len(femto_sec), "\n")).format(*femto_sec))

    with open(filename, 'r') as fo:
        for line in fo:
            matches = re.search(r"Iteration\s*=\s+(\d+\.\d+E[-+]?\d+)", line)
            if matches:
                iteration = float(matches.group(1))
                if iteration in femto_sec:
                    while not re.search(r"\s*(--+)\s*", line):
                        structure.append(line.rstrip("\n"))
                        line = next(fo)
                    configure.append(structure)
                    structure = []
                elif iteration > femto_sec[-1]:
                    break
    return configure


def save_atom_config(configure: list, fs_list: list):
    """
    This function saves each specific time step as the atom_(*)fs.config file.

    Parameters
    ----------
    configure : list of lists
        A list of lists contains all the configurations during the MD simulation.

    Returns
    -------
    None.

    """

    for ind, atom_config in enumerate(configure):
        filename = "".join(("atom_", str(fs_list[ind]), "fs.config"))
        with open(filename, "w") as fw:
            for line in atom_config:
                fw.write(line + '\n')
        print(f"The {fs_list[ind]} fs step has saved as {filename}")


def main():
    ''' Workflow:
    (1) read the MOVEMENT file block by block and fetch the coordinates based on the specific step(s);
    (2) save the specific coordinates into atom_(*)fs.config file(s).
    '''
    inputfile, fs_list = args.filename, args.steps
    
    try:
        configure = read_data(inputfile, fs_list)
    except FileNotFoundError:
        print("\nThe MOVEMENT file does NOT exist!\n")
        exit()

    if not fs_list:
        print("\nPlease provide (at least one) specific times, separating by space. (for example: -s 10 20.5 32.6)\n")
        exit()

    save_atom_config(configure, fs_list)
    

if __name__ == "__main__":
    main()
