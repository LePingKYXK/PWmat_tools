from pathlib import Path
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
import re

parser = ap.ArgumentParser(add_help=True,
                           formatter_class=ap.ArgumentDefaultsHelpFormatter,
                           description="""
                           Author:   Dr. Huan Wang, 
                           Email:    huan.wang@whut.edu.cn,
                           Version:  v1.0,
                           Date:     August 7, 2024""")
parser.add_argument("-i",
                    metavar="<etot.input file>",
                    type=Path,
                    help="you can specify the file name",
                    default=(Path.cwd() / "etot.input"),
                    )
parser.add_argument("-r",
                    metavar="<RELAXSTEPS file>",
                    type=Path,
                    help="you can specify the file name",
                    default=(Path.cwd() / "RELAXSTEPS"),
                    )
parser.add_argument("-p",
                    metavar="<plot graph>",
                    type=str,
                    help="plot graph",
                    default="yes",
                    )
parser.add_argument("-v", "--verbose",
                    action="store_true",
                    help="verbose mode, show details of the RELAXSTEPS file"
                    )
args = parser.parse_args()


def read_etot_input(filename):
    pattern = r"RELAX_DETAIL = (.*)"
    with open(filename, "r") as fo:
        for line in fo:
            m = re.findall(pattern, line)
            if m:
                info = m[0].split()
                break
    return len(info)


def read_relaxsteps(filename, criteria):
    all_data = []
    with open(filename) as fo:
        for line in fo:
            info = line.strip().split()
            if criteria > 3:    # for cell optimization
                all_data.append((info[1],    # iteration steps
                                 info[2],    # opt status
                                 info[4],    # Total Energy
                                 info[6],    # Average Force
                                 info[8],    # Maximum Force
                                 info[10],   # Average stress (eV/Number_of_atom)
                                 info[12],   # Delta E_total
                                 info[14],   # Delta Rho
                                 info[16],   # SCF
                                 info[18],   # Delta |R - R(new_initial)|
                                 info[20],   # Delta AL
                                 info[22],   # p*F
                                 info[24],   # p*F0
                                 info[26]))  # Fch (check Force)

            elif criteria <= 3:    # for ionic optimization
                all_data.append((info[1],    # iteration steps
                                 info[2],    # opt status
                                 info[4],    # Total Energy
                                 info[6],    # Average Force
                                 info[8],    # Maximum Force
                                 info[10],   # Delta E_total
                                 info[12],   # Delta Rho
                                 info[14],   # SCF
                                 info[16],   # Delta |R - R(new_initial)|
                                 info[18],   # p*F
                                 info[20],   # p*F0
                                 info[22]))  # Fch (check Force)
    return all_data


def plot_deltE_deltF(data):
    """
    This function will plot the evoluation of total energy and average force
    with the relaxation steps.
    Parameters
    ----------
    data : array
        numpy array contains the relaxation step.

    Returns
    -------
    None.

    """
    step = data[:,0].astype(int)
    Etot = data[:,2].astype(float)
    aveF = data[:,6].astype(float)
    
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(12,8))
    ax1.plot(step, Etot, "k-") 
    ax1.set_xlim(step.min(), step.max())
    ax1.set_ylim(Etot.min(), Etot.max())
    ax1.set_xlabel("step")
    ax1.set_ylabel("total Energy (eV)")
    
    ax2.plot(step, aveF, "b-")
    ax2.set_xlim(step.min(), step.max())
    ax2.set_ylim(aveF.min(), aveF.max())
    ax2.set_xlabel("step")
    ax2.set_ylabel("average Force (eV/Angstrom)")
    plt.show()


def main():
    """
    Workflow:
    1) read data from RELAXSTEPS file. If the user specifid the filename, the given file will be read.
    2) plot the Energy and the average Force against relax steps. If the user activates the verbose mode, detailed information will be printed on the screen.
    """

    etot_input = args.i
    relaxsteps = args.r
    
    file_paths = [etot_input, relaxsteps]
    for file in file_paths:
        try:
            if not file.exists():
                raise FileNotFoundError(f"\nThe file {file.name} does not exist.")
        except FileNotFoundError as e:
            print(e)
            print("""Exiting the program...\n
                  Please check the file name, or you can specify the file name.\n""")
            exit(1)
    
    title_cell = "iter, status, E_tot, Aver_F_ion, Max_F_ion, Aver_F_ele, delt_E, delt_rho, SCF_cyc, delt_L, delt_AL, proj_F, proj_F0, F_check".strip().split(",")
    title_ion = "iter, status, E_tot, Aver_F_ion, Max_F_ion, delt_E, delt_rho, SCF_cyc, delt_L, proj_F, proj_F0, F_check".strip().split(",")
    
    tfmt_cell = "".join(("{:>4s}","{:^7s}","{:^21s}","{:>11s}"*5,"{:>8s}","{:>9s}","{:>11s}"*4))
    ifmt_cell = "".join(("{:>3}","{:>7s}","{:>22}","{:>11}"*5,"{:>6}","{:>11}"*5))

    tfmt_ion  = "".join(("{:>4s}","{:^7s}","{:^21s}","{:>11s}"*4,"{:>8s}","{:>9s}","{:>11s}"*3))
    ifmt_ion  = "".join(("{:>3}","{:>7s}","{:>22}","{:>11}"*4,"{:>6}","{:>11}"*4))

    criteria = read_etot_input(etot_input)
    info = read_relaxsteps(relaxsteps, criteria)
    
    drawline = "-" * 79
    
    if info[-1][1] == '*END':
        print('\n'.join((drawline, "The optimization job completed.", drawline))))
    else:
        print('\n'.join((drawline, "The optimization job is not yet complete!", drawline)))

    if args.verbose:
        if criteria > 3:
            print(tfmt_cell.format(*title_cell))
            for item in info:
                print(ifmt_cell.format(*item))
            plot_deltE_deltF(np.array(info).reshape(-1, 14))

        elif criteria <= 3:
            print(tfmt_ion.format(*title_ion))
            for item in info:
                print(ifmt_ion.format(*item))
            plot_deltE_deltF(np.array(info).reshape(-1, 12))

    else:
        if criteria > 3:
            plot_deltE_deltF(np.array(info).reshape(-1, 14))
        elif criteria <= 3:
            plot_deltE_deltF(np.array(info).reshape(-1, 12))


if __name__ == "__main__":
    main()
