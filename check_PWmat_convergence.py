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
                           Version:  v2.0,
                           Date:     August 7, 2024
                           Modified: August 16, 2024""")
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


def read_etot_input(filename: Path) -> int and str:
    """
    This function reads the etot.input file, and then returns the length of
    the parameters of RELAX_DETAIL.

    Parameters
    ----------
    filename : Path
        The path to the etot.input file.

    Returns
    -------
    len(info) : int
        The length of the parameters of RELAX_DETAIL.
    dft : str
        The functional used in the current calculation.
    """
    
    dft = "PBE"
    with open(filename, "r") as fo:
        for line in fo:
            functional = re.search(r"XCFUNCTIONAL\s+=\s+(\w+)", line)
            relaxation = re.findall(r"RELAX_DETAIL\s+=\s+(.*)", line)
            
            if functional:
                dft = functional.group(1)
            if relaxation:
                info = relaxation[0].split()
                
    return len(info), dft


def read_relaxsteps(filename: Path, dft: str) -> list:
    """
    This function deals with the RELAXSTEPS file.

    Parameters
    ----------
    filename : Path
        The path to the RELAXSTEPS file.
    dft : str
        The used functional.

    Returns
    -------
    The Data list contained the number of iteration setps, the total energies,
    and the average force of each ionic step.
    """

    all_data = []
    with open(filename) as fo:
        for line in fo:
            info = line.strip().split()
            data = re.findall(r'=\s*([-+]?\d*\.?\d*(?:[Ee][-+]?\d+)?)', line)
            status = info[2]
            if dft == "PBE" or dft == None:
                data.insert(1, status)
                all_data.append(data)

            elif dft == "HSE":
                if info[0] == "HSE":
                    data.insert(0, info[1])
                    data.insert(1, status)
                    all_data.append(data)
    return all_data


def plot_deltE_deltF(data: np.array):
    """
    This function will plot the evolution of total energy and average force
    with the relaxation steps.
    Parameters
    ----------
    data : array
        A numpy array contains the relaxation step.

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
    1) read data from the RELAXSTEPS file. If the user specifies the filename, the given file will be read.
    2) plot the Energy and the average Force against relax steps. Detailed information will be printed on the screen if the user activates the verbose mode.
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

    criteria, dft = read_etot_input(etot_input)
    info = read_relaxsteps(relaxsteps, dft)
    
    drawline = "-" * 79
    
    if info[-1][1] == '*END':
        print('\n'.join((drawline, "The optimization job completed. Congratulations！", drawline)))
    else:
        print('\n'.join((drawline, "The optimization job is NOT yet complete!", drawline)))

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
