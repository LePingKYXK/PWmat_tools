
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path


parser = ap.ArgumentParser(add_help=True,
                    formatter_class=ap.ArgumentDefaultsHelpFormatter,
                    description="""
                    Author:  Dr. Huan Wang,
                    Email:   huan.wang@whut.edu.cn,
                    Version: v1.0,
                    Date:    June 30, 2024,
                    Modify:  July 13, 2024"""
                    )
parser.add_argument("-i", "--inputfile",
                    metavar="<plot.TDDFT.DOS>",
                    type=Path,
                    help="The path to the plot.TDDFT.DOS file",
                    default=(Path.cwd() / "plot.TDDFT.DOS"),
                    )
parser.add_argument("-o", "--outputfile",
#                    metavar="<save to outputfile>",
                    type=Path,
                    help="The output file name",
                    default=(Path.cwd() / "Number_of_Excited_electrons.dat"),
                    )
parser.add_argument("-p", "--plot_fig",
#                    metavar="<option for plotting>",
                    type=str,
                    help="option for plotting. y for plotting, n for not",
                    default=("n"),
                    )
args = parser.parse_args()


def sum_electron(data):
    return np.sum(data[:,1:], axis=-1)


def read_file(inputfile, outputfile):
    info = np.genfromtxt(inputfile)
    data = sum_electron(info)
    num_elec = np.column_stack((info[:,0], data))
    np.savetxt(Path.cwd() / outputfile, num_elec,
               fmt=("%8.1f", "%10.8e"), delimiter='\t',
               header='Time\tNum_of_excited_electrons')
    return num_elec


def read_report(filename):
    with open(filename, "r") as fo:
        for line in fo:
            matches = re.search(r'NUM_ELECTRON\s*=\s+(\d+\.\d+)', line, re.IGNORECASE)
            if matches:
                break
    return float(matches.group(1))


def plot_figure(data):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(data[:, 0], data[:, 1], "-")
    ax.set_xlim([0, data[:, 0].max()])
    ax.set_ylim([0, data[:, 1].max() * 1.2])
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Number of Excited Electrons')
    plt.show()

def main():
    """
    Workflow:
        (1) Read the plot.TDDFT.DOS file
        (2) Plot figure if request

    Returns
    -------
    None.

    """
    inputfile, outputfile = args.inputfile, args.outputfile

    data = read_file(inputfile, outputfile)

    total_electrons = read_report("REPORT")
    num_excited_ele = data[:, 1].max()
    percent_excited_electrons = num_excited_ele / total_electrons * 100

    draw_line = "-" * 79
    print(''.join(("\n", draw_line)))
    print(f"Total number of electrons: {total_electrons},\n"
          f"Number of excited electrons: {num_excited_ele},\n"
          f"{percent_excited_electrons: .2f}% of electrons were excited.")
    print(''.join((draw_line, "\n")))

    if args.plot_fig.lower() in ["y", "yes"]:
        plot_figure(data)


if __name__ == "__main__":
    main()