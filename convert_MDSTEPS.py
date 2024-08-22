
import argparse as ap
import csv
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from matplotlib.gridspec import GridSpec


parser = ap.ArgumentParser(add_help=True,
                           formatter_class=ap.ArgumentDefaultsHelpFormatter,
                           description="""
                           Author:   Dr. Huan Wang,
                           Email:    huan.wang@whut.edu.cn,
                           Version:  v1.2,
                           Date:     August 12, 2024
                           Modified: August 22, 2024""")
parser.add_argument("-f", "--filename",
                    metavar="<PWmat MDSTEPS file>",
                    type=Path,
                    help="The PWmat MDSTEPS file",
                    default=(Path.cwd() / "MDSTEPS"),
                    )
parser.add_argument("-o", "--output",
                    metavar="<output file name>",
                    type=Path,
                    help="The csv file contains MD steps",
                    default=(Path.cwd() / "MDsteps.csv"),
                    )
parser.add_argument("-p", "--plot",
                    metavar="<option for plotting>",
                    type=str,
                    help="option for plotting. 'y' or 'yes' for plotting, while 'n' or 'no' for not plot",
                    default=("yes"),
                    )
args = parser.parse_args()


def parse_MDSTEPS(filename: Path) -> list and str:
    """
    This function grabs the information from the MDSTEPS file.

    Parameters
    ----------
    filename : Path
        The path to the MDSTEPS file.

    Returns
    -------
    header_line : list
        A list contains header variables.
    values : list
        A list of lists contains data in the MDSTEPS file, line by line.
    aveTemp : str
        A string flag. "yes" represents the "Average Temperature".
    """

    with open(filename, "r") as file:
        info = file.readlines()

    first_l = info[0]
    pattern = re.compile(r'\b(Iter|Etot|Ep|Ek|Temp|aveTemp|dE|dRho|SCF|dL|Fcheck)\b')
    header_line = re.findall(pattern, first_l)

    values = []
    for line in info:
        values.append(re.findall(r'[-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?', line))
    if "aveTemp" in header_line:
        aveTemp = "yes"
    else:
        aveTemp = None
    return header_line, values, aveTemp


def write_file(header_line: list, data: list, filename: Path):
    """
    This function saves the MDSTEPS information into a file with the .csv format.

    Parameters
    ----------
    header_line : list
        A list contains header variables.
    data : list
        A list of lists contains data in the MDSTEPS file, line by line.
    filename : Path
        The path to the .csv file of MD steps.

    Returns
    -------
    None.
    """

    with open(filename, "w", newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(header_line)
        csv_writer.writerows(data)


def plot_figure(data: list, flag: str):
    """
    This function deals with plotting. It will first convert the data into a numpy
    2D-array with the data type as the float. Then, a 2 by 2 figure will plot.

    Parameters
    ----------
    data : list
        A list of lists contains data in the MDSTEPS file, line by line.

    Returns
    -------
    None.
    """

    data = np.asfarray(data)
    time = data[:, 0]
    Etot = data[:, 1]
    Epot = data[:, 2]
    Ekin = data[:, 3]

    fig  = plt.figure(figsize=(8, 6))
    gs = GridSpec(3, 2)

    # Panel 1, Time vs Total Energy
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(time, Etot, '-k', label='Total Energy')
    ax1.set_xlim([0, time.max()])
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Total Energy (eV)')
    ax1.legend()

    # Panel 2, Time vs Potential Energy
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(time, Epot, '-b', label='Potential Energy')
    ax2.set_xlim([0, time.max()])
    ax2.set_xlabel('Time (fs)')
    ax2.set_ylabel('Potential Energy (eV)')
    ax2.legend()

    # Panel 3, Time vs Kinetic Energy
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(time, Ekin, '-r', label='Kinetic Energy')
    ax3.set_xlim([0, time.max()])
    ax3.set_xlabel('Time (fs)')
    ax3.set_ylabel('Kinetic Energy (eV)')
    ax3.legend()

    # Panel 4, Time vs All Energies
    ax4 = plt.subplot(gs[1, 1])
    ax4.scatter(time, Etot, alpha=0.2, label='Total Energy')
    ax4.plot(time, Epot, '-b', label='Potential Energy')
    ax4.plot(time, Ekin, '-r', label='Kinetic Energy')
    ax4.set_xlim([0, time.max()])
    ax4.set_xlabel('Time (fs)')
    ax4.set_ylabel('Energy (eV)')
    ax4.legend()

    # Panel 5, Time vs (average) Temperature
    ax5 = plt.subplot(gs[2, :])
    ax5.set_xlim([0, data[:, 0].max()])
    ax5.set_xlabel('Time (fs)')
    if flag:
        ax5.plot(time, data[:, 5], '-k', label='Average Temperature')
        ax5.set_ylim([0, data[:, 5].max()])
        ax5.set_ylabel('Average Temperature (K)')
    else:
        ax5.plot(time, data[:, 4], '-k', label='Temperature')
        ax5.set_ylim([0, data[:, 4].max()])
        ax5.set_ylabel('Temperature (K)')
    ax5.legend()

    plt.tight_layout()
    plt.show()


def main():
    """
    The workflow:
        1. Grab the information from MDSTEPS.
        2. Save the information in the .csv table format.
        3. Plot figure if necessary.

    Returns
    -------
    None.

    """
    input_file, output_file, plot_fig = args.filename, args.output, args.plot
    header_line, values, aveTemp = parse_MDSTEPS(input_file)
    write_file(header_line, values, output_file)

    if plot_fig.lower() in ["y", "yes"]:
        plot_figure(values, aveTemp)


if __name__ == "__main__":
    main()
