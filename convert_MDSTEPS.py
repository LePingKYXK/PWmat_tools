
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
                           Version:  v1.4,
                           Date:     August 12, 2024
                           Modified: August 30, 2024""")
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


def plot_variable_vs_time(x, y, color, label, ax):
    """
    Helper function to plot Total Energy, Potential Energy, and Kinetic vs time.
    """
    ax.plot(x, y, color, label=label)
    ax.set_xlim([0, x.max()])
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel(f'{label} (eV)')
    ax.legend()


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
    
    if not data:
        raise ValueError("Data list is empty.")

    data = np.asfarray(data)
    time = data[:, 0]
    label1 = ['Total Energy', 'Potential Energy', 'Kinetic Energy']
    label2 = ['$\\Delta$ Total Energy', '$\\Delta$ Potential Energy', '$\\Delta$ Kinetic Energy']
    colors = ['-k', '-b', '-r']

    fig  = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 2)

    # Panel 1 to 3 (Total Energy, Potential Energy, and Kinetic vs time)
    for i, (label, color) in enumerate(zip(label1, colors)):
        ax = plt.subplot(gs[i//2, i%2])
        plot_variable_vs_time(time, data[:, i+1], color, label, ax)

    # Panel 4, Delta Energies vs Times
    ax4 = plt.subplot(gs[1, 1])
    for i, (label, color) in enumerate(zip(label2, colors)):
#        ax4.plot(time, data[:, i+1], color, label=label)
        ax4.plot(time[1:], np.diff(data[:, i+1]), color, label=label)
    ax4.set_xlim([0, time.max()])
    ax4.set_xlabel('Time (fs)')
#    ax4.set_ylabel('Energy (eV)')
    ax4.set_ylabel('$\\Delta$ Energy (eV)')
    ax4.legend()

    # Panel 5and 6, (average) Temperature vs Time and SCF loops vs Time
    for i, (tag, label) in enumerate([(5 if flag else 4, 'Average Temperature' if flag else 'Temperature'),
                                      (8 if flag else 7, 'SCF loops')]):
        ax = fig.add_subplot(gs[2+i, :])
        plot_variable_vs_time(time, data[:, tag], '-k', label, ax)

    fig.tight_layout()
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
