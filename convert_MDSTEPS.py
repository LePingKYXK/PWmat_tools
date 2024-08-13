
import argparse as ap
import csv
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
                           Date:    August 12, 2024""")
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
                    help="option for plotting. y or yest for plotting, while n or no for not",
                    default=("yes"),
                    )
args = parser.parse_args()


def grab_info(filename: Path) -> list:
    """
    This function grab the information from the MDSTEPS file.

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
    """

    with open(filename, "r") as file:
        info = file.readlines()

    first_l = info[0]
    pattern = re.compile(r'\b(Iter|Etot|Ep|Ek|Temp|aveTemp|dE|dRho|SCF|dL|Fcheck)\b')
    header_line = re.findall(pattern, first_l)

    values = []
    for line in info:
        values.append(re.findall(r'[-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?', line))
    return header_line, values



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


def plot_figure(data: list):
    """
    This function deals with plotting. It will first convert the data into a numpy
    2D-array with the data type as float. Then, a 2 by 2 figure will plot.

    Parameters
    ----------
    data : list
        A list of lists contains data in the MDSTEPS file, line by line.

    Returns
    -------
    None.

    """

    data = np.asfarray(data)
    fig, axs = plt.subplots(2, 2, figsize=(16/2.54, 12/2.54))

    # Panel 1, Time vs Total Energy
    axs[0, 0].plot(data[:, 0], data[:, 1])
    axs[0, 0].set_xlim([0, data[:, 0].max()])
    axs[0, 0].set_xlabel('Time (fs)')
    axs[0, 0].set_ylabel('Total Energy (eV)')

    # Panel 2, Time vs Potential Energy
    axs[0, 1].plot(data[:, 0], data[:, 2])
    axs[0, 1].set_xlim([0, data[:, 0].max()])
    axs[0, 1].set_xlabel('Time (fs)')
    axs[0, 1].set_ylabel('Potential Energy (eV)')

    # Panel 3, Time vs Kinetic Energy
    axs[1, 0].plot(data[:, 0], data[:, 3])
    axs[1, 0].set_xlim([0, data[:, 0].max()])
    axs[1, 0].set_xlabel('Time (fs)')
    axs[1, 0].set_ylabel('Kinetic Energy (eV)')

    # Panel 4, Time vs Temperature
    axs[1, 1].plot(data[:, 0], data[:, 4])
    axs[1, 1].set_xlim([0, data[:, 0].max()])
    axs[1, 1].set_xlabel('Time (fs)')
    axs[1, 1].set_ylabel('Temperature (K)')

    plt.tight_layout()
    plt.show()


def main():
    """
    The work flow:
        1. Grab the information from MDSTEPS.
        2. Save the information as the .csv table format.
        3. Plot figure if necessary.

    Returns
    -------
    None.

    """
    input_file, output_file, plot_fig = args.filename, args.output, args.plot
    header_line, values = grab_info(input_file)
    write_file(header_line, values, output_file)

    if plot_fig.lower() in ["y", "yes"]:
        plot_figure(values)


if __name__ == "__main__":
    main()
