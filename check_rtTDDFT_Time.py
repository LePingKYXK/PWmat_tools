#/usr/bin python3

import argparse as ap
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as pc
from scipy.integrate import quad
from pathlib import Path


pc.c        # speed of light = 299792458.0 m s^(-1)
pc.e        # charge of electron = 1.602176634e-19 C
pc.hbar     # Plancck constant / 2pi = 1.0545718176461565e-34 J s
pc.femto    # femto = 10^(-15)


parser = ap.ArgumentParser(add_help=True,
                           formatter_class=ap.ArgumentDefaultsHelpFormatter,
                           description="""
                           Author:   Dr. Huan Wang,
                           Email:    huan.wang@whut.edu.cn,
                           Version:  v1.0,
                           Created:  August 08, 2024""")
parser.add_argument("-f", "--filename",
                    metavar="<The TDDFT input>",
                    type=Path,
                    help="The TDDFT etot.input file, you can specify the file name",
                    default=(Path.cwd().joinpath("etot.input")),
                    )
parser.add_argument("-p", "--plot",
                    metavar="<plot graph>",
                    type=str,
                    help="plot graph",
                    default="yes",
                    )
args = parser.parse_args()


def read_info(filename: Path) -> :
    """
    This function reads the etot.input (or REPORT) file and parses the TDDFT_TIME and MD_DETAIL parameters.

    Parameters
    ----------
    filename : Path
        The Path to the TDDFT input file, i.e. the etot.input file (or REPORT file).

    Returns
    -------
    values : list
        A list contains TDDFT_TIME parameters: itemtype, n, b1, b2, b3, b4, b5.
    """

    with open(filename, "r") as fo:
        for line in fo:
            matches_td = re.search(r'TDDFT_TIME\s*=\s*([\d., ]+)', line, re.IGNORECASE)
            matches_md = re.search(r'MD_DETAIL\s*=\s*([\d., ]+)', line, re.IGNORECASE)

            if matches_td:
                td_str = re.findall(r"\d+\.*\d*", matches_td.group(1))
                td_val = [float(num) for num in td_str]

            if matches_md:
                md_str = re.findall(r"\d+\.*\d*", matches_md.group(1))
                md_val = [float(num) for num in md_str]
    return td_val, md_val


def integrate_tddft_time(t, b1, b2, b3, b4, b5):
    return b1 * np.exp(-np.square((t - b2) / b3)) * np.sin(b4 * t + b5)


def tddft(data_td: list, data_md: list) -> np.array:
    """
    This function calculates the time dependent laser energy (in unit of eV),
    t vs eV.
    The formula is based on the Manual of PWmat in the rt-TDDFT section.

    Parameters
    ----------
    data_td : list
        A list contains TDDFT_TIME parameters.
    data_md : list
        A list contains MD_DETIAL parameters.

    Returns
    -------
    data : Numpy 2D-array.
        A Numpy 2D-array contains time and laser energy.
    """

    td_type, n, b1, b2, b3, b4, b5 = data_td
    md_type, md_step, del_t, T1, T2 = data_md

    time_arr = np.linspace(0, int(md_step) * del_t, int(md_step) + 1)

    if td_type == 2:
        tddft = b1 * np.exp(-np.square((time_arr - b2) / b3)) * np.sin(b4 * time_arr + b5)
        data = np.column_stack((time_arr, tddft))

    elif td_type == 22:
        #t_values = np.arange(0, md_step + del_t, del_t)
        tddft = np.zeros_like(time_arr)

        for i, t in enumerate(time_arr):
            tddft[i], _ = quad(integrate_tddft_time, t, t + del_t, args=(b1, b2, b3, b4, b5))
        data = np.column_stack((time_arr, tddft))
#    print(f"td: {tddft}\nsize:{tddft.size}")

    np.savetxt(Path.cwd() / "tddft_time.txt", data,
               fmt=("%10.6e", "%10.8e"), delimiter='\t', header='Time\tTDDFT(t)')
    return data


def plot_figure(data: np.array):
    """
    This function plots a figure of the laser pulse.

    Parameters
    ----------
    data : Numpy 2D-array.
        A Numpy 2D-array contains time and laser energy.

    Returns
    -------
    None.
    """

    fig, ax = plt.subplots(figsize=(16/2.54, 9/2.54))
    ax.plot(data[:, 0], data[:, 1], "-")
    ax.set_xlim([0, data[:, 0].max()])
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('laser (eV)')
    plt.show()


def main():
    """
    Work flow:
      (1) Read the TDDFT_TIME and MD_DETAIL parameters from the etot.input file.
      (2) Calculate the laser profile based on the step (1).
      (3) Plot figure if necessary.

    Returns
    -------
    None.
    """

    etot_input, plot_fig = args.filename, args.plot
    data_td, data_md = read_info(etot_input)
    print(f"\nTDDFT_Time: {data_td},\nMD_Detail: {data_md}\n")
    itemtype, n, b1, b2, b3, b4, b5 = data_td

    td = tddft(data_td, data_md)

    E_laser      = pc.hbar * b4 / pc.femto / pc.e  # in unit of eV
    lambda_laser = 2 * np.pi * pc.c * pc.giga * pc.femto / b4  # in unit of nm
    FWHM = b3 * np.sqrt(np.log(4)) * np.sqrt(2)

    draw_line = "".join(("\n", "-" * 79, "\n"))
    print(draw_line)
    print(f"The energy of laser is: {E_laser} eV")
    print(f"The wavelength of laser is: {lambda_laser} nm")
    print(f"The FWHM of laser is: {FWHM} nm")
    print(draw_line)

    if plot_fig.lower() in ["y", "yes"]:
        plot_figure(td)

  
if __name__ == "__main__":
    main()









