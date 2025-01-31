
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path


parser = ap.ArgumentParser(add_help=True,
                           formatter_class=ap.ArgumentDefaultsHelpFormatter,
                           description="""
                           Author:   Dr. Huan Wang,
                           Email:    huan.wang@whut.edu.cn,
                           Version:  v1.4,
                           Date:     December 08, 2024,
                           Modified: December 10, 2024""")
parser.add_argument("-f1", "--inputf1",
                    #metavar="<itype_time>",
                    type=Path,
                    required=True,
                    help="The TDOS file",
                    default="TDOS.totalspin_projected_ShiftFermi"
                    )
parser.add_argument("-f2", "--inputf2",
                    #metavar="<wavelength>",
                    type=Path,
                    required=True,
                    help="The occ. electronDOS file",
                    default="occDOS.totalspin_projected_ShiftFermi"
                    )
parser.add_argument("-o", "--output",
                    #metavar="<wavelength>",
                    type=Path,
                    required=True,
                    help="The output DOS file, containing the total, electron, and hole",
                    )
args = parser.parse_args()


def read_DOS_file(filename: Path) -> np.ndarray:
    """
    This function reads the tr-TDDFT DOS file and returns the data and header.

    Parameters
    ----------
    filename : Path
        The path of the tr-TDDFT DOS file.

    Returns
    -------
    data : np.ndarray
        The data of the tr-TDDFT DOS file.
    header : str
        The header for describing the tr-TDDFT DOS data.
    """
    data = np.genfromtxt(filename, comments='#')

    with open(filename, 'r') as fo:
        for line in fo:
            match = re.match(r'\s+#', line)
            if match:
                header = line.strip().split()
                break
    id = header.index("Energy")

    if "TDOS" in filename.name:
        header[id] = "TDOS"

    elif "occDOS" in filename.name:
        header[id] = "occDOS"

    header = ",".join(header[1:])
    return data, header


def prepare_header(header1: str, header2: str) -> str:
    """
    This function prepares the header for the combined DOS file.

    Parameters
    ----------
    header1 : str
        The header for the TDOS file.
    header2 : str
        The header for the occDOS file.

    Returns
    -------
    new_header : str
        The combined header: TDOS, occDOS, and holeDOS.
    """
    header3 = header1.replace("TDOS", "holeDOS")
    new_header = ",".join([header1, header2, header3])
    return new_header


def subtraction(TDOS: np.ndarray, occDOS: np.ndarray) -> np.ndarray:
    return TDOS - occDOS


def save_data(filename: Path, data: np.ndarray, header: str) -> None:
    np.savetxt(filename, data, delimiter=',', header=header)


def plotDOS(data: np.ndarray, header: str) -> None:
    """
    This function plots the tr-TDDFT DOS data.

        1. The total DOS, electron DOS, and hole DOS are plotted separately.
        2. The x-axis is the DOS, and the y-axis is the energy difference between the energy of each DOS point and the Fermi energy (E-E_f).
        3. The title of each subplot is the name of the corresponding spin state.
        4. The legend of each subplot shows the tr-TDDFTDOS type.

    Parameters
    ----------
    data : np.ndarray
        The combined DOS data.
    header : str
        The header for the combined DOS data.
    """
    N = data.shape[1] // 3
    titles = header.split(",")[1:]
    
    fig, axes = plt.subplots(1, N - 1, figsize=(N * 2, 5))

    for i in range(N - 1):
        x_columns = [1, N + i + 1, 2 * N + i + 1]
        for j in range(3):
            labels = ["total DOS", "elec DOS", "hole DOS"]
            colors = ['gray', 'r', 'b']
            axes[i].plot(data[:, x_columns[j]], data[:, 0], label=labels[j], c=colors[j])
        axes[i].set_title(f'DOS {titles[i]}')
        axes[i].set_xlabel('DOS')
        axes[i].set_ylabel('$E-E_f$ (eV)')
        axes[i].legend(loc='lower right', fontsize='small')

    plt.tight_layout()
    plt.show()


def main():
    """
    Workflow:
    1. Read the TDOS and occDOS files.
    2. Modify the occDOS array to remove negative values.
    3. Combine the TDOS and occDOS arrays.
    4. Modify the holeDOS array to remove positive values.
    5. Combine the Total DOS, occ DOS, and hole DOS arrays.
    6. Prepare the header for the combined DOS file.
    7. Save the combined DOS data and header.
    8. Plot the DOS data according to the header.
    """
    tdos_file   = args.inputf1
    occdos_file = args.inputf2
    WARNING = "Warning! The first column data in the two files are not equal."

    TDOS_array, header_TDOS     = read_DOS_file(tdos_file)
    occDOS_array, header_occDOS = read_DOS_file(occdos_file)
    modify_occDOS_array = occDOS_array.copy()
    rows_to_replace_occ = occDOS_array[:, 0] < 0
    modify_occDOS_array[rows_to_replace_occ] = 0
    print(f"The modified occDOS:\n{modify_occDOS_array}")

    try:
        combs = np.concatenate((TDOS_array, modify_occDOS_array), axis=1)
        print(f"The combined array:\n{combs}")
    except ValueError as e:
        print("Unable to combine arrays:", e)
        exit()
    else:
        if np.array_equal(TDOS_array[:, 0], occDOS_array[:, 0]):
            diff_array = subtraction(TDOS_array[:, 1:], occDOS_array[:, 1:])
            holeDOS_array = np.concatenate((TDOS_array[:,[0]], diff_array), axis=1)
            modify_holeDOS_array = holeDOS_array.copy()
            rows_to_replace_hole = holeDOS_array[:, 0] > 0
            modify_holeDOS_array[rows_to_replace_hole] = 0
            print(f"The modified holeDOS:\n{modify_holeDOS_array}")

            # combine the Total DOS, occ DOS, and hole DOS
            results = np.concatenate((combs, modify_holeDOS_array), axis=1)
            
        else:
            print(WARNING)
            exit()
    header = prepare_header(header_TDOS, header_occDOS)
    save_data(args.output, results, header)
    plotDOS(results, header_TDOS)

            
if __name__ == '__main__':
    main()
