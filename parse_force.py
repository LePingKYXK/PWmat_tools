
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path


parser = ap.ArgumentParser(add_help=True,
                           formatter_class=ap.ArgumentDefaultsHelpFormatter,
                           description="""
                           Author:   Dr. Huan Wang,
                           Email:    huan.wang@whut.edu.cn,
                           Version:  v1.1,
                           Date:     October 13, 2024
                           Modified: October 15, 2024
                           """)
parser.add_argument("-f", "--intputfile",
                    metavar="<input filename>",
                    type=Path,
                    help="The MOVEMENT file",
                    default=Path.cwd() / "MOVEMENT"
                    )
parser.add_argument("-o", "--outputfile",
                    metavar="<output filename>",
                    type=Path,
                    help="The force file",
                    default=Path.cwd() / "force.csv"
                    )
parser.add_argument("-i", "--indices",
                    metavar="<indices of atoms>",
                    type=int,
                    nargs="+",
                    help="The indices of atoms to be analyzed",
                    default=0
                    )
parser.add_argument("-p", "--plot",
                    metavar="<plot figure>",
                    type=str,
                    choices=["xyz", "elements"],
                    help="Plot the force according to the xyz components or the elements.",
                    default=None
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


def check_indices(indices: int or list, num_atom: int) -> int or np.ndarray:
    """Check the indices."""

    try:
        if isinstance(indices, int) and indices == 0:
            print(f"All {num_atom} atoms are selected.")
            return num_atom
        elif isinstance(indices, list):
            if max(indices) > num_atom:
                raise ValueError(f"Error: The maximum index is {num_atom}, but {max(indices)} is given.")
            else:
                row_marks = np.array(indices)
                print(f"The indices of selected atoms are: {row_marks}")
                return row_marks
        else:
            raise ValueError("Error: Invalid input. Please provide either an integer 0 or a list of indices.")
    except ValueError as e:
        print(e)
        exit()


def parse_MOVEMENT_file(filename: Path, row_marks: np.ndarray) -> np.ndarray:
    """Parse the MOVEMENT file to obtain time and Force vector components (x, y, z)."""
    
    time_list = []
    data_list = []
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

    time = np.asfarray(time_list) 
    
    if isinstance(row_marks, int):
        data = np.array(data_list).reshape(time.size, row_marks, 3)
#        print(f"force table:\n{data}")
        return time, data
    else:
        data = np.array(data_list).reshape(time.size, -1, 3)[:,row_marks - 1,:]
#        print(f"force table:\n{data}")
        return time, data


def save_data(filename: Path, time: np.ndarray, force: np.ndarray) -> None:
    """Save the force on each atom."""
    num_selected_atom = force.shape[1]
    repeat_coordinates = "".join(("x, y, z", ", ")) * num_selected_atom
    head_line = ", ".join(("Time (fs)", repeat_coordinates[:-2]))
    
    merged = np.zeros((time.size, force.shape[1] * 3 + 1))
    merged[:,0] = time
    for i in range(force.shape[1]):
        merged[:, i*3+1:i*3+4] = force[:, i, :]
    np.savetxt(filename, merged, fmt="%.15f", delimiter=",",header=head_line)


def plot_force_by_xyz(time: np.ndarray, force: np.ndarray, row_marks: int or list) -> None:
    """Plot the force according to the xyz components."""
    
    labels = ("Force_x", "Force_y", "Force_z")

    if isinstance(row_marks, int):
        indices = np.arange(1, row_marks + 1)
            
    fig, axs = plt.subplots(3, 1, figsize=(8,  3 * force.shape[1]))
    for i in range(3):
        axs[i].set_xlabel("Time (fs)")
        for j in range(force.shape[1]):
            axs[i].plot(time, force[:,j,i].T, label="_".join(("Element", str(row_marks[j]))))
            axs[i].set_ylabel(labels[i])
            ax.legend()
    plt.tight_layout()
    plt.show()


def plot_force_by_elements(time: np.ndarray, force: np.ndarray, row_marks: int or list) -> None:
    """Plot the force by each element."""
    
    labels = ("Force_x", "Force_y", "Force_z")

    if isinstance(row_marks, int):
        indices = np.arange(1, row_marks + 1)
            
    fig, axs = plt.subplots(force.shape[1], 1, figsize=(8, 3 * force.shape[1]))
    for i in range(force.shape[1]):
        axs[i].set_xlabel("Time (fs)")
        for j in range(3):
            axs[i].plot(time, force[:,i,j].T, label=labels[j])
            axs[i].set_ylabel("_".join(("Element", str(row_marks[i]))))
            axs[i].legend()
    plt.tight_layout()
    plt.show()


def main():
    intputfile = args.intputfile
    indices = args.indices
    outputfile = args.outputfile
    plot = args.plot
    
    num_atom = number_of_atoms(intputfile)
    row_marks = check_indices(indices, num_atom)
    time, data = parse_MOVEMENT_file(intputfile, row_marks)
    save_data(outputfile, time, data)
    
    if plot == "xyz":
        plot_force_by_xyz(time, data, row_marks)
    elif plot == "elements":
        plot_force_by_elements(time, data, row_marks)


if "__main__" == __name__:
    main()
