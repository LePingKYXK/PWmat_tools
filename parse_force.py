
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import re
import time
from itertools import product
from pathlib import Path
from periodic_table import Periodic_Table_dict


parser = ap.ArgumentParser(add_help=True,
                           formatter_class=ap.ArgumentDefaultsHelpFormatter,
                           description="""
                           Author:   Dr. Huan Wang,
                           Email:    huan.wang@whut.edu.cn,
                           Version:  v1.5,
                           Date:     October 13, 2024
                           Modified: October 22, 2024
                           """)
parser.add_argument("-f", "--intputfile",
                    metavar="<input filename>",
                    type=Path,
                    help="The MOVEMENT file",
                    default=Path.cwd() / "MOVEMENT"
                    )
parser.add_argument("-i", "--indices",
                    metavar="<indices of atoms>",
                    type=int,
                    nargs="+",
                    help="The indices of atoms to be analyzed",
                    #default=0
                    )
parser.add_argument("-p", "--plot",
                    metavar="<plot figure>",
                    type=str,
                    choices=["xyz", "elements"],
                    help="Plot the force according to the xyz components or the elements.",
                    default="elements"
                    )
args = parser.parse_args()


def number_of_atoms(filename: Path) -> int:
    """
    This function grabs and returns the number of atoms in the system.

    Parameters
    ----------
    filename : Path
        The MOVEMENT file.

    Returns
    -------
    num_atom : int
        The number of atoms in the system.

    """
    
    with open(filename, 'r') as fo:
        for line in fo:
            match_atom = re.search(r"(\d+)\s*atoms", line)
            if match_atom:
                break
    return int(match_atom.group(1))


def check_indices(indices: int or list, num_atom: int) -> np.ndarray:
    """
    This function validates whether the input indices exceed the range of the total atom count.

    Parameters
    ----------
    indices : int or list
        The custom input numbers (indices).
    num_atom : int
        The number of atoms in the system.

    Raises
    ------
    ValueError
        Warning, if the maximum indices exceed the number of atoms

    Returns
    -------
    indices : np.ndarray
        The 1D array contains the input indices.

    """
    
    try:
        if isinstance(indices, int) and indices == 0:
            print(f"All {num_atom} atoms are selected.")
            return np.array(list(range(1, num_atom + 1)))
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


def get_element_name(atomic_numbers: np.ndarray) -> list:
    """
    This function collects the element name from the data_array by retracing the periodic table of elements.

    Parameters
    ----------
    atomic_numbers : np.ndarray
        A 1D array contains the atomic numbers.

    Returns
    -------
    element_name : list[str]
        A list contains the selected elements.

    """
    
    element_name = []
    for atomic_number in atomic_numbers:
        for element, info in Periodic_Table_dict.items():
            if info[0] == atomic_number:
                element_name.append(element)
    return element_name


def parse_MOVEMENT_file(filename: Path, row_marks: np.ndarray) -> np.ndarray:
    """
    Parse the MOVEMENT file to obtain time and Force vector components (x, y, z).

    Parameters
    ----------
    filename : Path
        The MOVEMENT file.
    row_marks : np.ndarray
        The 1D array contains the indices of atoms to be analyzed.

    Returns
    -------
    time_array : np.ndarray
        A 1D array contains the simulation time in the unit of femtosecond.
    data_array : np.ndarray
        A 3D array contains the force components (Force_x, Force_y, Force_z).
    element_list : list[str]
        A list contains the symbols of selected elements.

    """
    
    time_list = []
    data_list = []
    table_data = []
    
    with open(filename, 'r') as fo:
        
        collecting = False
        current_line = 0
        print("The program is running...")
        
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
                table_data.append(line.strip().split())
                current_line += 1
                if isinstance(row_marks, int) and current_line == row_marks:
                    collecting = False
                elif isinstance(row_marks, np.ndarray) and current_line == row_marks.max():
                    collecting = False

    time_array = np.asfarray(time_list) 
    data_array = np.array(data_list).reshape(time_array.size, -1, 4)

    atomic_number_array = data_array[0,:,0][row_marks-1].astype(int)
    element_list = get_element_name(atomic_number_array)
    print(f"The element list is: {element_list}")

    if isinstance(row_marks, int):
        return time_array, data_array[:,:,1:], element_list
    else:
        return time_array, data_array[:,row_marks - 1,1:], element_list


def save_data(time_array: np.ndarray, force: np.ndarray, row_marks: np.ndarray, element_list: list, num_atom: int) -> None:
    """
    This function saves the force components of each selected atom along the simulation time to a CSV file.

    Parameters
    ----------
    time_array : np.ndarray
        A 1D array contains the simulation time in the unit of femtosecond.
    force : np.ndarray
        A 3D array contains the force components (Force_x, Force_y, Force_z).
    row_marks : np.ndarray
        A 1D array contains the input indices.
    element_list : list[str]
        A list contains the symbols of selected elements.
    num_atom : int
        The number of atoms in the system.

    Returns
    -------
    None
    
    """
    
    tags = ("Force_x", "Force_y", "Force_z")
    elements = [f"{e}_{i}" for e, i in zip(element_list, row_marks)]

    if len(element_list) <= 8:
        filename = ".".join(("_".join(elements), "csv"))
    elif len(element_list) == num_atom:
        filename = ".".join("all_elements", "csv")
    else:    # 8 < len(element_list) < num_atom:
        filename = ".".join(("_".join(set(element_list)), "csv"))
    
    labels = product(elements, tags)
    stings = ", ".join((f"{elem}_{force}" for elem, force in labels))
    head_line = ", ".join(("Time (fs)", stings))
    
    merged = np.zeros((time_array.size, force.shape[1] * 3 + 1))
    merged[:,0] = time_array
    
    for i in range(force.shape[1]):
        merged[:, i*3+1:i*3+4] = force[:, i, :]
    np.savetxt(filename, merged, fmt="%.15f", delimiter=", ", header=head_line)
    print(f"The force of selected atoms has saved to the '{filename}' file.")


def plot_force(flag: str, time_array: np.ndarray, force: np.ndarray, row_marks: int or list, element_list: list) -> None:
    """
    This function plots the force components according to the flags.

    Parameters
    ----------
    flag : str
        The conditions for plotting.
    time_array : np.ndarray
        A 1D array contains the simulation time in the unit of femtosecond.
    force : np.ndarray
        A 3D array contains the force components (Force_x, Force_y, Force_z).
    row_marks : np.ndarray
        A 1D array contains the input indices.
    element_list : list[str]
        A list contains the symbols of selected elements.

    Returns
    -------
    None

    """
    
    tags = ("Force$_x$", "Force$_y$", "Force$_z$")
    colors = ("dimgray", "tomato", "dodgerblue")

    elements = [f"{e}_{i}" for e, i in zip(element_list, row_marks)]

    if flag == "elements":
        fig, axs = plt.subplots(force.shape[1], 1, figsize=(8, 3 * force.shape[1]))
        for i in range(force.shape[1]):
            axs[i].set_xlabel("Time (fs)")
            axs[i].set_ylabel(elements[i])
            axs[i].set_xlim(time_array.min(), time_array.max())
            
            for j in range(3):
                label = "_".join((elements[i], tags[j]))
                axs[i].plot(time_array, force[:,i,j], color=colors[j], label=label, alpha=0.6)
                axs[i].legend()

    elif flag == "xyz":
        fig, axs = plt.subplots(3, 1, figsize=(8,  3 * force.shape[1]))
        for i in range(3):
            axs[i].set_xlabel("Time (fs)")
            axs[i].set_ylabel(tags[i])
            axs[i].set_xlim(time_array.min(), time_array.max())
            
            for j in range(force.shape[1]):
                label = "_".join((elements[j], str(row_marks[j]), tags[i]))
                axs[i].plot(time_array, force[:,j,i], color=colors[i], label=label, alpha=0.6)
                axs[i].legend()
    plt.tight_layout()
    plt.show()


def main():
    """
    Workflow:
        (1) grab the number of atoms in the system;
        (2) check whether the custom input numbers (indices) exceed the number of atoms;
        (3) parse the MOVEMENT file to obtain the simulation time, force components, and the element symbols;
        (4) save the force components based on the selected elements along the simulation time.
        (5) plot the force components of the selected elements vs the simulation time.

    Returns
    -------
    None

    """
    
    intputfile = args.intputfile
    indices = args.indices
    plot = args.plot

    drawline = "-" * 79
    print("".join(("\n", drawline)))
    start_time = time.time()

    num_atom = number_of_atoms(intputfile)
    print(f"Number of atoms in this system: {num_atom}")
    row_marks = check_indices(indices, num_atom)
    time_array, data_array, element_list = parse_MOVEMENT_file(intputfile, row_marks)
    
    save_data(time_array, data_array, row_marks, element_list, num_atom)
    print(f"Used Time: {time.time() - start_time:.2f} seconds.")
    print("".join((drawline, "\n")))
    
    try:
        if len(element_list) <= 8:
            plot_force(plot, time_array, np.asfarray(data_array), row_marks, element_list)
        else:
            raise ValueError("Too many elements to plot.")
    except ValueError as e:
        print(e)
        exit()


if "__main__" == __name__:
    main()
