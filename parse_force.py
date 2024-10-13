
import numpy as np
import re


def parse_MOVEMENT_file(filename: Path, row_marks: np.ndarray) -> np.ndarray:
    """Parse the MOVEMENT file to obtain time and Force vector components (x, y, z)."""

    data_list = []
    time_list = []
    table_data = []
    collecting = False
    
    with open(filename, 'r') as fo:
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
  
    return time_list, data_list


def main():
    parse_MOVEMENT_file(filename)


if "__main__" == __name__:
    main()
