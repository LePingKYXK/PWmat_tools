#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse as ap
import numpy as np
import re
import time
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Union, Dict, Tuple
try:
    from pymatgen.core import Element
except ImportError:
    raise ImportError("pymatgen is required. Install via: pip install pymatgen")


class MyFormatter(ap.RawDescriptionHelpFormatter,
                  ap.ArgumentDefaultsHelpFormatter):
    pass

def parse_arguments():
    """Parsing arguments"""
    parser = ap.ArgumentParser(formatter_class=MyFormatter,
                               description=textwrap.dedent(
    """A robust parser for PWmat MOVEMENT file.
    Author:
        Dr. Huan Wang <huan.wang@whut.edu.cn>
    """))
    parser.add_argument(
        '-f', '--file',
        type=Path,
        default=Path.cwd() / "MOVEMENT",
        help='Path to the MOVEMENT file.'
    )
    parser.add_argument(
        '-id', '--indices',
        type=int,
        nargs='+',
        help='List of atom indices to load (0-based). Example: -id 0 5 10'
    )
    parser.add_argument(
        '-e', '--elements',
        type=str,
        nargs='+',
        help='List of element symbols to load. Example: -e C Si'
    )
    parser.add_argument(
        '-mf', '--max-frames',
        type=int,
        help='Maximum number of frames to read.'
    )
    return parser.parse_args()


@dataclass
class MovementData:
    """
    Stores the parsed data from the MOVEMENT file.
    """
    num_atoms: int          # integer for storing the total number of atoms
    n_frames: int           # integer for storing the total number of frames
    iter_time: np.ndarray   # 1d array (n_frames,) for storing the iteration time of each frame
    lattice: np.ndarray     # 3d array (n_frames, 3, 3) for storing the lattice vectors of each frame
    elements: np.ndarray    # 1d array (n_selected_atoms,) for storing the element symbols of selected atoms
    position: np.ndarray    # 3d array (n_frames, n_selected_atoms, 3) for storing the fractional coordinate of each atom in each frame
    coordinate: np.ndarray  # 3d array (n_frames, n_selected_atoms, 3) for storing the Cartesian coordinate of each atom in each frame
    force: np.ndarray       # 3d array (n_frames, n_selected_atoms, 3) for storing the force of each atom in each frame
    velocity: np.ndarray    # 3d array (n_frames, n_selected_atoms, 3) for storing the velocity of each atom in each frame

    @classmethod
    def parse_file(
        cls,
        file_path: Path,
        atom_indices: Optional[List[int]] = None,
        element_filter: Optional[Union[str, List[str]]] = None,
        max_frames: Optional[int] = None,
    ) -> 'MovementData':
        """
        Parsing data from the MOVEMENT file.

        Arguments:
            file_path (Path):  path to the MOVEMENT file.
            atom_indices (Optional[List[int]]): list of indices of atoms to load.
            element_filter (Optional[Union[str, List[str]]]): list of element symbols to load.  
            max_frames (Optional[int]): The maximum number of frames to read.

        Returns:
            MovementData: Object containing the parsed data.
        """
        if atom_indices is not None and element_filter is not None:
            raise ValueError("Cannot specify both 'atom_indices' and 'element_filter'.")

        start_time = time.time()
        print(f"Parsing {file_path} ...")

        # Precompile regular expressions
        patterns = {
            'header': re.compile(r'^\s*(\d+)\s+atoms.*?Iteration[^=]*=\s*([0-9.E+-]+)', re.IGNORECASE),
            'lattice': re.compile(r'^\s*Lattice vector\b.*', re.IGNORECASE),
            'position': re.compile(r'^\s*Position', re.IGNORECASE),
            'force': re.compile(r'^\s*-?Force', re.IGNORECASE),
            'velocity': re.compile(r'^\s*Velocity', re.IGNORECASE),
        }

        with open(file_path, 'r') as f:
            # --- 1. Initialization ---
            num_atoms_total = None
            atomic_numbers_first_frame = []
            selected_atom_mask = None
            
            iter_times = []
            lattices = []
            positions = []
            forces = []
            velocities = []
            
            frame_count = 0
            state = 'HEADER'  # The initial label

            # --- 2. The first scan  ---
            for line in f:
                line = line.strip()
                if not line: continue

                # HEADER Label：Find the start of the new frame
                if state == 'HEADER':
                    match = patterns['header'].match(line)
                    if match:
                        if num_atoms_total is None:
                            # The first frame: determine the total number of atoms and select which atoms to load
                            num_atoms_total = int(match.group(1))

                            # Determine the atom selection mask
                            if atom_indices is not None:
                                if any(i < 0 or i >= num_atoms_total for i in atom_indices):
                                    raise ValueError(f"Index {max(i for i in atom_indices if i >= num_atoms_total)} is out of range [0, {num_atoms_total - 1}]")
                                selected_atom_mask = np.zeros(num_atoms_total, dtype=bool)
                                selected_atom_mask[atom_indices] = True
                                
                            elif element_filter is not None:
                                if isinstance(element_filter, str):
                                    element_filter = [element_filter]
                                element_filter = set(e.upper() for e in element_filter)
                                # read the atomic information in the next ‘Position’ block to construct the mask
                                selected_atom_mask = np.zeros(num_atoms_total, dtype=bool)
                                
                        else:
                            # The following frames: verify the identity of atom number
                            if int(match.group(1)) != num_atoms_total:
                                raise ValueError(f"Inconsistent atom count at frame {frame_count + 1}: expected {num_atoms_total}, got {int(match.group(1))}")
                        
                        iter_times.append(float(match.group(2)))
                        state = 'LATTICE'
                        continue

                # Lattice state: read the lattice vectors of the current frame
                elif state == 'LATTICE':
                    if patterns['lattice'].search(line):
                        lattice_frame = []
                        for _ in range(3):
                            vec_line = f.readline().strip()
                            if not vec_line:
                                raise ValueError("Incomplete Lattice block.")
                            lattice_frame.append([float(x) for x in vec_line.split()])
                        lattices.append(lattice_frame)
                        state = 'POSITION'
                        continue

                # Position Lable: read the fractional coordinates of the selected atoms in the current frame                elif state == 'POSITION':
                elif state == 'POSITION':
                    if patterns['position'].search(line):
                        pos_frame = []
                        for i in range(num_atoms_total):
                            pos_line = f.readline().strip()
                            if not pos_line:
                                raise ValueError("Incomplete Position block.")
                            parts = [float(x) for x in pos_line.split()]
                            
                            atomic_num = int(round(parts[0]))
                            coords = parts[1:4]
                        
                            # Collect atomic numbers at the first frame for later filtering
                            if frame_count == 0:
                                atomic_numbers_first_frame.append(atomic_num)

                            # Check if the current atom is selected
                            if selected_atom_mask is not None:
                                if frame_count == 0:
                                    #The first frame: determine the final selection mask
                                    if atom_indices is not None:
                                        pass
                                    elif element_filter is not None: 
                                        symbol = Element.from_Z(atomic_num).symbol.upper()
                                        selected_atom_mask[i] = symbol in element_filter
                                
                                # Determine whether to save based on the mask
                                if selected_atom_mask[i]:
                                    pos_frame.append(coords)
                            else:
                                # No filter criteria specified; save all coordinates
                                pos_frame.append(coords)
                        
                        positions.append(pos_frame)
                        state = 'FORCE'
                        continue

                # Force Lable: read the forces of the selected atoms in the current frame
                elif state == 'FORCE':
                    if patterns['force'].search(line):
                        force_frame = []
                        for i in range(num_atoms_total):
                            force_line = f.readline().strip()
                            if not force_line:
                                raise ValueError("Incomplete Force block.")
                            parts = [float(x) for x in force_line.split()]
                            # parts[0] is atom index, parts[1:4] are forces
                            if selected_atom_mask is None or selected_atom_mask[i]:
                                force_frame.append(parts[1:4])
                        forces.append(force_frame)
                        state = 'VELOCITY'
                        continue

                # Velocity Lable: read the velocities of the selected atoms in the current frame
                elif state == 'VELOCITY':
                    if patterns['velocity'].search(line):
                        vel_frame = []
                        for i in range(num_atoms_total):
                            vel_line = f.readline().strip()
                            if not vel_line:
                                raise ValueError("Incomplete Velocity block.")
                            parts = [float(x) for x in vel_line.split()]
                            # parts[0] is atom index, parts[1:4] are velocities
                            if selected_atom_mask is None or selected_atom_mask[i]:
                                vel_frame.append(parts[1:4])
                        velocities.append(vel_frame)
                        
                        # The current frame has completed; preparing for the next frame
                        frame_count += 1
                        if max_frames is not None and frame_count >= max_frames:
                            print(f"Stopped at frame {max_frames}.")
                            break
                        state = 'HEADER' # seek to the beginning of a new frame
                        continue
            
            # --- 3. Checking file completeness ---
            # expected_blocks_per_frame = 4  # lattice, position, force, velocity
            if len(lattices) != len(positions) or len(positions) != len(forces) or len(forces) != len(velocities):
                raise ValueError(f"File is incomplete. Mismatched number of data blocks: "
                                 f"Lattice={len(lattices)}, Position={len(positions)}, "
                                 f"Force={len(forces)}, Velocity={len(velocities)}.")
            
            if len(iter_times) != len(lattices):
                 raise ValueError(f"File is incomplete. Number of headers ({len(iter_times)}) does not match number of data blocks ({len(lattices)}).")
            
            if frame_count == 0:
                raise ValueError("No complete frames were found in the file.")

            # --- 4. Convert data to NumPy arrays ---
            n_selected_atoms = len(positions[0]) if positions else 0
            
            iter_time_arr = np.array(iter_times)
            lattice_arr = np.array(lattices, dtype=np.float64)
            position_arr = np.array(positions, dtype=np.float64)
            force_arr = np.array(forces, dtype=np.float64)
            velocity_arr = np.array(velocities, dtype=np.float64)

            # --- 5. Calculate Cartesian coordinates ---
            coordinate_arr = np.einsum('fij,faj->fai', lattice_arr, position_arr)

            # --- 6. Convert Atomic Number (zahl) to Element Symbol ---
            if selected_atom_mask is not None:
                temp_symbols_all = [Element.from_Z(zahl).symbol for zahl in atomic_numbers_first_frame]
                elements_arr = np.array(temp_symbols_all)[selected_atom_mask]
            else:
                temp_symbols_all = [Element.from_Z(zahl).symbol for zahl in atomic_numbers_first_frame]
                elements_arr = np.array(temp_symbols_all)

        n_frames_final = len(iter_time_arr)
        print(f"Successfully loaded {n_frames_final} frames, {n_selected_atoms} selected atoms.")
        
        parsing_duration = time.time() - start_time
        print(f"File parsing completed in {parsing_duration:.2f} seconds.")

        return cls(
            num_atoms=num_atoms_total,
            n_frames=n_frames_final,
            iter_time=iter_time_arr,
            lattice=lattice_arr,
            elements=elements_arr,
            position=position_arr,
            coordinate=coordinate_arr,
            force=force_arr,
            velocity=velocity_arr,
        )


def main_cli():
    """Command Line Interface entry point."""
    args = parse_arguments()
    
    try:
        data = MovementData.parse_file(
            file_path=args.file,
            atom_indices=args.indices,
            element_filter=args.elements,
            max_frames=args.max_frames
        )
        
        print("\n--- Data Summary ---")
        print(f"Total Atoms in File: {data.num_atoms}")
        print(f"Selected Atoms: {len(data.elements)}")
        print(f"Frames Loaded: {data.n_frames}")
        print(f"Lattice shape: {data.lattice.shape}")
        
        element_counts = Counter(data.elements)
        elements_summary = ", ".join([f"{elem}: {count}" for elem, count in sorted(element_counts.items())])
        print(f"Elements: {elements_summary}")
        
        print(f"Position shape: {data.position.shape}")
        print(f"Coordinate shape: {data.coordinate.shape}")
        print(f"Force shape: {data.force.shape}")
        print(f"Velocity shape: {data.velocity.shape}")
        if data.n_frames > 0 and len(data.elements) > 0:
            print(f"First frame, first atom coordinates: {data.coordinate[0, 0, :]}")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main_cli()
