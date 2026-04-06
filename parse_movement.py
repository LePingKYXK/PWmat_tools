#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
High-performance parser for PWmat MOVEMENT trajectory files.
Optimized for speed, supports selective atom loading and frame range.
No redundant file reopening.
"""

import argparse as ap
import numpy as np
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
try:
    from pymatgen.core import Element
except ImportError:
    raise ImportError("pymatgen is required. Install via: pip install pymatgen")

class MyFormatter(ap.RawDescriptionHelpFormatter, ap.ArgumentDefaultsHelpFormatter):
    pass


def parse_arguments():
    parser = ap.ArgumentParser(formatter_class=MyFormatter,
                               description="High-performance MOVEMENT file parser")
    parser.add_argument("-f", "--file", 
                        type=Path, 
                        default=Path.cwd() / "MOVEMENT",
                        help="Path to MOVEMENT file")
    parser.add_argument("-id", "--indices", 
                        type=int, nargs='+',
                        help="Space-separated 0-based atom indices to extract")
    parser.add_argument("-e", "--elements", 
                        type=str, nargs='+',
                        help="Element symbols to extract (e.g., C Si)")
    parser.add_argument("-sf", "--start-frame", 
                        type=int, default=0,
                        help="First frame index (0-based, inclusive, e.g., 30)")
    parser.add_argument("-ef", "--end-frame", 
                        type=int, default=None,
                        help="Last frame index (e.g. 200)")
    return parser.parse_args()


@dataclass
class MovementData:
    num_atoms: int                 # total atoms per frame
    n_frames: int                  # number of frames loaded
    iter_time: np.ndarray          # (n_frames,) iteration times
    lattice: np.ndarray            # (n_frames, 3, 3)
    elements: np.ndarray           # (n_selected,) element symbols
    position: np.ndarray           # fractional, (n_frames, n_selected, 3)
    coordinate: np.ndarray         # Cartesian, (n_frames, n_selected, 3)
    force: np.ndarray              # (n_frames, n_selected, 3)
    velocity: np.ndarray           # (n_frames, n_selected, 3)
    selected_indices: List[int]    # 0‑based indices of selected atoms


class MovementParser:
    # Pre-compiled regex patterns (class-level to avoid recompilation)
    _PATTERNS = {
        'header': re.compile(r'^\s*(\d+)\s+atoms.*?Iteration[^=]*=\s*([0-9.E+-]+)', re.IGNORECASE),
        'lattice': re.compile(r'^\s*Lattice vector\b', re.IGNORECASE),
        'position': re.compile(r'^\s*Position\b', re.IGNORECASE),
        'nonperiodic': re.compile(r'^\s*nonperiodic_Position\b', re.IGNORECASE),
        'force': re.compile(r'^\s*-?Force\b', re.IGNORECASE),
        'velocity': re.compile(r'^\s*Velocity\b', re.IGNORECASE),
        'separator': re.compile(r'^\s*-{3,}\s*$'),
    }

    @classmethod
    def parse(
        cls,
        filepath: Path,
        atom_indices: Optional[List[int]] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        element_filter: Optional[Union[str, List[str]]] = None,
    ) -> MovementData:
        """
        Parse MOVEMENT file with optional atom/frame filtering.

        Args:
            filepath: Path to MOVEMENT file.
            atom_indices: 0‑based indices to extract (duplicates removed, sorted).
            start_frame: First frame to read (0‑based, inclusive).
            end_frame: Last frame to read. None means read all.
            element_filter: Element symbol or list of symbols; mutually exclusive with atom_indices.

        Returns:
            MovementData object.
        """
        if atom_indices is not None and element_filter is not None:
            raise ValueError("Cannot specify both atom_indices and element_filter")

        start_time = time.perf_counter()
        print(f"Parsing {filepath} ...")

        # Process atom indices: remove duplicates and sort
        if atom_indices is not None:
            atom_indices = sorted(set(atom_indices))
            need_full_first_frame = False
            selected_indices = atom_indices
        elif element_filter is not None:
            need_full_first_frame = True
            selected_indices = None
        else:
            # No filter: read all atoms
            selected_indices = None
            need_full_first_frame = False

        # Data storage
        iter_times = []
        lattices = []
        positions = []
        forces = []
        velocities = []

        num_atoms_total = None
        atomic_numbers_all = []   # store atomic numbers of all atoms (only if needed)
        selected_atomic_numbers = []  # atomic numbers of selected atoms (for -id case)
        frame_abs = 0
        current_frame = None
        max_frame_to_read = None if end_frame is None else end_frame - 1

        try:
            with open(filepath, 'r') as f:
                line = f.readline()
                while line:
                    stripped = line.strip()
                    if not stripped:
                        line = f.readline()
                        continue

                    # ---------- Frame header ----------
                    m = cls._PATTERNS['header'].match(stripped)
                    if m:
                        if num_atoms_total is None:
                            num_atoms_total = int(m.group(1))
                            # Determine selection strategy
                            if need_full_first_frame:
                                # We will read full first frame to get atomic numbers
                                pass
                            elif selected_indices is None:
                                # No filter: select all atoms
                                selected_indices = list(range(num_atoms_total))
                            else:
                                # Validate indices
                                selected_indices = [i for i in selected_indices if 0 <= i < num_atoms_total]
                                if not selected_indices:
                                    raise ValueError("No valid atom indices after filtering")
                        else:
                            # Verify atom count consistency
                            if int(m.group(1)) != num_atoms_total:
                                raise ValueError(f"Atom count mismatch at frame {frame_abs+1}")

                        iter_time = float(m.group(2))
                        current_frame = {
                            'iter_time': iter_time,
                            'lattice': None,
                            'position': None,
                            'force': None,
                            'velocity': None,
                        }

                        # Check if this frame is within requested range
                        if start_frame <= frame_abs <= (max_frame_to_read if max_frame_to_read is not None else float('inf')):
                            iter_times.append(iter_time)
                        else:
                            current_frame = None

                        line = f.readline()
                        continue

                    # ---------- Lattice block ----------
                    if cls._PATTERNS['lattice'].match(stripped):
                        if current_frame is not None:
                            lattice = []
                            for _ in range(3):
                                vec_line = f.readline()
                                if not vec_line:
                                    raise EOFError("Unexpected EOF while reading lattice")
                                lattice.append(list(map(float, vec_line.split())))
                            current_frame['lattice'] = np.array(lattice)
                        else:
                            # Skip lattice lines
                            for _ in range(3):
                                f.readline()
                        line = f.readline()
                        continue

                    # ---------- Position block ----------
                    if cls._PATTERNS['position'].match(stripped):
                        if current_frame is not None:
                            if need_full_first_frame and not atomic_numbers_all:
                                # First frame: read all atoms to get atomic numbers and positions
                                pos_all, atomic_numbers_all = cls._read_full_position_block(f, num_atoms_total)
                                # Determine selected indices based on element filter
                                selected_indices = cls._select_indices_by_elements(atomic_numbers_all, element_filter)
                                # Store atomic numbers of selected atoms for later
                                selected_atomic_numbers = [atomic_numbers_all[i] for i in selected_indices]
                                # Extract selected positions
                                current_frame['position'] = pos_all[selected_indices]
                            else:
                                # Read only selected atoms, and also capture atomic numbers if this is the first frame
                                if frame_abs == 0 and not selected_atomic_numbers:
                                    pos_data, atomic_nums = cls._read_selected_block(
                                        f, num_atoms_total, selected_indices, data_cols=3, skip_cols=1,
                                        return_atomic_numbers=True
                                    )
                                    current_frame['position'] = pos_data
                                    selected_atomic_numbers = atomic_nums
                                else:
                                    current_frame['position'] = cls._read_selected_block(
                                        f, num_atoms_total, selected_indices, data_cols=3, skip_cols=1,
                                        return_atomic_numbers=False
                                    )
                        else:
                            # Skip entire position block
                            for _ in range(num_atoms_total):
                                f.readline()
                        line = f.readline()
                        continue

                    # ---------- nonperiodic_Position (skip) ----------
                    if cls._PATTERNS['nonperiodic'].match(stripped):
                        for _ in range(num_atoms_total):
                            f.readline()
                        line = f.readline()
                        continue

                    # ---------- Force block ----------
                    if cls._PATTERNS['force'].match(stripped):
                        if current_frame is not None:
                            current_frame['force'] = cls._read_selected_block(
                                f, num_atoms_total, selected_indices, data_cols=3, skip_cols=1,
                                return_atomic_numbers=False
                            )
                        else:
                            for _ in range(num_atoms_total):
                                f.readline()
                        line = f.readline()
                        continue

                    # ---------- Velocity block ----------
                    if cls._PATTERNS['velocity'].match(stripped):
                        if current_frame is not None:
                            current_frame['velocity'] = cls._read_selected_block(
                                f, num_atoms_total, selected_indices, data_cols=3, skip_cols=1,
                                return_atomic_numbers=False
                            )
                            # All data for this frame collected
                            if (current_frame['lattice'] is not None and
                                current_frame['position'] is not None and
                                current_frame['force'] is not None and
                                current_frame['velocity'] is not None):
                                lattices.append(current_frame['lattice'])
                                positions.append(current_frame['position'])
                                forces.append(current_frame['force'])
                                velocities.append(current_frame['velocity'])
                        else:
                            for _ in range(num_atoms_total):
                                f.readline()
                        line = f.readline()
                        frame_abs += 1
                        continue

                    # ---------- Separator line (skip) ----------
                    if cls._PATTERNS['separator'].match(stripped):
                        line = f.readline()
                        continue

                    # Unknown line: just advance
                    line = f.readline()

        except Exception as e:
            raise RuntimeError(f"Error at frame {frame_abs}: {str(e)}") from e

        # Post-process results
        n_frames = len(iter_times)
        n_selected = len(selected_indices) if selected_indices else 0
        if n_frames == 0:
            raise RuntimeError("No frames loaded. Check file and frame range.")

        # Convert to numpy arrays
        lattice_arr   = np.array(lattices, dtype=np.float64)
        position_arr  = np.array(positions, dtype=np.float64)
        force_arr     = np.array(forces, dtype=np.float64)
        velocity_arr  = np.array(velocities, dtype=np.float64)
        iter_time_arr = np.array(iter_times, dtype=np.float64)

        # Compute Cartesian coordinates: position @ lattice for each frame
        coordinate_arr = np.empty_like(position_arr)
        for i in range(n_frames):
            coordinate_arr[i] = position_arr[i] @ lattice_arr[i]

        # Convert atomic numbers to element symbols (using already collected numbers)
        if selected_atomic_numbers:
            elements_arr = np.array([Element.from_Z(z).symbol for z in selected_atomic_numbers])
        elif atomic_numbers_all and selected_indices:
            # For element filter case, we already have atomic_numbers_all and selected_indices
            elements_arr = np.array([Element.from_Z(atomic_numbers_all[i]).symbol for i in selected_indices])
        else:
            # Should not happen
            elements_arr = np.array([])

        elapsed = time.perf_counter() - start_time
        print(f"Loaded {n_frames} frames, {n_selected} atoms per frame in {elapsed:.2f} seconds.")
        if frame_abs > 0 and end_frame is not None and end_frame > frame_abs:
            print(f"Note: Requested end frame {end_frame} exceeds total frames {frame_abs}. "
                  f"Read all {frame_abs} frames instead.")

        return MovementData(
            num_atoms=num_atoms_total,
            n_frames=n_frames,
            iter_time=iter_time_arr,
            lattice=lattice_arr,
            elements=elements_arr,
            position=position_arr,
            coordinate=coordinate_arr,
            force=force_arr,
            velocity=velocity_arr,
            selected_indices=selected_indices,
        )

    @staticmethod
    def _read_selected_block(f, total_atoms: int, selected_indices: List[int],
                            data_cols: int = 3, skip_cols: int = 1,
                            return_atomic_numbers: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
        """
        Read a block (Position/Force/Velocity) for selected atoms.
        If return_atomic_numbers is True, returns (data_array, atomic_numbers_list).
        Otherwise returns only data_array.
        """
        if not selected_indices:
            if return_atomic_numbers:
                return np.empty((0, data_cols)), []
            return np.empty((0, data_cols))

        selected_set = set(selected_indices)
        max_idx = max(selected_indices)
        data = np.empty((len(selected_indices), data_cols))
        atomic_numbers = [] if return_atomic_numbers else None
        current = 0

        for atom_idx in range(total_atoms):
            line = f.readline()
            if not line:
                raise EOFError(f"Incomplete block: expected {total_atoms} lines, got {atom_idx}")
            if atom_idx in selected_set:
                parts = line.split()
                if len(parts) < skip_cols + data_cols:
                    raise ValueError(f"Line {atom_idx} insufficient columns: {line.strip()}")
                data[current] = [float(x) for x in parts[skip_cols:skip_cols+data_cols]]
                if return_atomic_numbers:
                    # Atomic number is the first column
                    atomic_numbers.append(int(round(float(parts[0]))))
                current += 1
            # Once we have read all selected atoms and passed the max index, skip the rest
            if atom_idx >= max_idx and current >= len(selected_indices):
                remaining = total_atoms - atom_idx - 1
                for _ in range(remaining):
                    f.readline()
                break

        if return_atomic_numbers:
            return data, atomic_numbers
        return data

    @staticmethod
    def _read_full_position_block(f, total_atoms: int) -> Tuple[np.ndarray, List[int]]:
        """Read entire Position block (all atoms) to get atomic numbers and positions."""
        positions = np.empty((total_atoms, 3))
        atomic_numbers = []
        for i in range(total_atoms):
            line = f.readline()
            if not line:
                raise EOFError(f"Incomplete Position block: expected {total_atoms} lines, got {i}")
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Position line {i} insufficient columns: {line.strip()}")
            # Atomic number may be stored as float, but it's actually integer
            atomic_numbers.append(int(round(float(parts[0]))))
            positions[i] = [float(x) for x in parts[1:4]]
        return positions, atomic_numbers

    @staticmethod
    def _select_indices_by_elements(atomic_numbers: List[int], element_filter) -> List[int]:
        """Return indices of atoms matching given element symbols."""
        if isinstance(element_filter, str):
            element_filter = [element_filter]
        target = {e.upper() for e in element_filter}
        indices = []
        for i, zahl in enumerate(atomic_numbers):
            sym = Element.from_Z(zahl).symbol.upper()
            if sym in target:
                indices.append(i)
        if not indices:
            raise ValueError(f"No atoms match elements {element_filter}")
        return indices




def main():
    args = parse_arguments()
    try:
        data = MovementParser.parse(
            filepath=args.file,
            atom_indices=args.indices,
            element_filter=args.elements,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        print("\n" + "="*60)
        print("MOVEMENT FILE PARSING RESULTS")
        print("="*60)
        print(f"Total atoms per frame: {data.num_atoms}")
        print(f"Frames loaded: {data.n_frames}")
        print(f"Selected atoms: {len(data.elements)}")
        print(f"Selected indices: {data.selected_indices}")
        elem_counts = Counter(data.elements)
        elem_str = ', '.join(f"{k}:{v}" for k, v in elem_counts.items())
        print(f"Element distribution: {elem_str}")
        if data.n_frames > 0:
            print(f"\nFirst frame example (first selected atom):")
            print(f"  Element:\n{data.elements[0]}")
            print(f"  Fractional position:\n{data.position[0,0]}")
            print(f"  Cartesian coordinate:\n{data.coordinate[0,0]}")
            print(f"  Force:\n{data.force[0,0]}")
            print(f"  Velocity:\n{data.velocity[0,0]}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()