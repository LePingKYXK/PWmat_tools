#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import textwrap
from collections import Counter
from typing import Tuple, List
from pathlib import Path
try:
    from parse_movement import add_parser_args, MovementParser
except ImportError:
    raise ImportError("parse_movement.py not found. Please ensure it is in the same directory.")
try:
    from pymatgen.core import Element
except ImportError:
    raise ImportError("pymatgen is required. Install via: pip install pymatgen")


class MyFormatter(ap.RawDescriptionHelpFormatter,
                  ap.ArgumentDefaultsHelpFormatter):
    pass


def parse_arguments():
    """Parsing command line arguments."""
    parser = ap.ArgumentParser(
        formatter_class=MyFormatter,
        description=textwrap.dedent(
        """
        A script to perform RMSD calculation for PWmat MOVEMENT files with PBC, using MovementParser module.
        Supports time range filtering (-st/--start_time, -et/--end_time) and flexible atom index selection.
        Author:
            Dr. Huan Wang <huan.wang@whut.edu.cn>
        """))
    # --- parameters adopted from parse_movement.py ---
    add_parser_args(parser)
    
    # --- RMSD calculation specific parameters ---
    parser.add_argument(
        '-id', '--indices',
        type=str, nargs='+',
        help='List of atom indices to load (0-based). Supports ranges, e.g., 2 5 10-45 50.'
    )
    parser.add_argument(
        "-o", "--output",
        type=Path, default=Path.cwd() / "RMSD.csv",
        help="Output file with .csv format."
    )
    parser.add_argument(
        "-p", "--plot",
        action="store_true",
        help="Show plot window and save figure as .png format (figure always saved)."
    )
    return parser.parse_args()


def parse_indices_string(s: str) -> List[int]:
    """
    Parses a string of atom indices and returns a list of integers. 
    Supports:
      - Single index, e.g., 45
      - Multiple indices, e.g., 2 5 10 45 50
      - Range of indices, e.g., 10-45
    """
    s = s.strip()
    if '-' in s:
        parts = s.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid range format: {s}")
        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            raise ValueError(f"Non-numeric range bounds: {s}")
        if start > end:
            raise ValueError(f"Start {start} > end {end} in range {s}")
        return list(range(start, end + 1))
    else:
        return [int(s)]


def kabsch_rmsd(ref: np.ndarray, mob: np.ndarray) -> float:
    """
    Calculate the RMSD between two sets of coordinates after optimal rotation.
    This function assumes ref and mob have the same shape (N, 3).
    """
    if ref.shape[0] == 0 or mob.shape[0] == 0:
        return 0.0
    if ref.shape[0] != mob.shape[0]:
        raise ValueError("Reference and mobile must have same number of atoms.")

    ref_center = ref.mean(axis=0)
    mob_center = mob.mean(axis=0)
    ref_shifted = ref - ref_center
    mob_shifted = mob - mob_center

    H = mob_shifted.T @ ref_shifted
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0.0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    mob_rotated = mob_shifted @ R
    diff = ref_shifted - mob_rotated
    return np.sqrt((diff ** 2).sum() / ref.shape[0])


def calculate_rmsd_from_parsed_data(data) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Calculate RMSD using parsed data from MovementParser.

    Arguments:
        data: Data object created by MovementParser.

    Returns:
        rmsd_data: Array of shape (n_frames-1, 1 + n_elements) where first column is total RMSD,
                   remaining columns are element-specific RMSD.
        unique_elements: List of element symbols.
        times: Time array (fs) for each calculated RMSD (i.e., from frame 1 to end).
    """
    if data.n_frames < 2:
        raise ValueError("Need at least 2 frames to calculate RMSD.")

    print(f"Calculating RMSD for {data.n_frames} frames, {len(data.elements)} atoms...")

    # --- Use the first frame as reference ---
    ref_cart = data.coordinate[0]  # Shape: (n_atoms, 3)

    # --- Obtain unique elements and their masks ---
    unique_elements = sorted(list(set(data.elements)))
    element_masks = {}
    for elem in unique_elements:
        mask = (data.elements == elem)
        element_masks[elem] = mask

    rmsd_data = []
    times = data.iter_time[1:]   # time for each RMSD calculation (skip reference frame)

    # --- Looping over the following frames ---
    for frame_idx in range(1, data.n_frames):
        # --- PBC processing: minimize image current frame to reference frame
        # 1. Calculate the fractional coordinate difference
        delta_frac = data.position[frame_idx] - data.position[0]  # Shape: (n_atoms, 3)
        # 2. Applying minimum image convention to the fractional coordinate difference
        delta_frac -= np.round(delta_frac)
        # 3. Converting fractional to Cartesian coordinates using CURRENT frame lattice
        current_cart_pbc = data.lattice[frame_idx] @ delta_frac.T  # (3,3) @ (3,n_atoms) -> (3,n_atoms)
        current_cart_pbc = current_cart_pbc.T  # Transpose back to (n_atoms, 3)
        # 4. Obtain absolute coordinates by adding reference frame (already in Cartesian)
        current_cart_pbc = current_cart_pbc + data.coordinate[0]  # Shape: (n_atoms, 3)

        # Calculate total RMSD
        total_rmsd = kabsch_rmsd(ref_cart, current_cart_pbc)

        # Calculate RMSD for each element
        elem_rmsds = []
        for elem in unique_elements:
            mask = element_masks[elem]
            ref_elem = ref_cart[mask]
            curr_elem = current_cart_pbc[mask]
            rmsd_elem = kabsch_rmsd(ref_elem, curr_elem)
            elem_rmsds.append(rmsd_elem)

        rmsd_data.append([total_rmsd] + elem_rmsds)

        # --- Printing progress ---
        if frame_idx % 1000 == 0:
            print(f"  Frame {frame_idx}/{data.n_frames-1}")

    # Confirm that the total number is a multiple of 1000
    if (data.n_frames - 1) % 1000 != 0:
        print(f"  Frame {data.n_frames - 1}/{data.n_frames-1}")

    return np.array(rmsd_data), unique_elements, times


def plot_figure(times: np.ndarray, rmsd_data: np.ndarray, 
                output_path: Path, unique_elements: List[str],
                show_plot: bool):
    """Draw RMSD figure with time as x-axis and save to output_path."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot total RMSD
    ax.plot(times, rmsd_data[:, 0], 'k-o', linewidth=1.5, markersize=3, label='Total RMSD')

    # Plot RMSD for each element
    colors = ['r', 'b', 'c', 'm', 'y', 'g']
    for i, elem in enumerate(unique_elements):
        if i < len(colors):
            ax.plot(times, rmsd_data[:, i+1], f'{colors[i]}-x', linewidth=1.0,
                    markersize=2, alpha=0.7, label=f'{elem} RMSD')
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel('Time (fs)', fontsize=12)
    ax.set_ylabel('RMSD (Å)', fontsize=12)
    ax.set_title('RMSD over Trajectory', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

    fig.tight_layout()
    plot_path = output_path.with_suffix('.png')
    fig.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")
    if show_plot:
        plt.show()
    plt.close()


def main():
    args = parse_arguments()

    # --- Validate input arguments ---
    if (args.start_time is None) != (args.end_time is None):
        raise ValueError("--start_time and --end_time must be provided together or both omitted.")

    if not args.file.exists():
        raise FileNotFoundError(f"Input file not found: {args.file}")

    # --- Parse atom indices ---
    atom_indices = None
    if args.indices is not None:
        all_idx = []
        for item in args.indices:
            all_idx.extend(parse_indices_string(item))
        atom_indices = sorted(set(all_idx))   # remove duplicates and sort
        print(f"Selected atom indices: {atom_indices} (total {len(atom_indices)})")

    # --- Parse data using MovementParser module with time filtering ---
    data = MovementParser.parse(
        file_path=args.file,
        atom_indices=atom_indices,
        element_filter=args.elements,
        start_time=args.start_time,
        end_time=args.end_time
    )

    element_counts = Counter(data.elements)
    elements_summary = ", ".join([f"{elem}: {count}" for elem, count in sorted(element_counts.items())])
    print(f"\nSystem parsed: {data.n_frames} frames, {len(data.elements)} atoms, Elements: {elements_summary}")
    print(f"Time range: {data.iter_time[0]:.3f} fs to {data.iter_time[-1]:.3f} fs")

    # --- Calculate RMSD ---
    try:
        rmsd_data, unique_elements, times = calculate_rmsd_from_parsed_data(data)
    except ValueError as e:
        print(f"Error during RMSD calculation: {e}")
        return

    # --- Create DataFrame for RMSD data (time as first column) ---
    column_names = ['Time (fs)', 'Total_RMSD'] + [f'{elem}_RMSD' for elem in unique_elements]
    all_data = np.column_stack((times, rmsd_data))
    df = pd.DataFrame(all_data, columns=column_names)

    # --- Save RMSD data to a CSV file---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, float_format='%.8f')
    print(f"RMSD data saved to: {args.output}")

    # --- Plot RMSD figure (always save, show window only if -p) ---
    if args.plot or True:   # always save plot
        plot_figure(times, rmsd_data, args.output, 
                    unique_elements, args.plot)
        if args.plot:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    main()