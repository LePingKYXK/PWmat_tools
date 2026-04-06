#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import textwrap
from collections import Counter
from typing import Tuple, List, Dict
from parse_movement import MovementParser
from pathlib import Path
try:
    from pymatgen.core import Element
except ImportError:
    raise ImportError("pymatgen is required. Install via: pip install pymatgen")


class MyFormatter(ap.RawDescriptionHelpFormatter,
                  ap.ArgumentDefaultsHelpFormatter):
    pass

def parse_arguments():
    """Parsing command line arguments."""
    parser = ap.ArgumentParser(formatter_class=MyFormatter,
                               description=textwrap.dedent(
    """
    A script to perform RMSD calculation for PWmat MOVEMENT files with PBC, using MovementParser module.
    Author:
        Dr. Huan Wang <huan.wang@whut.edu.cn>
    """
    ))
    parser.add_argument(
        "-f", "--file", 
        type=Path, default=Path.cwd() / "MOVEMENT",
        help="Input trajectory file with PWmat MOVEMENT format."
    )
    parser.add_argument(
        "-o", "--output", 
        type=Path, default=Path.cwd() / "RMSD.csv",
        help="Output file with .csv format."
    )
    parser.add_argument(
        "-p", "--plot", 
        action="store_true",
        help="Show plot window and save figure as .png format."
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
        '-sf', '--start_frame', 
        type=int, default=0,
        help='First frame to read (0-based).'
        )
    parser.add_argument(
        '-ef', '--end_frame', 
        type=int, default=None,
        help='Last frame to read (exclusive). If not given, read until end.'
        )
    return parser.parse_args()

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


def calculate_rmsd_from_parsed_data(data: MovementParser) -> Tuple[np.ndarray, List[str]]:
    """
    Calculate RMSD using parsed data from MovementParser.

    Arguments:
        data (MovementParser):            Data Object created by MovementParser.
        Tuple[np.ndarray, List[str]]:   A tuple containing RMSD data and a list of unique elements. 
                                      The array shape is [n_frames-1, 1 + n_elements], 
                                      where the first column is the total RMSD, 
                                      and the remaining columns are the RMSD for each element.
    """
    if data.n_frames < 2:
        raise ValueError("Need at least 2 frames to calculate RMSD.")
    
    print(f"Calculating RMSD for {data.n_frames} frames, {len(data.elements)} atoms...")

    # Use the first frame as reference
    ref_cart = data.coordinate[0]  # Shape: (n_atoms, 3)
    
    # Obtain unique elements and their masks
    unique_elements = sorted(list(set(data.elements)))
    element_masks = {}
    for elem in unique_elements:
        mask = (data.elements == elem)
        element_masks[elem] = mask

    rmsd_data = []

    # Looping over the following frames
    for frame_idx in range(1, data.n_frames):
        # --- PBC processing: minimize image current frame to reference frame
        # 1. Calculate the fractional coordinate difference
        delta_frac = data.position[frame_idx] - data.position[0]  # Shape: (n_atoms, 3)
        # 2. Applying minimum image convention to the fractional coordinate difference
        delta_frac -= np.round(delta_frac)
        # 3. Converting fractional to Cartesian coordinates
        current_cart_pbc = data.lattice[frame_idx] @ delta_frac.T  # (3, 3) @ (3, n_atoms) -> (3, n_atoms)
        current_cart_pbc = current_cart_pbc.T  # Transpose back to (n_atoms, 3)
        # 4. Obtain absolute coordinates by adding reference frame
        current_cart_pbc = current_cart_pbc + data.coordinate[0] # Shape: (n_atoms, 3)
        # ------------------------------------------------------------
        
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
            print(f"  Frame {frame_idx}/{data.n_frames}")
        # ----------------------

    # Confirm that the total number is a multiple of 1000
    if (data.n_frames - 1) % 1000 != 0:
        print(f"  Frame {data.n_frames - 1}/{data.n_frames}")
    
    return np.array(rmsd_data), unique_elements


def plot_figure(rmsd_data: np.ndarray, output_path: Path, unique_elements: List[str]):
    """ Draw RMSD figure and save to output_path."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ploting total RMSD
    ax.plot(rmsd_data[:, 0], 'b-o', linewidth=1.5, markersize=3, label='Total RMSD')
    
    # Ploting RMSD for ecoh element
    colors = ['r', 'g', 'm', 'c', 'y', 'k']
    for i, elem in enumerate(unique_elements):
        if i < len(colors):
            ax.plot(rmsd_data[:, i+1], f'{colors[i]}-x', linewidth=1.0, 
                    markersize=2, alpha=0.7, label=f'{elem} RMSD')
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('RMSD (Å)', fontsize=12)
    ax.set_title('RMSD over Trajectory', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    fig.tight_layout()
    plot_path = output_path.with_suffix('.png')
    fig.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")
    
    if plt.get_fignums():
        plt.show()


def main():
    args = parse_arguments()
    
    if not args.file.exists():
        raise FileNotFoundError(f"Input file not found: {args.file}")
    
    # --- Parse data using MovementParser module ---
    # Pass arguments to MovementParser
    data = MovementParser.parse(
        file_path=args.file,
        atom_indices=args.indices,
        element_filter=args.elements,
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )
    
    element_counts = Counter(data.elements)
    elements_summary = ", ".join([f"{elem}: {count}" for elem, count in sorted(element_counts.items())])
    print(f"\nSystem parsed: {data.n_frames} frames, {len(data.elements)} atoms, Elements: {elements_summary}")
    # -----------------------------

    # Calculate RMSD
    try:
        rmsd_data, unique_elements = calculate_rmsd_from_parsed_data(data)
    except ValueError as e:
        print(f"Error during RMSD calculation: {e}")
        return

    # Create DataFrame for RMSD data
    frame_col = np.arange(1, len(rmsd_data) + 1)  # Frame 1 to N (skip reference frame 0)
    all_data = np.column_stack((frame_col, rmsd_data))
    column_names = ['Frame', 'Total_RMSD'] + [f'{elem}_RMSD' for elem in unique_elements]
    df = pd.DataFrame(all_data, columns=column_names)
    
    # Save RMSD data to CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, float_format='%.8f')
    print(f"RMSD data saved to: {args.output}")
    
    # Plot RMSD figure if requested
    if args.plot:
        plot_figure(rmsd_data, args.output, unique_elements)


if __name__ == "__main__":
    main()