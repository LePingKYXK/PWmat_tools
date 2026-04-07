#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap
from dataclasses import dataclass
from parse_movement import MovementParser
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class MyFormatter(ap.RawDescriptionHelpFormatter, 
                  ap.ArgumentDefaultsHelpFormatter):
    pass

def parse_arguments():
    parser = ap.ArgumentParser(formatter_class=MyFormatter,
                               description=textwrap.dedent(
    """Calculate distances between atom pairs from PWmat MOVEMENT file.
        Author:
    Dr. Huan Wang <huan.wang@whut.edu.cn>
    """))
    # --- parameters adopted from parse_movement.py ---
    parser.add_argument("-f", "--file", 
                        type=Path, 
                        default=Path.cwd() / "MOVEMENT",
                        help="Path to MOVEMENT file")
    parser.add_argument("-sf", "--start-frame", 
                        type=int, default=0,
                        help="First frame index (0-based, inclusive)")
    parser.add_argument("-ef", "--end-frame", 
                        type=int, default=None,
                        help="Last frame index (0-based, inclusive)")
    # --- parameters in this script ---
    parser.add_argument("-id", "--indices", 
                        type=int, nargs='+', required=True,
                        help="Space-separated atom indices (0-based). Must be even number of indices to form pairs.")
    parser.add_argument("-p", "--plot", 
                        action="store_true",
                        help="Plot the distances")
    parser.add_argument("-o", "--output", 
                        type=Path, default=Path("distances.csv"),
                        help="Output CSV filename for distance data")
    
    return parser.parse_args()


def validate_and_prepare_indices(raw_indices: List[int]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Checking the input indices and preparing data.
    1. Check if the number of indices is even.
    2. Generate a deduplicated index list (select_id) for parsing MOVEMENT file.
    3. Generate a list of index pairs (pairs) for calculating distances.
    
    Args:
        raw_indices:    input indices, can be repeated and must be even number.
    
    Returns:
        select_ids:     set of unique indices, for the purpose of reading data.
        pairs:          list of tuple, pairs of indices for calculating distance.
    """
    if len(raw_indices) % 2 != 0:
        raise ValueError(
            f"""The number of atomic indices entered must be even so that pairs can be matched to calculate distances. 
            Please check your input.""")
    
    max_idx = max(raw_indices)
    select_ids = list(range(max_idx + 1))
    
    pairs = list(zip(raw_indices[0::2], raw_indices[1::2]))
    
    return select_ids, pairs

def calculate_pbc_distance_frac(frac_coords1: np.ndarray, 
                                frac_coords2: np.ndarray, 
                                lattice_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the distance taking periodic boundary conditions (PBC) into account.
    
    Args:
        frac_coords1:   shape (n_frames, 3) or (3,)
        frac_coords2:   shape (n_frames, 3) or (3,)
        lattice_matrix: shape (n_frames, 3, 3)
    
    Returns:
        distances: output data array, shape (n_frames,)
    """
    delta_frac = frac_coords1 - frac_coords2
    
    delta_frac = delta_frac - np.round(delta_frac)
    
    # convert fractional coordinates to cartesian coordinates
    # delta_cartesian = delta_frac @ lattice.T
    delta_cartesian = np.einsum('fij,fj->fi', lattice_matrix, delta_frac)
    
    distances = np.linalg.norm(delta_cartesian, axis=1)
    return distances


def plot_distances(frame_indices: np.ndarray, distance_dict: Dict[str, np.ndarray], 
                   output_path: Path, show_plot: bool):
    """
    Plot distances vs frame index.
    
    Args:
        frame_indices: indies of the data frames.
        distance_dict: dictionary contains distance data and column names.
        output_path:   path to the output image file.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, dist_data in distance_dict.items():
        ax.plot(frame_indices, dist_data, label=label, linewidth=2)
    
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Distance (Å)")
    ax.set_title("Atomic Distances vs Frame")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    img_path = output_path.with_suffix(".png")
    fig.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {img_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    args = parse_arguments()
    
    select_ids, pairs = validate_and_prepare_indices(args.indices)
    
    print(f"Validated Pairs: {pairs}")
    print(f"Reading atoms from 0 to {max(args.indices)} (total {len(select_ids)} atoms)...")

    data = MovementParser.parse(
        file_path=args.file,
        atom_indices=select_ids,
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )

    n_frames = data.n_frames
    frames = np.arange(args.start_frame, args.start_frame + n_frames)
    
    results = {}
    for idx1, idx2 in pairs:

        if idx1 >= data.position.shape[1] or idx2 >= data.position.shape[1]:
            raise IndexError(f"Index out of bounds. Requested {idx1} or {idx2} but only {data.position.shape[1]} atoms loaded.")
        
        # extract positions of the two atoms, shape: (n_frames, 3)
        pos1_frac = data.position[:, idx1, :] 
        pos2_frac = data.position[:, idx2, :]
        
        # calculate distances
        distances = calculate_pbc_distance_frac(pos1_frac, pos2_frac, data.lattice)
        
        # generating column names: element(index)
        elem1 = data.elements[idx1] if idx1 < len(data.elements) else f"X{idx1}"
        elem2 = data.elements[idx2] if idx2 < len(data.elements) else f"X{idx2}"
        col_name = f"dist_{elem1}({idx1})-{elem2}({idx2})"
        
        results[col_name] = distances
    
    # 5. save data to CSV file
    df_data = {"Frame": frames}
    df_data.update(results)
    df = pd.DataFrame(df_data)
    
    df.to_csv(args.output, index=False)
    print(f"Data saved to {args.output}")
    
    # 6. ploting
    plot_distances(frames, results, args.output, args.plot)
    

if __name__ == "__main__":
    main()