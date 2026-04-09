#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap
from pathlib import Path
from typing import List, Tuple, Dict
try:
    from parse_movement import add_parser_args, MovementParser
except ImportError:
    raise ImportError("parse_movement.py not found. Please ensure it is in the same directory.")


class MyFormatter(ap.RawDescriptionHelpFormatter, ap.ArgumentDefaultsHelpFormatter):
    pass

def parse_arguments():
    parser = ap.ArgumentParser(
        formatter_class=MyFormatter,
        description=textwrap.dedent(
            """Calculate distances between atom pairs from PWmat MOVEMENT file.
            Supports time range filtering via -st/--start_time and -et/--end_time.
            Author:
                Dr. Huan Wang <huan.wang@whut.edu.cn>
            """
        )
    )
    # --- parameters adopted from parse_movement.py ---
    add_parser_args(parser)

    # --- Distances calculation specific parameters ---
    parser.add_argument(
        "-id", "--indices",
        type=int, nargs='+', required=True,
        help="Space-separated atom indices (0-based). Must be even number to form pairs. "
             "Indices can repeat; pairs are formed in order: (1st,2nd), (3rd,4th), ..."
    )
    parser.add_argument(
        "-p", "--plot",
        action="store_true",
        help="Plot distances vs time"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path, default=Path("distances.csv"),
        help="Output CSV filename for distance data"
    )
    return parser.parse_args()

def validate_and_prepare_indices(raw_indices: List[int]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Validating the input indices and preparing data.
    1. Check if the number of indices is even.
    2. Generate a deduplicated index list (select_id) for parsing MOVEMENT file, 
    keeping the original order of indices.
    3. Generate pairs of indices for calculating distances.

    Args:
        raw_indices:  input atomic indices (allowing duplicates, must be even)

    Returns:
        select_ids:  list of indices to be read, from 0 to max index (included)
        pairs:       list of index pairs (idx1, idx2)
    """
    if len(raw_indices) % 2 != 0:
        raise ValueError(
            f"The number of atomic indices must be even to form pairs. "
            f"Got {len(raw_indices)} indices: {raw_indices}"
        )

    max_idx = max(raw_indices)
    select_ids = list(range(max_idx + 1))

    # pair-wise the input indices
    pairs = list(zip(raw_indices[0::2], raw_indices[1::2]))
    return select_ids, pairs

def calculate_pbc_distance_frac(
    frac_coords1: np.ndarray,
    frac_coords2: np.ndarray,
    lattice_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculates the distance between atoms using periodic boundary conditions (PBC).
    Using fractional coordinates and lattice matrix, the 并就近取整，
    and then calculate the distance using the Euclidean distance formula.

    Arguments:
        frac_coords1:   fractional coordinates of atom 1,(n_frames, 3) 或 (3,)
        frac_coords2:   fractional coordinates of atom 2, (n_frames, 3) or (3,)
        lattice_matrix: lattice matrix, (n_frames, 3, 3)

    Returns:
        distance:       distances data array, (n_frames,)
    """
    delta_frac = frac_coords1 - frac_coords2
    delta_frac -= np.round(delta_frac)       # Minimum mapping to [0,1)
    # delta_cart = delta_frac @ lattice.T    # convert fractional to cartesian coordinates
    delta_cart = np.einsum('fij, fj -> fi', lattice_matrix, delta_frac)
    distances = np.linalg.norm(delta_cart, axis=1)
    return distances

def plot_distances(
    times: np.ndarray,
    distance_dict: Dict[str, np.ndarray],
    output_path: Path,
    show_plot: bool
) -> None:
    """
    Plot distance vs time curves and save the figure. 

    Arguments:
        times:          time array (n_frames,) in unit of fs
        distance_dict:  distance dictionary, (key: labels, value: distances)
        output_path:    path to the output figure with .png format
        show_plot:      whether show figure on the screen or not
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, dist_data in distance_dict.items():
        ax.plot(times, dist_data, label=label, linewidth=2)
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Distance (Å)")
    ax.set_title("Atomic Distances vs Time")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    img_path = output_path.with_suffix(".png")
    fig.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {img_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def main() -> None:
    args = parse_arguments()

    # check and prepare atom indices and pairs
    print(f"Input atomic indices: {args.indices}")
    select_ids, pairs = validate_and_prepare_indices(args.indices)
    print(f"Validated pairs: {pairs}")
    print(f"Reading atoms 0..{max(args.indices)} (total {len(select_ids)} atoms) to preserve column indexing...")

    # parsing MOVEMENT file
    data = MovementParser.parse(
        file_path=args.file,
        atom_indices=select_ids,
        start_time=args.start_time,
        end_time=args.end_time
    )

    n_frames = data.n_frames
    if n_frames == 0:
        raise RuntimeError("No frames loaded. Check time range or file content.")

    times = data.iter_time   # shape (n_frames,)
    print(f"Loaded {n_frames} frames, time range: {times[0]:.3f} fs to {times[-1]:.3f} fs")

    results = {}
    for idx1, idx2 in pairs:
        # check index: idx1, idx2 should in [0..max_idx]
        if idx1 >= data.position.shape[1] or idx2 >= data.position.shape[1]:
            raise IndexError(
                f"Index out of bounds. Requested {idx1} or {idx2} but only "
                f"{data.position.shape[1]} atoms loaded (max index = {data.position.shape[1]-1})."
            )

        # extract fractional coordinates for atom pairs (n_frames, 3)
        pos1_frac = data.position[:, idx1, :]
        pos2_frac = data.position[:, idx2, :]

        # calculate distances (n_frames,)
        distances = calculate_pbc_distance_frac(pos1_frac, pos2_frac, data.lattice)

        # generate column name: element symbol(index)
        elem1 = data.elements[idx1] if idx1 < len(data.elements) else f"X{idx1}"
        elem2 = data.elements[idx2] if idx2 < len(data.elements) else f"X{idx2}"
        col_name = f"dist_{elem1}({idx1})-{elem2}({idx2})"
        results[col_name] = distances

    # save data to CSV file
    df_data = {"Time (fs)": times}
    df_data.update(results)
    df = pd.DataFrame(df_data)
    df.to_csv(args.output, index=False)
    print(f"Data saved to {args.output}")

    # plot distances vs time
    if args.plot:
        plot_distances(times, results, args.output, args.plot)


if __name__ == "__main__":
    main()