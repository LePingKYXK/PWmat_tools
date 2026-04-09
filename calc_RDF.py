#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Radial Distribution Function (RDF) from PWmat MOVEMENT trajectory.
Supports partial RDF for specific element pairs, and calculates coordination number
cumulative integral (4πρ ∫ g(r) r² dr) which matches VMD output.
"""

import argparse as ap
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.integrate import trapezoid
from typing import List, Tuple
try:
    from parse_movement import add_parser_args, MovementParser
except ImportError:
    raise ImportError("parse_movement.py not found. Please ensure it is in the same directory.")


class MyFormatter(ap.RawDescriptionHelpFormatter, 
                  ap.ArgumentDefaultsHelpFormatter):
    pass

def parse_arguments():
    parser = ap.ArgumentParser(
        formatter_class=MyFormatter,
        description=textwrap.dedent("""
        Compute Radial Distribution Function (RDF) from PWmat MOVEMENT file.
        Supports partial RDF for a specific element pair (--pair) and calculates
        coordination number cumulative integral (--coord) matching VMD.

        Examples:
          # Partial RDF for Si-C with coordination number (recommended)
          python compute_rdf.py -f MOVEMENT --pair Si C --coord

          # Partial RDF without coordination number (simple ∫g dr)
          python compute_rdf.py -f MOVEMENT --pair Si C

          # Total RDF for all atoms
          python compute_rdf.py -f MOVEMENT
        """)
    )
    add_parser_args(parser)

    # --- RDF specific parameters ---
    parser.add_argument(
        "--rmax", 
        type=float, default=8.0,
        help="Maximum pair distance (Å)"
        )
    parser.add_argument(
        "--dr", type=float, 
        default=0.05,
        help="Histogram bin width (Å)"
        )
    parser.add_argument(
        "-id", "--indices", 
        type=int, nargs='+',
        help="Space-separated 0-based atom indices to include (overrides -e if both given)"
        )
    parser.add_argument(
        "-o", "--output", 
        type=Path, default=Path("rdf_output"),
        help="Output base name (without extension). Creates .csv and .png files."
        )
    parser.add_argument(
        "-p", "--plot", 
        action="store_true",
        help="Show plot windows interactively (plots are always saved)."
        )
    parser.add_argument(
        "--time-resolved", 
        action="store_true",
        help="Compute RDF for each frame individually (no cumulative integral)."
        )
    parser.add_argument(
        "--pair", 
        type=str, nargs=2, 
        metavar=('ELEM1', 'ELEM2'),
        help="Compute partial RDF between two element types (e.g., --pair Si C). "
             "If not given, total RDF (all atoms) is computed."
             )
    parser.add_argument(
        "--coord", 
        action="store_true",
        help="Compute coordination number cumulative integral: 4πρ ∫ g(r) r² dr. "
             "Requires --pair. If not set, computes simple ∫ g(r) dr."
        )
    return parser.parse_args()

# ----------------------------------------------------------------------
# RDF calculation with PBC
# ----------------------------------------------------------------------
def compute_partial_rdf_frame(frac_coords1: np.ndarray, frac_coords2: np.ndarray, 
                              lattice: np.ndarray, r_max: float, dr: float, 
                              volume: float, n2_total: int ) -> (np.ndarray, np.ndarray):
    """
    Compute partial RDF g_{AB}(r) for A (frac_coords1) and B (frac_coords2).
    n2_total: total number of B atoms in the whole system (for density normalization).
    Returns r_centers, g_r.
    """
    N1 = frac_coords1.shape[0]
    if N1 == 0 or frac_coords2.shape[0] == 0:
        return None, None
    # Minimum image convention in fractional coordinates
    diff_frac = frac_coords1[:, None, :] - frac_coords2[None, :, :]   # (N1, N2, 3)
    diff_frac -= np.round(diff_frac)
    diff_cart = np.einsum('ijk,kl->ijl', diff_frac, lattice.T)
    distances = np.linalg.norm(diff_cart, axis=2)
    pair_dist = distances.flatten()
    # Histogram
    bins = np.arange(0.0, r_max + dr, dr)
    hist, _ = np.histogram(pair_dist, bins=bins)
    r_centers = bins[:-1] + dr / 2.0
    # Normalization: g_{AB}(r) = hist / (ρ_B * 4πr²Δr * N_A)
    shell_vol = 4.0 * np.pi * r_centers**2 * dr
    density_B = n2_total / volume
    norm = density_B * shell_vol * N1
    norm = np.where(norm > 0, norm, 1.0)
    g_r = hist / norm
    return r_centers, g_r

def compute_total_rdf_frame(frac_coords: np.ndarray, lattice: np.ndarray, 
                            r_max: float, dr: float, volume: float, 
                            num_atoms_total: float) -> (np.ndarray, np.ndarray):
    """
    Total RDF for all atoms (all pairs).
    """
    N = frac_coords.shape[0]
    diff_frac = frac_coords[:, None, :] - frac_coords[None, :, :]
    diff_frac -= np.round(diff_frac)
    diff_cart = np.einsum('ijk,kl->ijl', diff_frac, lattice.T)
    distances = np.linalg.norm(diff_cart, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_dist = distances[mask]
    bins = np.arange(0.0, r_max + dr, dr)
    hist, _ = np.histogram(pair_dist, bins=bins)
    r_centers = bins[:-1] + dr / 2.0
    shell_vol = 4.0 * np.pi * r_centers**2 * dr
    density = num_atoms_total / volume
    norm = density * shell_vol * N
    norm = np.where(norm > 0, norm, 1.0)
    g_r = hist / norm
    return r_centers, g_r

def compute_average_rdf(data: MovementParser, r_max: float,
                        dr: float, pair: Tuple[str, str]=None) -> (np.ndarray, np.ndarray, float):
    """
    Compute average RDF over all frames.
    If pair is (elem1, elem2), compute partial RDF between those elements.
    Otherwise compute total RDF.
    Returns r_centers, g_avg, and also the density of the second element (if partial) for later use.
    """
    n_frames = data.n_frames
    sum_g = None
    density_B = None   # only for partial RDF
    if pair is not None:
        elem1, elem2 = pair
        # Get indices for each element from selected atoms (assumes all atoms of these types are included)
        idx1 = [i for i, el in enumerate(data.elements) if el == elem1]
        idx2 = [i for i, el in enumerate(data.elements) if el == elem2]
        if not idx1 or not idx2:
            raise ValueError(f"Elements {elem1} or {elem2} not found in selected atoms.")
        # Total number of B atoms in the whole system (assuming selected contains all)
        n2_total = len(idx2)
        # Compute density of B atoms (constant over frames if volume changes, but we compute per frame later)
        # We'll compute volume per frame and density_B = n2_total / volume
        # So we store n2_total for later use.
    else:
        idx1 = idx2 = None
        n2_total = None

    for i in range(n_frames):
        vol = np.linalg.det(data.lattice[i])
        if pair is not None:
            # Extract fractional coordinates for the two groups
            pos1 = data.position[i][[data.selected_indices.index(j) for j in idx1]]
            pos2 = data.position[i][[data.selected_indices.index(j) for j in idx2]]
            r, g = compute_partial_rdf_frame(pos1, pos2, data.lattice[i], r_max, dr, vol, n2_total)
            if i == 0:
                density_B = n2_total / vol   # use first frame volume for consistency
        else:
            r, g = compute_total_rdf_frame(data.position[i], data.lattice[i], r_max, dr, vol, data.num_atoms)
        if r is None:
            continue
        if sum_g is None:
            sum_g = g
            r_centers = r
        else:
            sum_g += g
    g_avg = sum_g / n_frames
    return r_centers, g_avg, density_B

def compute_coordination_number(r_centers: np.ndarray, g_r: np.ndarray, density_B: float) -> np.ndarray:
    """
    Compute cumulative coordination number CN(r) = 4π ρ_B ∫_0^r g(r') r'^2 dr'.
    """
    # Compute integrand: g(r) * r^2
    integrand = g_r * r_centers**2
    # Cumulative trapezoidal integration
    cn = np.zeros_like(r_centers)
    for i in range(1, len(r_centers)):
        cn[i] = cn[i-1] + trapezoid(integrand[i-1:i+1], r_centers[i-1:i+1])
    cn *= 4.0 * np.pi * density_B
    return cn

def compute_simple_integral(r_centers: np.ndarray, g_r: np.ndarray) -> np.ndarray:
    """Compute simple ∫ g(r) dr."""
    integral = np.zeros_like(r_centers)
    for i in range(1, len(r_centers)):
        integral[i] = integral[i-1] + trapezoid(g_r[i-1:i+1], r_centers[i-1:i+1])
    return integral


def plot_rdf_with_integral(r_centers: np.ndarray, g_avg: np.ndarray, integral: np.ndarray, 
                           output_png: str, show_plot: bool, title: str, ylabel_right: str) -> None:
    """
    Plot RDF (left axis) and integral (right axis) on twin axes.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(r_centers, g_avg, 'b-', lw=1.5, label='g(r)')
    ax1.plot(r_centers, np.ones_like(r_centers), "k--", lw=1, alpha=0.3)
    ax1.set_xlim(r_centers[0], r_centers[-1])
    ax1.set_xlabel('Distance (Å)')
    ax1.set_ylabel('g(r)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(r_centers, integral, 'r-', lw=1.5, label=ylabel_right)
    ax2.set_xlim(r_centers[0], r_centers[-1])
    ax2.set_ylabel(ylabel_right, color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.set_title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_rdf_heatmap(times: np.ndarray, r_centers: np.ndarray, 
                     g_list: List[np.ndarray], output_png: str, show_plot: bool) -> None:
    g_matrix = np.array(g_list).T
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(g_matrix, aspect='auto', origin='lower',
                   extent=[times[0], times[-1], r_centers[0], r_centers[-1]],
                   cmap='viridis', interpolation='bilinear')
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Distance (Å)')
    ax.set_title('Time-Resolved RDF')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('g(r)')
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    args = parse_arguments()

    # Handle pair specification
    if args.pair is not None:
        if len(args.pair) != 2:
            raise ValueError("--pair requires two element symbols, e.g., --pair Si C")
        elem1, elem2 = args.pair
        # Force element_filter to include both elements (if not already)
        if args.elements is None:
            args.elements = [elem1, elem2]
        else:
            if elem1 not in args.elements:
                args.elements.append(elem1)
            if elem2 not in args.elements:
                args.elements.append(elem2)
        print(f"Computing partial RDF between {elem1} and {elem2}")
        if args.coord:
            print("Coordination number integral will be computed (4πρ ∫ g r² dr).")
        else:
            print("Simple integral ∫ g dr will be computed.")
    else:
        if args.coord:
            print("Warning: --coord only meaningful with --pair. Ignoring --coord.")
            args.coord = False

    # If both -id and -e given, prefer -id
    atom_indices = args.indices
    element_filter = args.elements
    if atom_indices is not None and element_filter is not None:
        print("Both -id and -e provided. Using -id (explicit indices).")
        element_filter = None

    # Load trajectory
    print("Loading trajectory...")
    data = MovementParser.parse(
        file_path=args.file,
        atom_indices=atom_indices,
        element_filter=element_filter,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    print(f"Loaded {data.n_frames} frames, {len(data.selected_indices)} atoms per frame.")
    if data.n_frames == 0:
        raise RuntimeError("No frames loaded. Check time range or file.")

    output_base = args.output
    if output_base.suffix:
        output_base = output_base.with_suffix('')
    csv_path = output_base.with_suffix('.csv')
    png_path = output_base.with_suffix('.png')

    if args.time_resolved:
        # Time-resolved RDF (no integral for simplicity)
        print("Computing time-resolved RDF...")
        times = data.iter_time
        g_list = []
        r_centers = None
        # Precompute indices for pair if needed
        if args.pair:
            elem1, elem2 = args.pair
            idx1 = [i for i, el in enumerate(data.elements) if el == elem1]
            idx2 = [i for i, el in enumerate(data.elements) if el == elem2]
            n2_total = len(idx2)
        for i in range(data.n_frames):
            vol = np.linalg.det(data.lattice[i])
            if args.pair is not None:
                pos1 = data.position[i][idx1]
                pos2 = data.position[i][idx2]
                r, g = compute_partial_rdf_frame(pos1, pos2, data.lattice[i], args.rmax, args.dr, vol, n2_total)
            else:
                r, g = compute_total_rdf_frame(data.position[i], data.lattice[i], args.rmax, args.dr, vol, data.num_atoms)
            if r_centers is None:
                r_centers = r
            g_list.append(g)
        # Save CSV
        df = pd.DataFrame(g_list, columns=[f"{r:.3f}" for r in r_centers])
        df.insert(0, 'Time_fs', times)
        df.to_csv(csv_path, index=False)
        print(f"Per-frame RDF saved to {csv_path}")
        # Heatmap
        heatmap_path = output_base.parent / (output_base.stem + '_heatmap.png')
        plot_rdf_heatmap(times, r_centers, g_list, heatmap_path, args.plot)
        print(f"Heatmap saved to {heatmap_path}")
    else:
        print("Computing average RDF...")
        r_centers, g_avg, density_B = compute_average_rdf(data, args.rmax, args.dr, pair=args.pair)
        if args.pair and args.coord:
            integral = compute_coordination_number(r_centers, g_avg, density_B)
            ylabel = "Coordination number CN(r)"
        else:
            integral = compute_simple_integral(r_centers, g_avg)
            ylabel = "∫ g(r) dr (arb. units)"
        # Save CSV
        df = pd.DataFrame({'r(A)': r_centers, 'g(r)': g_avg, 'integral': integral})
        df.to_csv(csv_path, index=False)
        print(f"RDF and integral saved to {csv_path}")
        # Plot
        if args.pair:
            title = f"Partial RDF {args.pair[0]}-{args.pair[1]}"
        else:
            title = "Total RDF"
        plot_rdf_with_integral(r_centers, g_avg, integral, png_path, args.plot, title, ylabel)
        print(f"Plot saved to {png_path}")

    # Summary
    print("\n" + "="*50)
    print("RDF ANALYSIS COMPLETE")
    print("="*50)
    print(f"Time range: {data.iter_time[0]:.2f} – {data.iter_time[-1]:.2f} fs")
    print(f"Selected atoms: {len(data.selected_indices)}")
    if len(data.elements) > 0:
        from collections import Counter
        elem_counts = Counter(data.elements)
        print(f"Elements: {', '.join(f'{k}:{v}' for k, v in elem_counts.items())}")


if __name__ == "__main__":
    main()