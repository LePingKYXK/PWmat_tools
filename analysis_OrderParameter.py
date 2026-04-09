#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import textwrap
from pathlib import Path
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from typing import List
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
        Compute CDW order parameter (radial contraction) for star-of-David clusters
        and calculate time autocorrelation function to analyze recurring dynamics.
        
        Author:
            Dr. Huan Wang <huan.wang@whut.edu.cn>

        Example:
            python cdw_autocorr.py -f MOVEMENT -cid 42 -sid 0-11 --ref-distance 3.3 -st 0 -et 1000 -o cdw_analysis
        """)
    )
    # --- parameters adopted from parse_movement.py ---
    add_parser_args(parser)
    
    # --- Order Parameter calculation specific parameters ---
    parser.add_argument(
        "-cid", "--center-idx", 
        type=int, required=True,
        help="Index (0-based) of the central Ta atom of the star-of-David"
        )
    parser.add_argument(
        "-sid", "--surrounding-idx", 
        type=str, nargs='+', required=True,
        help="Indices of 12 surrounding Ta atoms (0-based), e.g., '0-11' or '1 2 3 4 5 6 7 8 9 10 11 12'"
        )
    parser.add_argument(
        "--ref-distance",
        type=float, 
        help="Reference Ta-Ta distance in undistorted lattice (Å)"
        )
    parser.add_argument(
        "-o", "--output",
        type=Path, default=Path("cdw_analysis"),
        help="Output base name (without extension)"
        )
    parser.add_argument(
        "-p", "--plot", 
        action="store_true",
        help="Show plot windows interactively"
        )
    return parser.parse_args()

def parse_indices_range(s: str) -> List[int]:
    """
    Parses a string of atom indices and returns a list of integers. 
    Supports:
      - Single index, e.g., 25
      - Multiple indices, e.g., 1 2 3 4 5 6 11 12 13 14 15 16
      - Range of indices, e.g., 1-12
    """
    s = s.strip()
    if '-' in s:
        parts = s.split('-')
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
    
def compute_radial_contraction(center_frac: np.ndarray, surrounding_frac: np.ndarray, 
                               lattice: np.ndarray, ref_dist: float) -> float:
    """
    Compute radial contraction order parameter.

    Arguements:
        center_frac:        shape (3,) array of central atom fractional coordinates
        surrounding_frac:   shape(12,3) array of surrounding atom fractional coordinates
        ref_dist:           reference distance (Å) in undistorted lattice
    
    Returns: 
        phi:                (mean_distance - ref_dist) / ref_dist
                        Negative means contraction (ordered), zero means disordered.
    """
    # Fractional differences
    diff_frac = surrounding_frac - center_frac
    # Minimum image convention
    diff_frac -= np.round(diff_frac)
    # Convert to Cartesian coordinates
    diff_cart = diff_frac @ lattice.T
    # calculate distances
    distances = np.linalg.norm(diff_cart, axis=-1)
    mean_dist = np.mean(distances)
    phi = (mean_dist - ref_dist) / ref_dist
    return phi

def autocorrelation(x: np.ndarray, normalize: bool = True) -> tuple:
    """
    Compute autocorrelation C(t) for a 1D array x (time series).
    Returns lags (in indices) and correlation values.
    """
    n = len(x)
    x = x - np.mean(x)
    from scipy import signal
    corr = signal.correlate(x, x, mode='full', method='auto')
    corr = corr[n-1:] / (n - np.arange(n))
    if normalize:
        corr = corr / corr[0]
    return np.arange(n), corr

def fit_exponential_decay(t: np.ndarray, corr: np.ndarray) -> float:
    """
    Fit C(t) = exp(-t/tau) for t where corr > 0.1
    Returns tau (fs) if successful, else None.
    """
    positive = corr > 0.1
    if not np.any(positive):
        return None
    last_idx = np.where(positive)[0][-1]
    t_fit = t[:last_idx+1]
    corr_fit = corr[:last_idx+1]
    try:
        popt, _ = curve_fit(lambda t, tau: np.exp(-t/tau), t_fit, corr_fit, p0=[10.0])
        return popt[0]
    except:
        return None

def plot_results(times: np.ndarray, phi: float, tau_lags: np.ndarray, corr: np.ndarray, 
                 tau: float, avg_period: float, output_png: str, show_plot: bool) -> None:
    """Create and save the figure with two subplots."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Top: order parameter vs time
    ax1.plot(times, phi, 'b-', lw=1)
    ax1.set_xlim(times[0], times[-1])
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Radial contraction φ')
    ax1.set_title('CDW Order Parameter (star-of-David)')
    ax1.grid(alpha=0.3)

    # Bottom: autocorrelation
    ax2.plot(tau_lags, corr, 'r-', lw=1.5)
    if tau is not None:
        ax2.axvline(tau, color='g', linestyle='--', label=f'τ = {tau:.1f} fs')
        t_fit = np.linspace(0, tau_lags[-1], 200)
        ax2.plot(t_fit, np.exp(-t_fit/tau), 'g:', alpha=0.7)
    if avg_period:
        ax2.axvline(avg_period, color='m', linestyle='--', label=f'Period ≈ {avg_period:.1f} fs')
    ax2.set_xlim(times[0], times[-1])
    ax2.set_xlabel('Lag time (fs)')
    ax2.set_ylabel('Autocorrelation C(t)')
    ax2.set_title('Time Autocorrelation of φ')
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def main():
    args = parse_arguments()

    # Parse surrounding indices
    surrounding_list = []
    for item in args.surrounding_idx:
        surrounding_list.extend(parse_indices_range(item))

#    if len(surrounding_list) != 12:
#        raise ValueError(f"Expected 12 surrounding atoms, got {len(surrounding_list)}")
    all_indices = [args.center_idx] + surrounding_list
    print(f"======= Selected atom indices: {all_indices} =======")

    # Load trajectory
    print("Loading trajectory...")
    data = MovementParser.parse(
        file_path=args.file,
        atom_indices=np.arange(max(all_indices)+1),
        element_filter=None,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    print(f"Loaded {data.n_frames} frames, {data.position.shape[1]} atoms per frame.")
    if data.n_frames == 0:
        raise RuntimeError("No frames loaded.")

    # Compute order parameter for each frame
    phi = np.zeros(data.n_frames)
    for i in range(data.n_frames):
        frac_coords = data.position[i]
        lattice = data.lattice[i]
        center = frac_coords[all_indices[0]]
        surrounding = frac_coords[all_indices[1:]]
        phi[i] = compute_radial_contraction(center, surrounding, lattice, args.ref_distance)

    times = data.iter_time

    # Compute autocorrelation
    lags, corr = autocorrelation(phi, normalize=True)
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    tau_lags = lags * dt

    # Fit decay
    tau = fit_exponential_decay(tau_lags, corr)

    # Find oscillation periods
    peaks, _ = find_peaks(corr, height=0.2)
    periods = []
    if len(peaks) >= 2:
        periods = [tau_lags[peaks[i+1]] - tau_lags[peaks[i]] for i in range(len(peaks)-1)]
        avg_period = np.mean(periods)
    else:
        avg_period = None

    # Save data
    output_base = args.output
    if output_base.suffix:
        output_base = output_base.with_suffix('')
    phi_csv = output_base.parent / (output_base.stem + '_phi.csv')
    corr_csv = output_base.parent / (output_base.stem + '_autocorr.csv')
    png_path = output_base.parent / (output_base.stem + '.png')

    pd.DataFrame({'time_fs': times, 'phi': phi}).to_csv(phi_csv, index=False)
    pd.DataFrame({'lag_fs': tau_lags, 'autocorr': corr}).to_csv(corr_csv, index=False)

    # Plot
    plot_results(times, phi, tau_lags, corr, tau, avg_period, png_path, args.plot)

    # Print summary
    print("\n" + "="*50)
    print("CDW AUTOCORRELATION ANALYSIS")
    print("="*50)
    print(f"Time range: {times[0]:.2f} – {times[-1]:.2f} fs")
    print(f"Mean φ: {np.mean(phi):.4f} ± {np.std(phi):.4f}")
    if tau:
        print(f"Coherence time τ (exponential decay): {tau:.2f} fs")
    else:
        print("Coherence time could not be reliably fitted.")
    if avg_period:
        print(f"Mean oscillation period: {avg_period:.2f} fs")
        print(f"   Individual periods: {', '.join(f'{p:.2f}' for p in periods)}")
    else:
        print("No clear oscillatory behavior detected.")
    print(f"Output files: {phi_csv}, {corr_csv}, {png_path}")


if __name__ == "__main__":
    main()