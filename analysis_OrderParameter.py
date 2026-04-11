#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import textwrap
from pathlib import Path
from scipy.signal import correlate, find_peaks
from scipy.optimize import curve_fit
from typing import Any, List, Tuple
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
            python analysis_OrderParameter.py -f MOVEMENT -cid 25 -sid 0-11 --ref-distance 3.3 -st 0 -et 1000 -o analysis
            python analysis_OrderParameter.py -f MOVEMENT -cid 25 -sid 0-11 -f2 REF_MOVEMENT -st 0 -et 1000 -o analysis
        """)
    )
    # --- parameters adopted from parse_movement.py ---
    add_parser_args(parser)
    
    # --- Order Parameter calculation specific parameters ---
    parser.add_argument(
        "-cid", "--center-idx", 
        type=int, required=True,
        help="Index (0-based) of the central Ta atom of the star-of-David",
        )
    parser.add_argument(
        "-sid", "--surrounding-idx", 
        type=str, nargs='+', required=True,
        help="Indices of 12 surrounding Ta atoms (0-based), e.g., '0-11' or '1 2 3 4 5 6 7 8 9 10 11 12'",
        )
    #### Mutually exclusive reference distance source ####
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument(
        "-f2", "--ref-trajectory", 
        type=Path, default=None,
        help="Second MOVEMENT file (e.g., equilibrium AIMD) to compute average distance as reference",
        )
    ref_group.add_argument(
        "--ref-distance", type=float, 
        default=None,
        help="Reference Ta-Ta distance in undistorted lattice (Å)",
        )
    #### --------------------------------------------  ####
    parser.add_argument(
        "--fit-model", 
        type=str, default='damped_osc',
        choices=['exp', 'damped_osc'],
        help="Fit model: 'exp' (exponential decay) or 'damped_osc' (damped oscillation)",
        )
    parser.add_argument(
        "--fit-threshold", 
        type=float, default=0.1,
        help="Threshold for fitting range (fit where C(t) > threshold)",
        )
    parser.add_argument(
        "-o", "--output",
        type=Path, default=Path("cdw_analysis"),
        help="Output base name (without extension)",
        )
    parser.add_argument(
        "-p", "--plot", 
        action="store_true",
        help="Show plot windows interactively",
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
    
def compute_distances_pbc(center_frac: np.ndarray, surrounding_frac: np.ndarray, 
                          lattice: np.ndarray) -> np.ndarray:
    """
    Compute distances from center to each surrounding atom using periodic boundary conditions.
    
    Parameters
    ----------
    center_frac: np.ndarray, shape (3,)
        Fractional coordinates of the central atom.
    surrounding_frac: np.ndarray, shape (n, 3)
        Fractional coordinates of surrounding atoms.
    lattice: np.ndarray, shape (3, 3)
        Lattice vectors (rows).
    
    Returns
    -------
    distances: np.ndarray, shape (n,)
        Distances in Å.
    """
    diff_frac = surrounding_frac - center_frac
    diff_frac -= np.round(diff_frac)                # minimum image convention
    diff_cart = diff_frac @ lattice.T               # convert to Cartesian
    distances = np.linalg.norm(diff_cart, axis=-1)
    return distances

def compute_per_frame_distances(data: Any, center_idx: int, 
                                surrounding_indices: List[int]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    For each frame, compute distances from center atom to surrounding atoms.
    
    Returns
    -------
    times: np.ndarray, shape (n_frames,)
        Time for each frame.
    distances_list: list of np.ndarray
        Each element is an array of distances (length = number of surrounding atoms).
    """
    n_frames = data.n_frames
    times = data.iter_time
    distances_list = []
    for i in range(n_frames):
        frac_all = data.position[i]
        lattice = data.lattice[i]
        center_frac = frac_all[center_idx]
        surrounding_frac = frac_all[surrounding_indices]
        dists = compute_distances_pbc(center_frac, surrounding_frac, lattice)
        distances_list.append(dists)
    return times, distances_list

def compute_mean_distance(data: Any, center_idx: int, surrounding_indices: List[int]) -> float:
    """
    Compute the average distance (with PBC) over all frames and all surrounding atoms.
    """
    times, dist_list = compute_per_frame_distances(data, center_idx, surrounding_indices)
    all_distances = np.concatenate(dist_list)
    mean_dist = np.mean(all_distances)
    return mean_dist

def compute_radial_contraction_phi(center_frac: np.ndarray, surrounding_frac: np.ndarray, 
                                   lattice: np.ndarray, ref_dist: float) -> float:
    """
    Compute radial contraction order parameter for a single frame.
    """
    distances = compute_distances_pbc(center_frac, surrounding_frac, lattice)
    mean_dist = np.mean(distances)
    phi = (mean_dist - ref_dist) / ref_dist
    return phi

def autocorrelation(x: np.ndarray, normalize: bool = True) -> tuple:
    """Compute autocorrelation C(t) for a 1D array x."""
    n = len(x)
    x = x - np.mean(x)
    corr = correlate(x, x, mode='full', method='auto')
    corr = corr[n-1:] / (n - np.arange(n))
    if normalize:
        corr = corr / corr[0]
    return np.arange(n), corr

def exp_decay(t: np.ndarray, tau: float) -> np.ndarray:
    return np.exp(-t / tau)

def damped_osc(t: float, A: float, tau: float, T: float, phi: float) -> float:
    return A * np.exp(-t / tau) * np.cos(2 * np.pi * t / T + phi)

def fit_autocorrelation(t: np.ndarray, corr: np.ndarray, 
                        model: str = 'exp', threshold: float = 0.1) -> Tuple[float, float, dict]:
    positive = corr > threshold
    if not np.any(positive):
        return None, None, {}
    last_idx = np.where(positive)[0][-1]
    t_fit = t[:last_idx+1]
    corr_fit = corr[:last_idx+1]

    if model == 'exp':
        try:
            popt, _ = curve_fit(exp_decay, t_fit, corr_fit, p0=[10.0])
            return popt[0], None, {'tau': popt[0]}
        except:
            return None, None, {}
    elif model == 'damped_osc':
        A0 = 1.0
        tau_guess = 10.0
        idx_e = np.where(corr < 0.3679)[0]
        if len(idx_e) > 0:
            tau_guess = t[idx_e[0]]
        peaks, _ = find_peaks(corr, height=0.2)
        if len(peaks) >= 2:
            T_guess = t[peaks[1]] - t[peaks[0]]
        else:
            T_guess = 20.0
        phi_guess = 0.0
        try:
            popt, _ = curve_fit(damped_osc, t_fit, corr_fit,
                                p0=[A0, tau_guess, T_guess, phi_guess],
                                bounds=([0, 0, 0, -np.pi], [2, np.inf, np.inf, np.pi]))
            A, tau, T, phi = popt
            return tau, T, {'A': A, 'tau': tau, 'T': T, 'phi': phi}
        except:
            print("Damped oscillation fit failed. Falling back to exponential fit.")
            tau, _, _ = fit_autocorrelation(t, corr, model='exp', threshold=threshold)
            return tau, None, {'tau': tau}
    else:
        return None, None, {}

def plot_results(times: np.ndarray, phi: np.ndarray, tau_lags: np.ndarray, corr: np.ndarray,
                 tau: float, period: float, fit_params: dict, model: str, output_png: Path, show_plot: bool) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(times, phi, 'k-', lw=1, label='Radial contraction')
    ax1.set_xlim(times[0], times[-1])
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Radial Order Parameter φ')
    ax1.set_title('CDW Order Parameter')
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(tau_lags, corr, 'b-', lw=1.5, label='Autocorrelation')
    if model == 'exp' and tau is not None:
        t_fit = np.linspace(0, tau_lags[-1], 200)
        ax2.plot(t_fit, exp_decay(t_fit, tau), 'r--', label=f'Exp fit: τ = {tau:.1f} fs')
    elif model == 'damped_osc' and tau is not None:
        t_fit = np.linspace(0, tau_lags[-1], 200)
        A = fit_params.get('A', 1.0)
        T = fit_params.get('T', period)
        phi_phase = fit_params.get('phi', 0.0)
        y_fit = damped_osc(t_fit, A, tau, T, phi_phase)
        ax2.plot(t_fit, y_fit, 'r--', label=f'Damped osc: τ = {tau:.1f} fs, T = {T:.1f} fs')
    if period:
        ax2.axvline(period, color='m', linestyle=':', label=f'Period ≈ {period:.1f} fs')
    ax2.set_xlim(times[0], times[-1])
    ax2.set_xlabel('Lag time (fs)')
    ax2.set_ylabel('C(t)')
    ax2.set_title('Autocorrelation')
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def load_trajectory(file_path, start_time, end_time, max_index):
    data = MovementParser.parse(
        file_path=file_path,
        atom_indices=list(range(max_index + 1)),
        element_filter=None,
        start_time=start_time,
        end_time=end_time,
    )
    print(f"Loaded {data.n_frames} frames from {file_path}")
    if data.n_frames == 0:
        raise RuntimeError(f"No frames loaded from {file_path}")
    return data

def main():
    args = parse_arguments()

    # Parse surrounding indices
    surrounding_list = []
    for item in args.surrounding_idx:
        surrounding_list.extend(parse_indices_range(item))
    # if len(surrounding_list) != 12:
    #     raise ValueError(f"Expected 12 surrounding atoms, got {len(surrounding_list)}")
    all_indices = [args.center_idx] + surrounding_list
    max_idx = max(all_indices)

    # Determine reference distance
    if args.ref_distance is not None:
        ref_dist = args.ref_distance
        print(f"Using user-provided reference distance: {ref_dist:.3f} Å")
    else:
        print(f"Computing reference distance from trajectory: {args.ref_trajectory}")
        ref_data = load_trajectory(args.ref_trajectory, args.start_time, args.end_time, max_idx)
        ref_dist = compute_mean_distance(ref_data, args.center_idx, surrounding_list)
        print(f"Computed reference distance: {ref_dist:.3f} Å")

    # Load main trajectory
    print("Loading main trajectory...")
    data = load_trajectory(args.file, args.start_time, args.end_time, max_idx)

    # ===== Use the unified per-frame distance computation =====
    times, distances_list = compute_per_frame_distances(data, args.center_idx, surrounding_list)
    # Compute phi for each frame
    phi = np.array([(np.mean(d) - ref_dist) / ref_dist for d in distances_list])

    # Autocorrelation
    lags, corr = autocorrelation(phi, normalize=True)
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    tau_lags = lags * dt

    # Fit
    tau, period, fit_params = fit_autocorrelation(tau_lags, corr,
                                                   model=args.fit_model,
                                                   threshold=args.fit_threshold)

    # Find peaks
    peaks, _ = find_peaks(corr, height=0.2)
    periods = []
    if len(peaks) >= 2:
        periods = [tau_lags[peaks[i+1]] - tau_lags[peaks[i]] for i in range(len(peaks)-1)]
        avg_period = np.mean(periods)
    else:
        avg_period = None
    if period is not None:
        avg_period = period

    # Save output
    output_base = args.output
    if output_base.suffix:
        output_base = output_base.with_suffix('')
    phi_csv = output_base.parent / (output_base.stem + '_phi.csv')
    corr_csv = output_base.parent / (output_base.stem + '_autocorr.csv')
    png_path = output_base.parent / (output_base.stem + '.png')

    pd.DataFrame({'time_fs': times, 'phi': phi}).to_csv(phi_csv, index=False)
    pd.DataFrame({'lag_fs': tau_lags, 'autocorr': corr}).to_csv(corr_csv, index=False)

    plot_results(times, phi, tau_lags, corr, tau, avg_period, fit_params, args.fit_model, png_path, args.plot)

    # Summary
    print("\n" + "="*50)
    print("CDW AUTOCORRELATION ANALYSIS")
    print("="*50)
    print(f"Reference distance: {ref_dist:.3f} Å")
    print(f"Time range: {times[0]:.2f} – {times[-1]:.2f} fs")
    print(f"Mean φ: {np.mean(phi):.4f} ± {np.std(phi):.4f}")
    if tau is not None:
        print(f"Coherence time τ: {tau:.2f} fs")
    else:
        print("Coherence time could not be reliably fitted.")
    if avg_period:
        print(f"Mean oscillation period: {avg_period:.2f} fs")
    else:
        print("No clear oscillatory behavior detected.")
    print(f"Output files: {phi_csv}, {corr_csv}, {png_path}")


if __name__ == "__main__":
    main()
