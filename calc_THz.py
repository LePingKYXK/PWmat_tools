#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import textwrap
from scipy.fft import rfftfreq, next_fast_len, rfftn, irfftn
from scipy.signal import welch, windows, find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.constants import femto, tera
from pathlib import Path
from typing import Any, List, Tuple
try:
    from parse_movement import add_parser_args, MovementParser
except ImportError:
    raise ImportError("parse_movement.py not found. Please ensure it is in the same directory.")
class MyFormatter(ap.RawDescriptionHelpFormatter, 
                  ap.ArgumentDefaultsHelpFormatter):
    pass

# ======================== Argument Parsing ========================
def parse_arguments():
    parser = ap.ArgumentParser(
        formatter_class=MyFormatter,
        description=textwrap.dedent("""
        THz spectrum analysis from PWmat MOVEMENT trajectory.
        Computes distances between specified atom pairs (with PBC) and calculates THz spectrum.
                                    
        Author:
            Dr. Huan Wang <huan.wang@whut.edu.cn>
        Example:
          python thz_from_movement.py -f MOVEMENT -id 0 1 2 3 -st 0 -et 500 -o thz_results
        """)
    )
    # --- Inherit arguments from parse_movement.py ---
    add_parser_args(parser)

    # --- THz analysis parameters ---
    parser.add_argument(
        "-id", "--indices", 
        type=str, nargs='+', required=True,
        help="Atom indices (0-based) in pairs, e.g., '0 1 2 3' or with ranges '0-5 10-15'.",
        )
    parser.add_argument(
        "-o", "--output", 
        type=Path, default=Path.cwd() / "THz_analysis",
        help="Output base name (without extension). Creates .csv and .png files.",
        )
    parser.add_argument(
        "-w", "--window", 
        type=str, default='hann',
        choices=['hann', 'hamming', 'blackman', 'bartlett', 'none'],
        help="Window function",
        )
    parser.add_argument(
        "--method", 
        type=str, 
        default='rfftn', 
        choices=['rfftn', 'welch'],
        )
    parser.add_argument(
        "--preprocess", 
        type=str, 
        default='detrend',
        choices=['detrend', 'diff', 'raw'],
        )
    parser.add_argument(
        "--smooth", 
        type=float, 
        default=0.0, 
        help="Gaussian smoothing sigma (Å)",
        )
    parser.add_argument(
        "--min-peak-height", 
        type=float, 
        default=0.05,
        )
    parser.add_argument(
        "--no-avg", 
        action="store_true", 
        help="Do NOT compute average distance spectrum",
        )
    parser.add_argument(
        "--dt", 
        type=float, 
        default=None, 
        help="Time step in fs (auto-detect if not given)",
        )
    parser.add_argument(
        "-p", "--plot", 
        action="store_true", 
        help="Show plot window interactively",
        )
    parser.add_argument(
        "--no-plot", 
        action="store_true", 
        help="Do NOT show plot window",
        )
    parser.add_argument(
        "--create-example", 
        action="store_true", 
        help="Not implemented",
        )
    return parser.parse_args()

# ======================== Index Parsing ========================
def parse_indices_list_keep_order(indices_str_list: List[str]):
    indices = []
    for token in indices_str_list:
        if '-' in token:
            start, end = map(int, token.split('-'))
            if start > end:
                raise ValueError(f"Invalid range: {token}")
            indices.extend(range(start, end+1))
        else:
            indices.append(int(token))
    return indices

# ======================== PBC Distance ========================
def compute_pbc_distance(frac1: np.ndarray, frac2: np.ndarray, lattice: np.ndarray) -> float:
    diff_frac = frac2 - frac1
    diff_frac -= np.round(diff_frac)
    diff_cart = diff_frac @ lattice.T
    return np.linalg.norm(diff_cart)

def compute_distances_from_trajectory(data: Any, pairs: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    n_frames = data.n_frames
    n_pairs = len(pairs)
    times = data.iter_time
    distances = np.zeros((n_frames, n_pairs))
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(data.selected_indices)}
    for i in range(n_frames):
        frac = data.position[i]
        lattice = data.lattice[i]
        for j, (idx1, idx2) in enumerate(pairs):
            local1 = global_to_local[idx1]
            local2 = global_to_local[idx2]
            distances[i, j] = compute_pbc_distance(frac[local1], frac[local2], lattice)
    return times, distances

# ======================== THz Analysis ========================
def preprocess_distances(distances: np.ndarray, method: str='detrend', smooth_sigma: float=0.0) -> np.ndarray:
    n_time, n_dist = distances.shape
    if method == 'detrend':
        x = np.arange(n_time)[:, np.newaxis]
        A = np.vstack([x[:, 0], np.ones(n_time)]).T
        coeffs = np.linalg.lstsq(A, distances, rcond=None)[0]
        trends = A @ coeffs
        processed = distances - trends
        print(f"Detrend: mean slope = {np.mean(np.abs(coeffs[0])):.2e} Å/step")
    elif method == 'diff':
        processed = np.zeros_like(distances)
        processed[1:-1] = distances[2:] - distances[:-2]
        processed[0] = distances[1] - distances[0]
        processed[-1] = distances[-1] - distances[-2]
    else:
        processed = distances.copy()
    processed -= np.mean(processed, axis=0, keepdims=True)
    if smooth_sigma > 0:
        for i in range(n_dist):
            processed[:, i] = gaussian_filter1d(processed[:, i], sigma=smooth_sigma)
        print(f"Smoothing sigma = {smooth_sigma}")
    return processed

def compute_velocity(distances: np.ndarray, dt_s: float) -> np.ndarray:
    return np.gradient(distances * 1e-10, dt_s, axis=0)

def compute_vacf_fft(velocity: np.ndarray, zero_pad_to_power2: bool=True) -> np.ndarray:
    n_time, n_dist = velocity.shape
    if zero_pad_to_power2:
        n_fft = next_fast_len(2 * n_time - 1)
    else:
        n_fft = 2 * n_time - 1
    v_padded = np.zeros((n_fft, n_dist))
    v_padded[:n_time] = velocity
    V = rfftn(v_padded, axes=0)
    S = V * np.conj(V)
    corr = irfftn(S, axes=0)
    vacf = corr[:n_time]
    vacf /= vacf[0, :]
    return vacf

def compute_thz_spectrum(vacf: np.ndarray, dt_s: float, method: str='rfftn', apply_window: bool=True,
                         window_type: str='hann', freq_limit_thz: float=20.0) -> Tuple[np.ndarray, np.ndarray]:
    n_time, n_dist = vacf.shape
    if method == 'rfftn':
        if apply_window:
            if window_type.lower() == 'hann':
                window = windows.hann(n_time)[:, np.newaxis]
            elif window_type.lower() == 'hamming':
                window = windows.hamming(n_time)[:, np.newaxis]
            elif window_type.lower() == 'blackman':
                window = windows.blackman(n_time)[:, np.newaxis]
            elif window_type.lower() == 'bartlett':
                window = windows.bartlett(n_time)[:, np.newaxis]
            else:
                window = np.ones((n_time, 1))
            vacf_windowed = vacf * window
        else:
            vacf_windowed = vacf
        n_fft = next_fast_len(n_time)
        vacf_padded = np.zeros((n_fft, n_dist))
        vacf_padded[:n_time] = vacf_windowed
        spectrum = np.abs(rfftn(vacf_padded, axes=0))
        freq = rfftfreq(n_fft, d=dt_s)
        freq_thz = freq / tera
    else:  # welch
        n_seg = min(256, n_time // 4)
        freq_thz = None
        spec_list = []
        for i in range(n_dist):
            f, psd = welch(vacf[:, i], fs=1/dt_s, window=window_type,
                           nperseg=n_seg, scaling='density')
            if freq_thz is None:
                freq_thz = f / tera
            spec_list.append(psd)
        spectrum = np.column_stack(spec_list)
    mask = (freq_thz >= 0) & (freq_thz <= freq_limit_thz)
    freq_thz = freq_thz[mask]
    spectrum = spectrum[mask]
    spectrum /= np.max(spectrum, axis=0, keepdims=True)
    return freq_thz, spectrum

def find_peaks_spectrum(freq_thz: np.ndarray, spectrum: np.ndarray, 
                        min_height: float=0.05, min_distance_thz: float=1.0) -> List[dict]:
    if spectrum.ndim > 1:
        spectrum = np.mean(spectrum, axis=1)
    spacing = freq_thz[1] - freq_thz[0]
    min_dist_idx = int(min_distance_thz / spacing) if spacing > 0 else 1
    try:
        peaks_idx, props = find_peaks(spectrum, height=min_height, distance=min_dist_idx)
        peaks = []
        for idx in peaks_idx:
            p = {'frequency': freq_thz[idx], 'intensity': spectrum[idx]}
            if 'widths' in props:
                p['fwhm'] = props['widths'][list(peaks_idx).index(idx)] * spacing
            if 'prominences' in props:
                p['prominence'] = props['prominences'][list(peaks_idx).index(idx)]
            peaks.append(p)
        return peaks
    except:
        return []

# ======================== Save Functions (fixed path handling) ========================
def save_spectra_csv(freq_thz: np.ndarray, spectrum_matrix: np.ndarray, 
                     output_base: Path, avg_spectrum: np.ndarray=None) -> None:
    """Save spectra to CSV. output_base is Path without extension."""
    csv_path = output_base.with_suffix('.csv')
    data = {'Frequency (THz)': freq_thz}
    for i in range(spectrum_matrix.shape[1]):
        data[f'Pair_{i+1}'] = spectrum_matrix[:, i]
    if avg_spectrum is not None:
        data['Average'] = avg_spectrum
    pd.DataFrame(data).to_csv(csv_path, index=False)
    print(f"Spectra saved to {csv_path}")

def save_peaks_csv(peaks: List[dict], output_base: Path) -> None:
    """Save peaks to CSV. output_base is Path without extension."""
    if not peaks:
        return
    peaks_path = output_base.parent / (output_base.stem + '_peaks.csv')
    pd.DataFrame(peaks).to_csv(peaks_path, index=False)
    print(f"Peaks saved to {peaks_path}")

# ======================== Plotting ========================
def plot_results(time_fs: np.ndarray, distances: np.ndarray, vacf: np.ndarray, freq_thz: np.ndarray, 
                 spectrum_matrix: np.ndarray, pairs_info: List[str], avg_spectrum: np.ndarray, 
                 peaks: List[dict], output_base: Path, interactive: bool=False) -> None:
    n_pairs = distances.shape[1]
    fig = plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(2, 3, 1)
    for i in range(n_pairs):
        ax1.plot(time_fs, distances[:, i], linewidth=1, alpha=0.7, label=pairs_info[i])
    ax1.set_xlim([time_fs[0], time_fs[-1]])
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Distance (Å)')
    ax1.set_title('Atom Pair Distances')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    avg_dist = np.mean(distances, axis=1)
    std_dist = np.std(distances, axis=1)
    ax2.plot(time_fs, avg_dist, 'k-', linewidth=1.5)
    ax2.fill_between(time_fs, avg_dist - std_dist, avg_dist + std_dist, alpha=0.3, color='gray')
    ax2.set_xlim([time_fs[0], time_fs[-1]])
    ax2.set_xlabel('Time (fs)')
    ax2.set_ylabel('Average Distance (Å)')
    ax2.set_title('Average Distance ± Std')
    ax2.grid(alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    all_dist = distances.flatten()
    ax3.hist(all_dist, bins=50, alpha=0.7, color='gray', edgecolor='black')
    ax3.axvline(all_dist.mean(), color='r', linestyle='--', label=f'Mean: {all_dist.mean():.3f} Å')
    ax3.set_xlabel('Distance (Å)')
    ax3.set_ylabel('Count')
    ax3.set_title('Distance Distribution')
    ax3.legend()
    ax3.grid(alpha=0.3)

    ax4 = plt.subplot(2, 3, 4)
    for i in range(n_pairs):
        ax4.plot(time_fs, vacf[:, i], linewidth=1, alpha=0.7, label=pairs_info[i])
    ax4.set_xlim([time_fs[0], time_fs[-1]])
    ax4.set_xlabel('Time (fs)')
    ax4.set_ylabel('VACF')
    ax4.set_title('Velocity Autocorrelation Function')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    for i in range(n_pairs):
        ax5.plot(freq_thz, spectrum_matrix[:, i], linewidth=1, alpha=0.7, label=pairs_info[i])
    ax5.set_xlabel('Frequency (THz)')
    ax5.set_ylabel('Intensity (a.u.)')
    ax5.set_title('THz Spectra (Individual)')
    ax5.set_xlim([0, min(20, freq_thz[-1])])
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    mean_spec = np.mean(spectrum_matrix, axis=1)
    std_spec = np.std(spectrum_matrix, axis=1)
    ax6.plot(freq_thz, mean_spec, color='dodgerblue', linewidth=2, label='Mean Spectrum')
    ax6.fill_between(freq_thz, mean_spec - std_spec, mean_spec + std_spec, color='dodgerblue', alpha=0.2)
    if avg_spectrum is not None:
        ax6.plot(freq_thz, avg_spectrum, 'r--', linewidth=1.5, label='Avg Distance Spectrum')
    for peak in peaks:
        ax6.axvline(peak['frequency'], color='g', linestyle='--', alpha=0.7, linewidth=1)
        ax6.text(peak['frequency'], peak['intensity'] + 0.05, f"{peak['frequency']:.2f} THz",
                 ha='center', fontsize=8, rotation=90)
    ax6.set_xlabel('Frequency (THz)')
    ax6.set_ylabel('Intensity (a.u.)')
    ax6.set_title('Mean Spectrum ± Std')
    ax6.set_xlim([0, min(20, freq_thz[-1])])
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    png_path = output_base.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {png_path}")
    if interactive:
        plt.show()
    else:
        plt.close(fig)

# ======================== Main ========================
def main():
    args = parse_arguments()

    if args.create_example:
        print("--create-example not supported for MOVEMENT trajectories.")
        return

    # Parse atom indices into pairs
    raw_indices = parse_indices_list_keep_order(args.indices)
    if len(raw_indices) % 2 != 0:
        raise ValueError(f"Number of indices ({len(raw_indices)}) is not even.")
    pairs = [(raw_indices[i], raw_indices[i+1]) for i in range(0, len(raw_indices), 2)]
    print(f"Atom pairs: {pairs}")

    unique_indices = sorted(set(raw_indices))

    # Load trajectory
    data = MovementParser.parse(
        file_path=args.file,
        atom_indices=unique_indices,
        element_filter=None,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    print(f"Loaded {data.n_frames} frames, {data.position.shape[1]} atoms per frame.")
    if data.n_frames == 0:
        raise RuntimeError("No frames loaded.")

    # Compute distances
    times_fs, distances = compute_distances_from_trajectory(data, pairs)
    n_frames, n_pairs = distances.shape

    # Time step
    if args.dt is None:
        dt_fs = np.mean(np.diff(times_fs))
    else:
        dt_fs = args.dt
    dt_s = dt_fs * femto
    print(f"Time step: {dt_fs:.4f} fs, total time: {times_fs[-1]-times_fs[0]:.2f} fs")

    # Legend labels
    idx_to_element = {global_idx: data.elements[i] for i, global_idx in enumerate(data.selected_indices)}
    pairs_info = []
    for i1, i2 in pairs:
        elem1 = idx_to_element.get(i1, f"X{i1}")
        elem2 = idx_to_element.get(i2, f"X{i2}")
        pairs_info.append(f"{elem1}({i1})-{elem2}({i2})")

    # Preprocess
    processed = preprocess_distances(distances, method=args.preprocess, smooth_sigma=args.smooth)

    # Velocity
    velocity = compute_velocity(processed, dt_s)

    # VACF
    vacf = compute_vacf_fft(velocity, zero_pad_to_power2=True)

    # THz spectrum
    freq_thz, spectrum_matrix = compute_thz_spectrum(
        vacf, dt_s, method=args.method, apply_window=True,
        window_type=args.window, freq_limit_thz=20.0
    )

    # Average distance spectrum (optional)
    avg_spectrum = None
    if not args.no_avg and n_pairs > 1:
        avg_dist = np.mean(distances, axis=1)
        avg_processed = preprocess_distances(avg_dist.reshape(-1, 1), method=args.preprocess, smooth_sigma=args.smooth)
        avg_vel = compute_velocity(avg_processed, dt_s)
        avg_vacf = compute_vacf_fft(avg_vel, zero_pad_to_power2=True)
        _, avg_spec = compute_thz_spectrum(avg_vacf, dt_s, method=args.method, apply_window=True,
                                           window_type=args.window, freq_limit_thz=20.0)
        avg_spectrum = avg_spec[:, 0]

    # Find peaks
    peaks = find_peaks_spectrum(freq_thz, spectrum_matrix, min_height=args.min_peak_height)

    # Prepare output base (remove extension if any)
    output_base = args.output
    if output_base.suffix:
        output_base = output_base.with_suffix('')

    # Save outputs (fixed path generation)
    save_spectra_csv(freq_thz, spectrum_matrix, output_base, avg_spectrum)
    if peaks:
        save_peaks_csv(peaks, output_base)

    # Plot
    interactive = args.plot and not args.no_plot
    plot_results(times_fs, distances, vacf, freq_thz, spectrum_matrix, pairs_info,
                 avg_spectrum, peaks, output_base, interactive)

    print("\n" + "="*60)
    print("THz analysis completed.")
    print(f"Output files: {output_base}.csv, {output_base}.png")
    if peaks:
        print(f"Peaks file: {output_base.parent / (output_base.stem + '_peaks.csv')}")
    print("="*60)


if __name__ == "__main__":
    main()