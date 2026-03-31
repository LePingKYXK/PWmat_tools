#!/usr/bin python3
# -*- coding: utf-8 -*-
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time
import warnings
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from scipy.constants import angstrom,femto, tera, milli
from scipy.fft import rfftn, rfftfreq, next_fast_len, irfftn
from scipy.signal import welch, windows, find_peaks
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings('ignore')

def parse_arguments():
    parser = ap.ArgumentParser(
        #description='Calculate THz spectrum from atom distance time series',
        formatter_class=ap.RawDescriptionHelpFormatter,
        epilog="""
Developers: Dr. Huan Wang
Email: huan.wang@whut.edu.cn
Version: 1.0.0
Date: 2025-12-16

This script is used to calculate THz spectrum from atom distance time series.
Example usage:
  %(prog)s -f distance.dat -o spectra.csv -w hann
  %(prog)s -f distance.dat -c 1 3 5  # only treat 1,3,5 columns of distance data
  %(prog)s --method welch --smooth 1.0
  %(prog)s --create-example  # create example data file by this script for testing
        """
    )
    
    parser.add_argument('-f', '--filename',
                        type=Path,
                        default=Path.cwd() / 'distance.dat',
                        help='Input data file name, e.g. distance.dat')
    
    parser.add_argument('-o', '--output',
                        type=Path,
                        default=Path.cwd() / 'THz_spectra.csv',
                        help='Output file name, e.g. THz_spectra.csv')
    
    parser.add_argument('-w', '--window',
                        type=str,
                        default='hann',
                        choices=['hann', 'hamming', 'blackman', 'bartlett', 'none'],
                        help='Window function to use, e.g.: hann')
    
    parser.add_argument('-c', '--cols_of_distances',
                        nargs='+',
                        type=int,
                        help='Index of distance columns to process, e.g. 1 3 5')
    
    parser.add_argument('--plot',
                        type=Path,
                        default=Path.cwd() / 'all_figures.png',
                        help='Figure file name (Optional)')
    
    parser.add_argument('--method',
                        type=str,
                        default='rfftn',
                        choices=['rfftn', 'welch'],
                        help='Method for calculating spectrum, default: rfft')
    
    parser.add_argument('--preprocess',
                        type=str,
                        default='detrend',
                        choices=['detrend', 'diff', 'raw'],
                        help='Method for preprocessing data, default: detrend')
    
    parser.add_argument('--min-peak-height',
                        type=float,
                        default=0.05,
                        help='Thereshold for minimum peak height, default: 0.05')
    
    parser.add_argument('--no-avg',
                        action='store_true',
                        help='Do NOT calculate the average distance spectrum')
    
    parser.add_argument('--no-window',
                        action='store_true',
                        help='Do NOT use window function')
    
    parser.add_argument('--dt',
                        type=float,
                        help='Time step in units of fs, (Optional, if provided, it will override auto-detection)')
    
    parser.add_argument('--smooth',
                        type=float,
                        default=0.0,
                        help='Sigma value for Gaussian smoothing, default 0 (no smoothing)')
    
    parser.add_argument('--no-plot',
                        action='store_true',
                        help='Do NOT plot the spectrum on the screen')
    
    parser.add_argument('--create-example',
                        action='store_true',
                        help='Create example data if input file does not exist')
    
    return parser.parse_args()


def load_multi_distance_data(filename: Path, dt_fs: Optional[float] = None, 
                             dist_cols: Optional[List[int]] = None,) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load multi-distance data from file, and return time, distance matrix, and time step.
    
    Parameters:
    ----------------------------------------------------------------------------------------------
    filename:   Path to the input file name
    dt_fs:      Time step in units of femtosecond, if None, it will be calculated from data file
    dist_cols:  List of index of distance, if None, all columns of distance data will be processed
    
    Returns:
    ----------------------------------------------------------------------------------------------
    time_fs:    Time series in units of femtosecond
    distances:  Distance matrix (shape: n_time times n_distances)   
    dt_fs:      Time step in units of femtosecond (float number)
    """
    try:
        # Use pandas to read data file, more flexible to handle columns with different formats
        df = pd.read_csv(filename, sep=r'\s+|,')
        data = df.dropna(axis=1, how='all').values
    except Exception as e:
        print(f"ERROR: Unable to load {filename}")
        print(f"ERROR information: {e}")
    
    time_fs = data[:, 0]  # Origanl time in units of femtosecond
    if dist_cols:
        distances = data[:, dist_cols]
    else:
        distances = data[:, 1:]
    
    if dt_fs is None:
        dt_fs = np.mean(np.diff(time_fs)) # in units of femtosecond
    
    n_points, n_distances = distances.shape
    
    print(f"Data Information:")
    print(f"  Data points: {n_points}")
    print(f"  Range of Time: {time_fs[0]:.1f} -- {time_fs[-1]:.1E} fs")
    print(f"  Time step: {dt_fs:.4f} fs")
    print(f"  Total Time: {time_fs[-1] - time_fs[0]:.1E} fs")
    print(f"  Column of Distances: {n_distances}")
    
    # Print the statistics of each distance column
    for i in range(n_distances):
        dist = distances[:, i]
        print(f"  Distance Column {i+1}: {dist.min():.3f} -- {dist.max():.3f} Angstrom, Mean: {dist.mean():.3f} ± {dist.std():.4f} Angstrom")
    
    return time_fs, distances, dt_fs * femto # time step in units of second


def preprocess_distances(distances: np.ndarray,
                         method: str = 'detrend',
                         smooth_sigma: float = 0.0) -> np.ndarray:
    """
    Preprocess distance data
    
    Parameters:
    ----------------------------------------------------------------------
    distances:      distance matrix (with shape: n_time times n_distances)
    method:         processing method ('detrend', 'diff', 'raw')
    smooth_sigma:   gaussian smoothing sigma value, 0 means no smoothing
    
    Returns:
    -----------------------------------------------------------------------
    Distance data array after preprocessing
    """
    n_time, n_distances = distances.shape

    # Remove linear trend for each column
    if method == 'detrend':
        x = np.arange(n_time)[:, np.newaxis]
        A = np.vstack([x[:, 0], np.ones(n_time)]).T
        # # Linear fitting for each column
        coeffs = np.linalg.lstsq(A, distances, rcond=None)[0]
        trends = np.matmul(A, coeffs)
        print(f"shape of A: {A.shape}")
        print(f"shape of coeffs: {coeffs.shape}")
        print(f"shape of trends: {trends.shape}")
        processed = distances - trends
        
        print(f"Detrend Completed: mean slope = {np.mean(np.abs(coeffs[0])):.2e} angstrom/step")
    
    # first order difference (velocity variation)
    elif method == 'diff':
        processed = np.zeros_like(distances)
        # Using central difference
        processed[1:-1, :] = distances[2:, :] - distances[:-2, :]
        processed[0, :] = distances[1, :] - distances[0, :]
        processed[-1, :] = distances[-1, :] - distances[-2, :]
    
    else:  # 'raw'
        processed = distances.copy()
    
    # Remove the DC components for each column
    processed = processed - np.mean(processed, axis=0, keepdims=True)
    
    # Gaussian smoothing (OPTIONAL)
    if smooth_sigma > 0:
        # smoothing for each column
        for i in range(n_distances):
            processed[:, i] = gaussian_filter1d(processed[:, i], sigma=smooth_sigma)
        print(f"Gaussian smoothing: sigma = {smooth_sigma}")
    
    return processed


def compute_velocity(distances: np.ndarray, dt_s: float) -> np.ndarray:
    """
    Calculate velocity (first derivative of distance)
    
    Parameters:
    --------------------------------------------------------------------------------------
    distances:  distance matrix (with shape: n_time times n_distances)
    dt_s:       time step in units of second (float number)
    
    Returns:
    --------------------------------------------------------------------------------------
    velocity:   velocity array (shapge: n_time times n_distances, in m/s)
    """
    velocity = np.gradient(distances * angstrom, dt_s, axis=0)  # in units of m/s
    return velocity


def compute_vacf_fft(velocity: np.ndarray,
                     zero_pad_to_power2: bool = True) -> np.ndarray:
    """
    Using n-dimensional real FFT to calculate velocity auto-correlation function (VACF), and 
    automatically padding the length of the signal to the nearest power of 2 for accelerating FFT.

    Parameters:
    --------------------------------------------------------------------------------------------------------
    velocity:           velocity matrix (with shape: n_time times n_distances)
    zero_pad_to_power2: padding the length of the signal to the nearest power of 2 for accelerating FFT
    
    Returns:
    --------------------------------------------------------------------------------------------------------
    vacf:               velocity autocorrelation function (with shaep: n_time times n_distances, Normalized)
    """
    n_time, n_distances = velocity.shape
    
    # padding the length of the signal to the nearest power of 2
    if zero_pad_to_power2:
        n_fft = next_fast_len(2 * n_time - 1)
    else:
        n_fft = 2 * n_time - 1
    
    # velocity array after padding zeros
    v_padded = np.zeros((n_fft, n_distances))
    v_padded[:n_time, :] = velocity
    
    # calculate FFT along axes 0 (time axis)
    V = rfftn(v_padded, axes=0)  # shape: (n_freq, n_distances)

    # Calculate the power spectrum
    S = np.multiply(V, np.conj(V))

    # auto-correlation function obtained by inverse FFT
    corr = irfftn(S, axes=0)  # Shape: (n_fft, n_distances)
    
    # Collect the auto-correlation function for the first n_time points along time column
    vacf = corr[:n_time, :]
    
    # Normalization for each column
    vacf = vacf / vacf[0, :]
    
    return vacf


def compute_thz_spectrum(vacf: np.ndarray, 
                         dt_s: float, 
                         method: str = 'rfftn',
                         apply_window: bool = True,
                         window_type: str = 'hann',
                         freq_limit_thz: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate THz spectrum based one VACF
    
    Parameters:
    ----------------------------------------------------------------------------------------------------
    vacf:               velocity autocorrelation function matrix (with shape: n_time times n_distances)
    dt_s:               time step in units of second (float number)
    method:             calculation method ('rfft', 'welch')
    apply_window:       whether to apply window function
    window_type:        window function type
    freq_limit_thz:     frequency limit in THz
    
    Returns:
    ----------------------------------------------------------------------------------------------------
    freq_thz:           frequency array in THz
    spectrum_matrix:    spectrum matrix (with shape: n_freq times n_distances)
    """
    n_time, n_distances = vacf.shape
    
    if method == 'rfftn':
        # Apply window function
        if apply_window:
            if window_type.lower() == 'hann':
                window = windows.hann(n_time)[:, np.newaxis]
            elif window_type.lower() == 'hamming':
                window = windows.hamming(n_time)[:, np.newaxis]
            elif window_type.lower() == 'blackman':
                window = windows.blackman(n_time)[:, np.newaxis]
            elif window_type.lower() == 'bartlett':
                window = windows.bartlett(n_time)[:, np.newaxis]
            elif window_type.lower() == 'none':
                window = np.ones((n_time, 1))
            else:
                print(f"Warning: Unknown window type: '{window_type}', using Hann window instead.")
                window = windows.hann(n_time)[:, np.newaxis]
            
            vacf_windowed = vacf * window
        else:
            vacf_windowed = vacf
        
        # Padding the length of the signal to the nearest power of 2 for accelerating FFT
        n_fft = next_fast_len(n_time)
        vacf_padded = np.zeros((n_fft, n_distances))
        vacf_padded[:n_time, :] = vacf_windowed
        
        # Calculate FFT along time axis
        spectrum_matrix = np.abs(rfftn(vacf_padded, axes=0))
        
        # Generate frequency array
        freq = rfftfreq(n_fft, d=dt_s)  # Frequency unit: 1/s
        freq_thz = freq / tera  # Convert to THz

    # Welch method needs to process each column separately    
    elif method == 'welch':
        n_freq = min(256, n_time // 4)
        freq_thz = None
        spectrum_list = []
        
        for i in range(n_distances):
            freq, spectrum = welch(
                vacf[:, i],
                fs=1/dt_s,
                window=window_type,
                nperseg=n_freq,
                scaling='density'
            )
            if freq_thz is None:
                freq_thz = freq / tera  # Convert to THz
            spectrum_list.append(spectrum)
        
        spectrum_matrix = np.column_stack(spectrum_list)
    
    else:
        raise ValueError("method should be either 'rfftn' or 'welch'")
    
    # Frequency range
    mask = (freq_thz >= 0) & (freq_thz <= freq_limit_thz)
    freq_thz = freq_thz[mask]
    spectrum_matrix = spectrum_matrix[mask, :]
    
    # Normalization for each column
    spectrum_matrix = spectrum_matrix / np.max(spectrum_matrix, axis=0, keepdims=True)
    
    return freq_thz, spectrum_matrix


def find_spectral_peaks(freq_thz: np.ndarray, 
                        spectrum: np.ndarray,
                        min_height: float = 0.2,
                        min_distance_thz: float = 1.0,
                        peak_width: float = 0.3) -> List[Dict[str, Any]]:
    """
    Find spectral peaks based on the given spectrum
    
    Parameters:
    ------------------------------------
    freq_thz:           Frequency array (THz)
    spectrum:           Spectrum Intensity array
    min_height:         Minimum peak height
    min_distance_thz:   Minimum distance between peaks (THz)
    peak_width:         Estimated width of the peak (in units of THz)
    
    Returns:
    ------------------------------------
    peaks:              List of peak information dictionaries
    """
    freq_spacing = freq_thz[1] - freq_thz[0]
    min_distance_idx = int(min_distance_thz / freq_spacing) if freq_spacing > 0 else 1
    
    try:
        peak_indices, properties = find_peaks(
            spectrum,
            height=min_height,
            distance=min_distance_idx,
            width=peak_width/freq_spacing if freq_spacing > 0 else 1
        )
    except:
        # return empty list if no peaks are detected
        return []
    
    peaks = []
    for idx in peak_indices:
        peak_info = {
            'frequency': freq_thz[idx],
            'intensity': spectrum[idx],
            'index': idx
        }
        # add width information
        if 'widths' in properties and idx in peak_indices:
            idx_in_list = list(peak_indices).index(idx)
            if idx_in_list < len(properties['widths']):
                peak_info['fwhm'] = properties['widths'][idx_in_list] * freq_spacing

        # add prominence information
        if 'prominences' in properties and idx in peak_indices:
            idx_in_list = list(peak_indices).index(idx)
            if idx_in_list < len(properties['prominences']):
                peak_info['prominence'] = properties['prominences'][idx_in_list]
        
        peaks.append(peak_info)
    
    return peaks


def save_spectra_csv(freq_thz: np.ndarray,
                     spectrum_array: np.ndarray,
                     filename: Path = "THz_spectra.csv",
                     avg_spectrum: Optional[np.ndarray] = None) -> None:
                     
    """
    Save THz spectra information to a CSV file
    
    Parameters:
    --------------------------------------------------------------------
    freq_thz:           Frequency array (THz)
    spectrum_array:     Spectrum array (shape: n_freq times n_distances)
    avg_spectrum:       Average spectrum (Optional)
    filename:           Output file name
    """
    # Ensure the file extension is .csv
    if not filename.suffix == '.csv':
        filename = filename.with_suffix('.csv')
    
    # Create a dictionary and convert it to a DataFrame
    data_dict = {'Frequency (THz)': freq_thz}
    
    n_distances = spectrum_array.shape[1]
    for i in range(n_distances):
        data_dict[f'Intensity_{i+1} (a.u.)'] = spectrum_array[:, i]
    
    if avg_spectrum.any():
        data_dict['Average (a.u.)'] = avg_spectrum
    
    df = pd.DataFrame(data_dict)
    df.to_csv(filename, index=False)
    
    print(f"All THz spectra saved to: {filename}")
    print(f"  Frequency range: {freq_thz[0]:.3f} -- {freq_thz[-1]:.3f} THz")
    print(f"  The spectra obtained from {n_distances} distances trajectories")
    if avg_spectrum.any():
        print(f"  The spectra include the average spectrum")


def save_peaks_csv(peaks: List[Dict[str, Any]],
                   filename: Path = "THz_peaks.csv") -> None:
    """
    Save peak information as CSV file
    
    Parameters:
    ----------------------------------
    peaks:      Peaks list (list of dictionaries)
    filename:   Output file name (default: "THz_peaks.csv")
    """
    if not peaks:
        print("没有检测到峰，不保存峰信息")
        return
    
    if not filename.with_suffix('.csv'):
        filename = filename.with_suffix('.csv')
    
    # peak information
    peak_data = []
    headers = ["Peak", "Frequency(THz)", "Intensity"]
    
    for i, peak in enumerate(peaks, 1):
        row = [i, peak['frequency'], peak['intensity']]
        
        # add optional information
        if 'fwhm' in peak:
            if 'FWHM(THz)' not in headers:
                headers.append("FWHM(THz)")
            row.append(peak['fwhm'])
        
        if 'prominence' in peak:
            if 'Prominence' not in headers:
                headers.append("Prominence")
            row.append(peak['prominence'])
        
        peak_data.append(row)
    
    # save as a .csv file
    with open(filename, 'w') as f:
        f.write(','.join(headers) + '\n')
        for row in peak_data:
            f.write(','.join([str(x) for x in row]) + '\n')
    
    print(f"Peak information saved to: {filename}")


def plot_results(time_fs: np.ndarray,
                 distances: np.ndarray,
                 freq_thz: np.ndarray,
                 vacf: Optional[np.ndarray],
                 spectrum_matrix: np.ndarray,
                 peaks_list: Optional[List[List[Dict[str, Any]]]] = None,
                 save_fig_path: Optional[str] = None):
    """
    Plot results
    
    Parameters:
    -------------------------------------------------------------------------------
    time_fs:            Time series (femtosecond)
    distances:          Distance matrix (shape: n_time times n_distances)
    freq_thz:           Frequency array (THz)
    vacf:               velocity autocorrelation function matrix (optional)
    spectrum_matrix:    Frequency spectrum matrix (shape: n_freq times n_distances)
    avg_spectrum:       Average spectrum (optional)
    peaks_list:         Peaks list (optional)
    save_fig_path:          Path to save the plot (optional)
    """
    fig = plt.figure(figsize=(9, 6))
    
    # 1. Orignal Distrance Trajectories (first 3 columns)
    ax1 = plt.subplot(2, 3, 1)
    for i in range(distances.shape[1]):
        ax1.plot(time_fs, distances[:, i], linewidth=1, alpha=0.7, 
                label=f'Distance {i+1}')
    ax1.set_xlim([time_fs[0], time_fs[-1]])
    ax1.set_xlabel('Time (fs)', fontsize=10)
    ax1.set_ylabel('Distance (Å)', fontsize=10)
    ax1.set_title('Selected Distance Trajectories', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Average of distance trajectories
    ax2 = plt.subplot(2, 3, 2)
    mean_distance = np.mean(distances, axis=1)
    ax2.plot(time_fs, mean_distance, 'b-', linewidth=1.5)
    ax2.fill_between(time_fs, 
                     mean_distance - np.std(distances, axis=1),
                     mean_distance + np.std(distances, axis=1),
                     alpha=0.3, color='blue')
    ax2.set_xlim([time_fs[0], time_fs[-1]])
    ax2.set_xlabel('Time (fs)', fontsize=10)
    ax2.set_ylabel('Average Distance (Å)', fontsize=10)
    ax2.set_title('Average Distance ± Std', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # 3. Histogram of distance values
    ax3 = plt.subplot(2, 3, 3)
    all_distances = distances.flatten()
    ax3.hist(all_distances, bins=50, alpha=0.3, color='dimgray', edgecolor='black')
    ax3.axvline(all_distances.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {all_distances.mean():.3f} Å')
    ax3.set_xlabel('Distance (Å)', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('All Distance Distribution', fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. VACF for each distance
    ax4 = plt.subplot(2, 3, 4)
    for i in range(distances.shape[1]):
        ax4.plot(time_fs, vacf[:, i], linewidth=1, alpha=0.7, label=f'Distance {i+1}')
    ax4.set_xlabel('Time (fs)', fontsize=10)
    ax4.set_ylabel('VACF', fontsize=10)
    ax4.set_title('Individual VACF', fontsize=11)
    ax4.set_xlim([time_fs[0], time_fs[-1]])
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. THz spectrum for each distance
    ax5 = plt.subplot(2, 3, 5)
    for i in range(distances.shape[1]):
        ax5.plot(freq_thz, spectrum_matrix[:, i], linewidth=1, alpha=0.7,
                 label=f'Distance {i+1}')
    ax5.set_xlabel('Frequency (THz)', fontsize=10)
    ax5.set_ylabel('Intensity', fontsize=10)
    ax5.set_title('Individual THz Spectra', fontsize=11)
    ax5.set_xlim([0, min(20, freq_thz[-1])])
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Mean spectrum with std
    ax6 = plt.subplot(2, 3, 6)
    avg_spectrum = np.mean(spectrum_matrix, axis=1)
    std_spectrum = np.std(spectrum_matrix, axis=1)
    
    ax6.plot(freq_thz, avg_spectrum, color='dodgerblue', linewidth=2, label='Mean Spectrum')
    ax6.fill_between(freq_thz, 
                     avg_spectrum - std_spectrum,
                     avg_spectrum + std_spectrum,
                     alpha=0.2, color='dodgerblue', label='±1 Std')
    
    if avg_spectrum.any():
        ax5.plot(freq_thz, avg_spectrum, 'r--', linewidth=1.5, 
                 label='Avg Distance Spectrum')
    
    ax6.set_xlabel('Frequency (THz)', fontsize=10)
    ax6.set_ylabel('Intensity', fontsize=10)
    ax6.set_title('Mean Spectrum ± Std', fontsize=11)
    ax6.set_xlim([0, min(20, freq_thz[-1])])
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 6.1 Peak frequency distribution
    if peaks_list:
        for peak in peaks_list:
            ax6.axvline(x=peak['frequency'], color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax6.text(peak['frequency'], peak['intensity'] + 0.05, 
                     f"{peak['frequency']:.2f} THz",
                     ha='center', fontsize=9, rotation=90)
    plt.tight_layout(pad=0.2, w_pad=1.0, h_pad=1.0, rect=(0, 0, 1.0, 0.97))
    
    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_fig_path}")
    
    plt.show()


def main_multi_thz_analysis(filename: str = "distance.dat",
                            output_csv: str = "THz_spectra.csv",
                            window_type: str = "hann",
                            dt_fs: Optional[float] = None,
                            method: str = 'rfft',
                            preprocess_method: str = 'detrend',
                            smooth_sigma: float = 0.0,
                            compute_avg: bool = True,
                            min_peak_height: float = 0.02,
                            save_fig_path: Optional[str] = None,
                            dist_cols: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Main function for THz analysis, enabling for multiple distance columns.
    
    Parameters:
    --------------------------------------------------------------------------
    filename:           input file name of distance data
    output_csv:         output file name with .csv suffix
    window_type:        name of window function
    dt_fs:              time step (fs), None to auto-calculate
    method:             spectrum calculation method ('rfft' or 'welch')
    preprocess_method:  preprocessing method ('detrend', 'diff', 'raw')
    smooth_sigma:       sigma value for Gaussian smoothing
    compute_avg:        whether to compute spectrum of average distance
    min_peak_height:    minimum peak height for peak detection
    save_plot:          save plot to file path
    dist_cols:          index of distance columns, None refers to all columns
    
    Returns:
    --------------------------------------------------------------------------
    Dict:               Dictionary containing all results
    """
    print("=" * 60)
    print("Starting THz analysis for multiple distance columns...")
    print(f"Input file: {filename}")
    print(f"Output file: {output_csv}")
    print(f"Window function: {window_type}")
    print("=" * 60)
    
    # 1. Load distance data
    time_s, distances, dt_s = load_multi_distance_data(filename, dt_fs, dist_cols)
    n_time, n_distances = distances.shape
    
    # # 2. Resoluton and Nyquist Frequency
    # total_time_fs = dt_s * n_time / femto   # convert to femtosecond
    # freq_resolution_thz = (1 / total_time_fs) * milli
    # nyquist_freq_thz = (1 / (2 * dt_s)) / tera
    
    # print(f"\nSpectrum Resolution and Nyquist Frequency:")
    # print(f"  Total Time: {total_time_fs:.1f} fs")
    # print(f"  Resolution: {freq_resolution_thz:.5f} THz")
    # print(f"  Nyquist Frequencies: {nyquist_freq_thz:.1f} THz")
    
    # 3. Prep-rocess distance data
    print(f"\nData Pre-processing: {preprocess_method}")
    distances_processed = preprocess_distances(
        distances, preprocess_method, smooth_sigma
    )
    
    # 4. Calculate velocity
    print(f"\nCalculating velocity...")
    velocity = compute_velocity(distances_processed, dt_s)
    print(f"  Velocity Stats:")
    print(f"    Mean: {np.mean(velocity):.4f} Angstrom/fs")
    print(f"    Std: {np.std(velocity):.4f} Angstrom/fs")
    
    # 5. Calculate velocity autocorrelation function
    print(f"\nCalculating velocity autocorrelation function...")
    vacf = compute_vacf_fft(velocity, zero_pad_to_power2=True)
    
    # 6. Calculate THz spectrum
    print(f"\nCalculating THz spectrum (method: {method})...")
    freq_thz, spectrum_matrix = compute_thz_spectrum(
        vacf, dt_s, method, True, window_type, freq_limit_thz=20.0
    )
    
    # 7. 计算平均距离的频谱（如果需要）
    # avg_spectrum = None
    # if compute_avg and n_distances > 1:
    #     print(f"\nCalculating average distance spectrum...")
    #     # 计算平均距离
    #     avg_distance = np.mean(distances, axis=1)
    #     avg_distance_processed = preprocess_distances_vectorized(
    #         avg_distance.reshape(-1, 1), preprocess_method, smooth_sigma
    #     )
    #     avg_velocity = compute_velocity_vectorized(avg_distance_processed, dt_fs)
    #     avg_vacf = compute_vacf_fft_vectorized(avg_velocity, zero_pad_to_power2=True)
    #     _, avg_spectrum_matrix = compute_thz_spectrum_vectorized(
    #         avg_vacf, dt_fs, method, True, window_type, freq_limit_thz=20.0
    #     )
    #     avg_spectrum = avg_spectrum_matrix[:, 0]

    # 8. Finding spectral peaks
    print(f"\nFinding spectral peaks...")
    mean_spectrum = np.mean(spectrum_matrix, axis=1)
    peaks = find_spectral_peaks(freq_thz, mean_spectrum, min_peak_height)
    
    # 9. Save peak information to file
    if peaks:
        print(f"\nThere are ({len(peaks)} peaks detected):")
        print("-" * 40)
        for i, peak in enumerate(peaks, 1):
            print(f"Peak {i}: {peak['frequency']:.3f} THz")
            print(f"    Intensity: {peak['intensity']:.3f}")
            if 'fwhm' in peak:
                print(f"    FWHM: {peak['fwhm']:.3f} THz")
            if 'prominence' in peak:
                print(f"    Prominence: {peak['prominence']:.3f}")
        save_peaks_csv(peaks, Path.cwd() / "THz_peaks.csv")
    
    # 10. Plot results
    plot_results(time_s, distances, freq_thz, vacf, spectrum_matrix,
                 peaks, save_fig_path)

    # 11. Print statistic information
    mean_spectrum_all = np.mean(spectrum_matrix, axis=1)
    print(f"\nStatistic Information of THz Spectra:")
    print(f"  Frequency Range: {freq_thz[0]:.2f} - {freq_thz[-1]:.2f} THz")
    print(f"  Mean Intensity of All Spectra: {np.mean(mean_spectrum_all):.4f}")
    print(f"  Std Intensity of All Spectra: {np.std(mean_spectrum_all):.4f}")

    # 12. save THz spectrum to file
    save_spectra_csv(freq_thz, spectrum_matrix, output_csv, mean_spectrum_all)

    # if compute_avg and avg_spectrum is not None:
    #     print(f"  Mean Intensity of Avg Spectrum: {np.mean(avg_spectrum):.4f}")
    
    return {
        'time_s': time_s,
        'distances': distances,
        'n_distances': n_distances,
        'velocity': velocity,
        'vacf': vacf,
        'freq_thz': freq_thz,
        'spectrum_matrix': spectrum_matrix,
        'peaks_list': peaks,
    #     'avg_spectrum': avg_spectrum,     
    }


def create_multi_example_data(filename: Path = "example.dat"):
    """
    Create example data with multiple distance columns.
    """
    dt_fs = 0.5  # Time step 0.5 fs
    n_points = 4000  # Total time 2 ps
    n_distances = 5  # 5 distance series
    
    time_fs = np.arange(n_points) * dt_fs
    
    distances = np.zeros((n_points, n_distances))
    
    # 每个距离序列有不同的频率成分
    frequency_sets = [
        [1.5, 4.2, 8.7],      # 序列1
        [1.2, 3.8, 9.1],      # 序列2
        [1.8, 4.5, 7.9],      # 序列3
        [1.3, 5.1, 10.2],     # 序列4
        [1.6, 4.8, 8.3],      # 序列5
    ]
    
    amplitudes = [
        [0.05, 0.03, 0.02],   # 序列1
        [0.04, 0.025, 0.015], # 序列2
        [0.06, 0.035, 0.025], # 序列3
        [0.045, 0.028, 0.018],# 序列4
        [0.055, 0.032, 0.022],# 序列5
    ]
    
    equilibrium = 2.5  # 平衡距离 2.5 Å
    
    for col in range(n_distances):
        distances[:, col] = equilibrium * np.ones(n_points)
        
        for freq_thz, amp in zip(frequency_sets[col], amplitudes[col]):
            omega = 2 * np.pi * freq_thz * 0.001
            phase = np.random.random() * 2 * np.pi
            distances[:, col] += amp * np.sin(omega * time_fs + phase)
        
        # Add noise and drift to each colunm
        distances[:, col] += 0.001 * np.random.randn(n_points)
        drift = 0.0001 * time_fs / 1000 * (col + 1)
        distances[:, col] += drift
    
    # Save data to file
    data = np.column_stack([time_fs] + [distances[:, i] for i in range(n_distances)])
    header = "time(fs) " + " ".join([f"distance_{i+1}(Angstrom)" for i in range(n_distances)])
    np.savetxt(filename, data, header=header)
    
    print(f"Data with multiple distance columns loaded from: {filename}")
    print(f"Time Range: {time_fs[0]:.1f} -- {time_fs[-1]:.1f} fs, with ({n_points} points)")
    print(f"Series of Distances: {n_distances}")
    
    return time_fs, distances

def format_time(elapsed_seconds: float) -> str:
    """Display elapsed time in a human-readable format"""
    if elapsed_seconds < 1e-6:  # 微秒级别
        return f"{elapsed_seconds * 1e6:.4f} µs"
    elif elapsed_seconds < 1e-3:  # 毫秒级别
        return f"{elapsed_seconds * 1e3:.4f} ms"
    else:  # 秒级别
        return f"{elapsed_seconds:.4f} s"


def main():
    # Parseing command line arguments
    args = parse_arguments()

    # Create example data if needed
    if args.create_example:
        print("Creating example data...")
        create_multi_example_data(args.filename)
    
    # Check if input file exists
    elif not args.filename.exists():
        print(f"ERROR: {args.filename} does NOT exist!")
        print("Please use -f option to specify the correct input file path and try again.")
        print("Or, please use the --create-example option to create example data, and have a try again.")
        sys.exit(1)
    
    # Execute analysis
    results = main_multi_thz_analysis(
        filename=args.filename,
        output_csv=args.output,
        window_type=args.window,
        dt_fs=args.dt,
        method=args.method,
        preprocess_method=args.preprocess,
        smooth_sigma=args.smooth,
        compute_avg=not args.no_avg,
        min_peak_height=args.min_peak_height,
        save_fig_path=args.plot,
        dist_cols=args.cols_of_distances
    )

    print("\n" + "=" * 60)
    print("Job Completed!")
    print(f"{results['n_distances']} distance seriers treated.")
    print(f"Save the THz spectrum data to: {Path.cwd() / args.output}")
    print("=" * 60)
    
    # If plotting is disabled, exit
    if args.no_plot:
        print("\nCation: Plotting function is disabled by --no-plot option.")
        sys.exit(0)


if __name__ == "__main__":
    main()