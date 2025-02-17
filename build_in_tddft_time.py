

import argparse as ap
import numpy as np
from pathlib import Path


def parse_args():
    parser = ap.ArgumentParser(add_help=True,
                               formatter_class=ap.ArgumentDefaultsHelpFormatter,
                               description="""
                               Author:   Dr. Huan Wang
                               Email:    huan.wang@whut.edu.cn,
                               Version:  v1.0,
                               Date:     February 17, 2025,
                               """)
    parser.add_argument("-f",
                        metavar="<OUT.TDDFT_TIME>",
                        type=Path,
                        help="The OUT.TDDFT_TIME file",
                        default=Path.cwd() / "OUT.TDDFT_TIME"
                        )
    parser.add_argument("-dt",
                        metavar="<time interval>",
                        type=float,
                        required=True,
                        help="The time interval",
                        default=0.1
                        )
    return parser.parse_args()


def parse_out_tddft_time(file_path: Path, time_interval: float) -> np.ndarray:
    try:
        data = np.genfromtxt(file_path)
    except Exception as e:
        print(f"Reading {file_path} failed with error: {e}")
        return None
    
    data[:, 0] = np.arange(0, data.shape[0] * time_interval, time_interval)
    return data


def write_in_tddft_time(data: np.ndarray, file_path: Path):
    #### "% 20.15E" The space ensures the negative sign does not disrupt the alignment.
    fmt = ["%15.8f", "% 20.15E"]
    np.savetxt(file_path, data, fmt=fmt, delimiter=" " * 8)


def main():
    args = parse_args()
    time_interval = args.dt
    input_file = args.f
    data = parse_out_tddft_time(input_file, time_interval)
    output_file = input_file.with_name("IN.TDDFT_TIME")
    write_in_tddft_time(data, output_file)


if __name__ == "__main__":
    main()
