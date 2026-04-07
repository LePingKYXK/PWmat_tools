#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 PWmat MOVEMENT 文件中原子对的距离脚本。
支持通过 -id 输入原子索引对，并计算考虑周期性边界条件的距离。
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# --- 1. 模拟导入或假设 parse_movement 的接口 ---
# 注意：实际使用中，你需要确保 parse_movement.py 在同一目录或 PYTHONPATH 中
# 为了代码的独立性和演示，这里假设 MovementData 的结构
# 实际运行时，请取消注释下面的 import 并移除模拟类
# from parse_movement import MovementParser, MovementData

try:
    from parse_movement import MovementParser, MovementData
except ImportError:
    # 模拟类，仅用于代码结构展示，实际运行请勿使用此块
    @dataclass
    class MovementData:
        num_atoms: int
        n_frames: int
        lattice: np.ndarray # (n_frames, 3, 3)
        position: np.ndarray # (n_frames, n_selected, 3) 分数坐标
        elements: np.ndarray # (n_selected,) 元素符号
        selected_indices: List[int]
except Exception as e:
    print(f"Warning: Using mock dataclass. {e}")

# --- 2. 自定义参数解析 ---
def parse_arguments():
    """整合 parse_movement 参数与自定义参数"""
    parser = argparse.ArgumentParser(
        description="Calculate distances between atom pairs from PWmat MOVEMENT file."
    )
    
    # --- 继承 parse_movement.py 的参数 ---
    parser.add_argument("-f", "--file", type=Path, default=Path.cwd() / "MOVEMENT",
                        help="Path to MOVEMENT file")
    parser.add_argument("-sf", "--start-frame", type=int, default=0,
                        help="First frame index (0-based, inclusive)")
    parser.add_argument("-ef", "--end-frame", type=int, default=None,
                        help="Last frame index (0-based, inclusive)")
    
    # --- 本脚本特有参数 ---
    parser.add_argument("-id", "--indices", type=int, nargs='+', required=True,
                        help="Space-separated atom indices (0-based). Must be even number of indices to form pairs.")
    parser.add_argument("-p", "--plot", action="store_true", default=True,
                        help="Plot the distances (default: True)")
    parser.add_argument("-o", "--output", type=Path, default=Path("distances.csv"),
                        help="Output CSV filename for distance data")
    
    return parser.parse_args()

# --- 3. 输入验证函数 ---
def validate_and_prepare_indices(raw_indices: List[int]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    验证输入索引并准备数据。
    逻辑：
    1. 检查是否为偶数个（成对）。
    2. 生成去重后的索引列表 (select_id)，用于读取数据。
    3. 生成成对的索引列表 (pairs)，用于计算距离。
    
    Args:
        raw_indices: 用户输入的原始索引列表，可能包含重复，长度必须为偶数。
    
    Returns:
        select_ids: 去重并排序后的索引列表，用于告诉 parser 读取哪些原子。
        pairs: 元组列表，每个元组是一对用于计算距离的索引。
    """
    # 1. 验证奇偶性
    if len(raw_indices) % 2 != 0:
        raise ValueError("输入的原子索引数量必须是偶数，以便两两配对计算距离。请检查输入。")
    
    # 2. 生成读取索引 (select_id)
    # 根据逻辑：select_id = range(max(raw_indices) + 1)
    # 这意味着读取从 0 到最大索引的所有原子
    max_idx = max(raw_indices)
    select_ids = list(range(max_idx + 1))
    
    # 3. 生成计算对 (pairs)
    # 将原始列表每两个分为一组
    pairs = []
    for i in range(0, len(raw_indices), 2):
        pair = (raw_indices[i], raw_indices[i+1])
        pairs.append(pair)
    
    return select_ids, pairs

# --- 4. 距离计算核心 ---
def calculate_pbc_distance_frac(frac_coords1: np.ndarray, 
                                frac_coords2: np.ndarray, 
                                lattice_matrix: np.ndarray) -> np.ndarray:
    """
    计算考虑周期性边界条件 (PBC) 的距离。
    使用最小镜像法。
    
    Args:
        frac_coords1: 形状 (n_frames, 3) 或 (3,)
        frac_coords2: 形状 (n_frames, 3) 或 (3,)
        lattice_matrix: 形状 (n_frames, 3, 3)
    
    Returns:
        distances: 形状 (n_frames,) 的距离数组
    """
    # 计算分数坐标差
    delta_frac = frac_coords1 - frac_coords2
    
    # 最小镜像法：将差值限制在 [-0.5, 0.5) 范围内
    delta_frac = delta_frac - np.round(delta_frac)
    
    # 将分数坐标差转换为笛卡尔坐标差
    # delta_cartesian = delta_frac @ lattice.T
    # 由于 lattice 是 (n_frames, 3, 3)，需要进行批量矩阵乘法
    # 结果形状: (n_frames, 3)
    delta_cartesian = np.einsum('fij,fj->fi', lattice_matrix, delta_frac)
    
    # 计算欧几里得距离
    distances = np.linalg.norm(delta_cartesian, axis=1)
    
    return distances

# --- 5. 绘图函数 ---
def plot_distances(frame_indices: np.ndarray, distance_dict: Dict[str, np.ndarray], output_path: Path):
    """
    使用 ax 绘图。
    
    Args:
        frame_indices: 帧索引数组
        distance_dict: 包含列名和距离数据的字典
        output_path: 图片保存路径
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
    plt.close(fig)
    print(f"Plot saved to {img_path}")

# --- 6. 主函数流程 ---
def main():
    args = parse_arguments()
    
    try:
        # 1. 验证并处理索引
        # raw_indices: [2, 5, 2, 16, 5, 30, 8, 45]
        # select_ids:  range(max(raw_indices)+1) -> 读取 0 到 45 的所有原子
        # pairs:       [(2,5), (2,16), (5,30), (8,45)] -> 用于计算
        select_ids, pairs = validate_and_prepare_indices(args.indices)
        
        print(f"Validated Pairs: {pairs}")
        print(f"Reading atoms from 0 to {max(args.indices)} (total {len(select_ids)} atoms)...")

        # 2. 调用 MovementParser 读取数据
        # 注意：这里传入的是 select_ids (即 range(max+1))，确保 data.position 包含所有需要的原子
        data = MovementParser.parse(
            file_path=args.file,
            atom_indices=select_ids, # 关键点：读取范围
            start_frame=args.start_frame,
            end_frame=args.end_frame
        )
        
        # 3. 数据提取
        # data.position.shape 应该是 (n_frames, len(select_ids), 3)
        # data.lattice.shape 应该是 (n_frames, 3, 3)
        n_frames = data.n_frames
        frames = np.arange(args.start_frame, args.start_frame + n_frames)
        
        # 4. 计算距离
        results = {}
        
        for idx1, idx2 in pairs:
            # 检查索引是否在读取范围内（理论上不会报错，因为 select_ids 是 range(max+1)）
            if idx1 >= data.position.shape[1] or idx2 >= data.position.shape[1]:
                raise IndexError(f"Index out of bounds. Requested {idx1} or {idx2} but only {data.position.shape[1]} atoms loaded.")
            
            # 提取这对原子的分数坐标
            # shape: (n_frames, 3)
            pos1_frac = data.position[:, idx1, :] 
            pos2_frac = data.position[:, idx2, :]
            
            # 计算距离
            distances = calculate_pbc_distance_frac(pos1_frac, pos2_frac, data.lattice)
            
            # 生成列名：元素(索引)
            # 注意：data.elements 是根据 select_ids 顺序排列的
            # 因为 select_ids 是 range(n)，所以 data.elements[i] 对应原子 i 的元素
            elem1 = data.elements[idx1] if idx1 < len(data.elements) else f"X{idx1}"
            elem2 = data.elements[idx2] if idx2 < len(data.elements) else f"X{idx2}"
            col_name = f"dist_{elem1}({idx1})-{elem2}({idx2})"
            
            results[col_name] = distances
        
        # 5. 保存数据
        # 构建 DataFrame
        df_data = {"Frame": frames}
        df_data.update(results)
        df = pd.DataFrame(df_data)
        
        df.to_csv(args.output, index=False)
        print(f"Data saved to {args.output}")
        
        # 6. 绘图
        if args.plot:
            plot_distances(frames, results, args.output)
            
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()