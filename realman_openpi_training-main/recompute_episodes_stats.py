#!/usr/bin/env python3
"""
重新计算 LeRobot v2.1 数据集的 episodes_stats.jsonl（仅 action 字段）

当数据集的 action 字段被修改后（如通过 convert_action_to_next_state.py），
需要运行此脚本重新计算 action 的统计量，同时保留其他字段的原始统计量。

使用方法：
    python recompute_episodes_stats.py --dataset-path /path/to/converted_dataset --original-dataset-path /path/to/original_dataset

注意：
    - 此脚本会覆盖原有的 episodes_stats.jsonl 文件
    - 需要在 RoboCOIN 环境中运行（conda activate robocoin）
"""

import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

# 设置导入路径
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_ROBOCOIN_SRC = _PROJECT_ROOT / "RoboCOIN" / "src"

if str(_ROBOCOIN_SRC) not in sys.path:
    sys.path.insert(0, str(_ROBOCOIN_SRC))

from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.utils import (
    EPISODES_STATS_PATH,
    load_info,
    load_episodes,
    load_episodes_stats,
    write_episode_stats,
)


def compute_action_stats_from_parquet(parquet_path: str) -> dict:
    """
    从单个 parquet 文件计算 action 字段的统计量
    
    Args:
        parquet_path: parquet 文件路径
    
    Returns:
        action 统计量字典
    """
    df = pd.read_parquet(parquet_path, columns=["action"])
    
    col_data = df["action"].values
    first_val = col_data[0]
    
    if isinstance(first_val, np.ndarray):
        action_data = np.stack(col_data)
    elif isinstance(first_val, (list, tuple)):
        action_data = np.array([np.array(x) for x in col_data])
    else:
        action_data = np.array(col_data)
    
    return get_feature_stats(action_data, axis=0, keepdims=action_data.ndim == 1)


def process_episode(args: tuple) -> tuple[int, dict]:
    """处理单个 episode（用于多进程）"""
    ep_idx, parquet_path = args
    action_stats = compute_action_stats_from_parquet(parquet_path)
    return ep_idx, action_stats


def recompute_episodes_stats(
    dataset_path: str,
    original_dataset_path: str,
    num_workers: int = 8,
):
    """
    重新计算数据集的 episodes_stats.jsonl（仅 action 字段）
    
    从原始数据集复制其他字段的统计量，只重新计算 action 字段。
    
    Args:
        dataset_path: 转换后的数据集路径
        original_dataset_path: 原始数据集路径（用于复制非 action 统计量）
        num_workers: 并行 worker 数量
    """
    dataset_path = Path(dataset_path)
    original_dataset_path = Path(original_dataset_path)
    
    # 验证路径
    if not dataset_path.exists():
        raise FileNotFoundError(f"转换后的数据集不存在: {dataset_path}")
    if not original_dataset_path.exists():
        raise FileNotFoundError(f"原始数据集不存在: {original_dataset_path}")
    
    # 检查原始数据集的 episodes_stats.jsonl
    original_stats_path = original_dataset_path / EPISODES_STATS_PATH
    if not original_stats_path.exists():
        raise FileNotFoundError(f"原始数据集缺少统计文件: {original_stats_path}")
    
    print(f"[INFO] 转换后数据集: {dataset_path}")
    print(f"[INFO] 原始数据集: {original_dataset_path}")
    
    # 加载元数据
    print(f"[INFO] 加载元数据...")
    info = load_info(dataset_path)
    episodes = load_episodes(dataset_path)
    
    total_episodes = info["total_episodes"]
    chunks_size = info.get("chunks_size", 1000)
    data_path_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    
    print(f"[INFO] 总 episode 数: {total_episodes}")
    print(f"[INFO] 并行 worker 数: {num_workers}")
    
    # 加载原始统计量
    print(f"[INFO] 加载原始统计量...")
    original_episodes_stats = load_episodes_stats(original_dataset_path)
    
    if len(original_episodes_stats) != total_episodes:
        raise ValueError(
            f"原始统计量 episode 数 ({len(original_episodes_stats)}) "
            f"与转换后数据集 episode 数 ({total_episodes}) 不匹配"
        )
    
    # 删除旧的 episodes_stats.jsonl（如果存在）
    stats_path = dataset_path / EPISODES_STATS_PATH
    if stats_path.exists():
        print(f"[INFO] 删除旧的统计文件: {stats_path}")
        stats_path.unlink()
    
    # 构建 parquet 文件路径列表
    parquet_files = []
    for ep_idx in range(total_episodes):
        ep_chunk = ep_idx // chunks_size
        parquet_path = dataset_path / data_path_template.format(
            episode_chunk=ep_chunk, episode_index=ep_idx
        )
        parquet_files.append((ep_idx, str(parquet_path)))
    
    # 并行计算 action 统计量
    print(f"[INFO] 重新计算 action 统计量...")
    new_action_stats = {}
    
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_episode, args): args[0] for args in parquet_files}
            for future in tqdm(as_completed(futures), total=total_episodes, desc="Computing action stats"):
                ep_idx = futures[future]
                _, action_stats = future.result()
                new_action_stats[ep_idx] = action_stats
    else:
        for args in tqdm(parquet_files, desc="Computing action stats"):
            ep_idx, action_stats = process_episode(args)
            new_action_stats[ep_idx] = action_stats
    
    # 合并统计量：原始统计量 + 新的 action 统计量
    print(f"[INFO] 合并统计量...")
    merged_episodes_stats = {}
    for ep_idx in range(total_episodes):
        # 复制原始统计量
        merged_stats = dict(original_episodes_stats[ep_idx])
        # 替换 action 统计量
        merged_stats["action"] = new_action_stats[ep_idx]
        merged_episodes_stats[ep_idx] = merged_stats
    
    # 写入 episodes_stats.jsonl（按顺序）
    print(f"[INFO] 写入统计文件...")
    for ep_idx in range(total_episodes):
        write_episode_stats(ep_idx, merged_episodes_stats[ep_idx], dataset_path)
    
    # 验证聚合统计量
    print(f"[INFO] 验证聚合统计量...")
    agg_stats = aggregate_stats(list(merged_episodes_stats.values()))
    
    # 打印统计量摘要
    print(f"\n[INFO] 统计量摘要:")
    print(f"  总 keys: {list(agg_stats.keys())}")
    for key in ["action", "observation.state"]:
        if key in agg_stats:
            stats = agg_stats[key]
            print(f"  {key}:")
            print(f"    min:  {stats['min'][:4]}...")
            print(f"    max:  {stats['max'][:4]}...")
            print(f"    mean: {stats['mean'][:4]}...")
            print(f"    std:  {stats['std'][:4]}...")
    
    print(f"\n[INFO] 完成！统计文件已保存到: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="重新计算 LeRobot v2.1 数据集的 episodes_stats.jsonl（仅 action 字段）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python recompute_episodes_stats.py \\
        --dataset-path /path/to/converted_dataset \\
        --original-dataset-path /path/to/original_dataset
        
    python recompute_episodes_stats.py \\
        --dataset-path /path/to/converted_dataset \\
        --original-dataset-path /path/to/original_dataset \\
        --num-workers 16
        """
    )
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="转换后的数据集路径（LeRobot v2.1 格式）",
    )
    parser.add_argument(
        "--original-dataset-path",
        type=str,
        required=True,
        help="原始数据集路径（用于复制非 action 统计量）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="并行处理的 worker 数量（默认 8）",
    )
    
    args = parser.parse_args()
    
    recompute_episodes_stats(
        dataset_path=args.dataset_path,
        original_dataset_path=args.original_dataset_path,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
