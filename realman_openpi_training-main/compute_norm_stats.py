#!/usr/bin/env python3
"""
Realman 数据集归一化统计量计算脚本

计算 LeRobot 格式数据集的归一化统计量（normalization statistics），
生成的 norm_stats.json 文件用于 OpenPI 训练时对输入数据进行标准化处理。

计算的统计量包括：
- state: 机械臂状态观测（7 joints + 1 gripper = 8 维）
- actions: delta action（action - state，相对变化量）

使用方法：
    cd openpi
    uv run ../realman_openpi_training/compute_norm_stats.py \
        --config-name pi0_realman_pytorch \
        --dataset-path /path/to/dataset

输出文件：
    openpi/assets/{config_name}/{repo_id}/norm_stats.json
"""

import sys
import os
from pathlib import Path

# 确保 openpi 在 Python 路径中
openpi_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "openpi", "src")
if openpi_path not in sys.path:
    sys.path.insert(0, openpi_path)

# 确保当前目录在 Python 路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入 train.py 以执行 patch 和注册配置
import train  # noqa: F401 - 导入以执行 patch

import argparse
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import openpi.shared.normalize as normalize
import openpi.training.config as _config


def load_parquet_data(parquet_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    从 parquet 文件加载 action 和 observation.state
    
    Returns:
        (actions, states) 两个 numpy 数组
    """
    df = pd.read_parquet(parquet_path, columns=["action", "observation.state"])
    
    actions = np.stack(df["action"].values)
    states = np.stack(df["observation.state"].values)
    
    return actions, states


def process_episode(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    处理单个 episode，计算 delta_actions
    
    Returns:
        (states_8dim, delta_actions)
    """
    parquet_path, state_dim, action_dim = args
    
    actions, states = load_parquet_data(parquet_path)
    
    # 提取前 state_dim 维作为 state（通常是 8 维：7 joints + 1 gripper）
    states_truncated = states[:, :state_dim]
    
    # 计算 delta_actions = action - state（只对前 action_dim 维）
    delta_actions = actions[:, :action_dim] - states[:, :action_dim]
    
    return states_truncated, delta_actions


def compute_norm_stats(
    dataset_path: str,
    config_name: str,
    output_path: str | None = None,
    num_workers: int = 8,
    state_dim: int = 8,
    action_dim: int = 8,
):
    """
    计算归一化统计量
    
    Args:
        dataset_path: 数据集路径
        config_name: 配置名称（用于确定输出路径）
        output_path: 自定义输出路径
        num_workers: 并行 worker 数量
        state_dim: state 维度（默认 8）
        action_dim: action 维度（默认 8）
    """
    dataset_path = Path(dataset_path)
    
    # 加载数据集元数据
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"数据集不存在: {info_path}")
    
    with open(info_path) as f:
        info = json.load(f)
    
    total_episodes = info["total_episodes"]
    chunks_size = info.get("chunks_size", 1000)
    data_path_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    
    print(f"[INFO] 数据集路径: {dataset_path}")
    print(f"[INFO] 总 episode 数: {total_episodes}")
    print(f"[INFO] state 维度: {state_dim}")
    print(f"[INFO] action 维度: {action_dim}")
    
    # 构建 parquet 文件路径列表
    parquet_files = []
    for ep_idx in range(total_episodes):
        ep_chunk = ep_idx // chunks_size
        parquet_path = dataset_path / data_path_template.format(
            episode_chunk=ep_chunk, episode_index=ep_idx
        )
        parquet_files.append((str(parquet_path), state_dim, action_dim))
    
    # 并行处理所有 episode
    print(f"[INFO] 并行加载数据（{num_workers} workers）...")
    all_states = []
    all_delta_actions = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_episode, args): i for i, args in enumerate(parquet_files)}
        for future in tqdm(as_completed(futures), total=total_episodes, desc="Loading episodes"):
            states, delta_actions = future.result()
            all_states.append(states)
            all_delta_actions.append(delta_actions)
    
    # 合并所有数据
    print(f"[INFO] 合并数据...")
    all_states = np.concatenate(all_states, axis=0)
    all_delta_actions = np.concatenate(all_delta_actions, axis=0)
    
    print(f"[INFO] 总帧数: {len(all_states)}")
    
    # 计算统计量
    print(f"[INFO] 计算统计量...")
    
    def compute_stats(data: np.ndarray) -> normalize.NormStats:
        """计算 mean, std, q01, q99"""
        return normalize.NormStats(
            mean=np.mean(data, axis=0).astype(np.float32),
            std=np.std(data, axis=0).astype(np.float32),
            q01=np.percentile(data, 1, axis=0).astype(np.float32),
            q99=np.percentile(data, 99, axis=0).astype(np.float32),
        )
    
    norm_stats = {
        "state": compute_stats(all_states),
        "actions": compute_stats(all_delta_actions),
    }
    
    # 打印统计量摘要
    print("\n" + "=" * 60)
    print("统计量摘要")
    print("=" * 60)
    for key, stat in norm_stats.items():
        print(f"\n{key}:")
        print(f"  shape: {stat.mean.shape}")
        print(f"  mean:  {stat.mean[:4]}...")
        print(f"  std:   {stat.std[:4]}...")
        print(f"  q01:   {stat.q01[:4]}...")
        print(f"  q99:   {stat.q99[:4]}...")
    
    # 确定输出路径
    if output_path:
        output_dir = Path(output_path)
    else:
        config = _config.get_config(config_name)
        data_config = config.data.create(config.assets_dirs, config.model)
        output_dir = config.assets_dirs / data_config.repo_id
    
    # 保存统计量
    print(f"\n[INFO] 保存统计量到: {output_dir}")
    normalize.save(output_dir, norm_stats)
    print(f"[INFO] 完成！文件已保存到: {output_dir / 'norm_stats.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="计算 LeRobot 数据集的归一化统计量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  cd openpi
  uv run ../realman_openpi_training/compute_norm_stats.py \\
      --config-name pi0_realman_pytorch \\
      --dataset-path /path/to/dataset
        """
    )
    parser.add_argument(
        "--config-name", 
        type=str, 
        required=True,
        help="训练配置名称，如 pi0_realman_pytorch"
    )
    parser.add_argument(
        "--dataset-path", 
        type=str, 
        required=True,
        help="数据集路径"
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        default=None,
        help="自定义输出路径（可选）"
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=8,
        help="并行 worker 数量，默认 8"
    )
    parser.add_argument(
        "--state-dim", 
        type=int, 
        default=8,
        help="state 维度，默认 8（7 joints + 1 gripper）"
    )
    parser.add_argument(
        "--action-dim", 
        type=int, 
        default=8,
        help="action 维度，默认 8"
    )
    
    args = parser.parse_args()
    
    compute_norm_stats(
        dataset_path=args.dataset_path,
        config_name=args.config_name,
        output_path=args.output_path,
        num_workers=args.num_workers,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
    )


if __name__ == "__main__":
    main()
