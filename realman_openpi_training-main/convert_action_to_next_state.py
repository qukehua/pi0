#!/usr/bin/env python3
"""
LeRobot 数据集 Action 转换脚本

将遥操数据集的 action 从"当前帧目标指令"转换为"下一帧状态"。

转换逻辑：
- 前 7 维（关节）：action_t[:7] = state_{t+1}[:7]
- 第 8 维（夹爪）：保持原样（absolute 模式）
- 最后一帧：action_t[:7] = state_t[:7]

使用方法：
    python convert_action_to_next_state.py \\
        --input-path /path/to/original/dataset \\
        --output-path /path/to/converted/dataset \\
        [--num-workers 4]
"""

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def convert_episode_parquet(
    input_parquet: Path,
    output_parquet: Path,
    state_key: str = "observation.state",
    action_key: str = "action",
    joint_dim: int = 7,
) -> dict:
    """
    转换单个 episode 的 parquet 文件。
    
    转换逻辑：
    - 前 joint_dim 维（关节）：action_t = state_{t+1}[:joint_dim]
    - 第 joint_dim+1 维（夹爪）：保持原样（absolute 模式）
    
    Args:
        input_parquet: 输入 parquet 文件路径
        output_parquet: 输出 parquet 文件路径
        state_key: state 字段名
        action_key: action 字段名
        joint_dim: 关节维度（默认 7，不包括夹爪）
    
    Returns:
        转换统计信息
    """
    # 读取 parquet 文件
    df = pd.read_parquet(input_parquet)
    
    # 获取 state 和 action 数据
    states = np.stack(df[state_key].values)  # shape: (num_frames, state_dim)
    actions = np.stack(df[action_key].values)  # shape: (num_frames, action_dim)
    
    num_frames = len(df)
    action_dim = actions.shape[1]
    
    # 计算原始 delta（只看关节部分，用于统计）
    original_delta = actions[:, :joint_dim] - states[:, :joint_dim]
    original_delta_max = np.abs(original_delta).max()
    
    # 创建新的 action
    new_actions = actions.copy()  # 保留原始 action（包括夹爪）
    
    # 只转换关节部分（前 joint_dim 维）
    # 对于 t = 0, 1, ..., num_frames-2：action_t[:joint_dim] = state_{t+1}[:joint_dim]
    new_actions[:-1, :joint_dim] = states[1:, :joint_dim]
    
    # 对于最后一帧：action_t[:joint_dim] = state_t[:joint_dim]（保持不变）
    new_actions[-1, :joint_dim] = states[-1, :joint_dim]
    
    # 夹爪维度（第 joint_dim 维）保持原样，不做转换
    # new_actions[:, joint_dim] 已经是原始值
    
    # 计算新的 delta（只看关节部分，用于统计）
    new_delta = new_actions[:, :joint_dim] - states[:, :joint_dim]
    new_delta_max = np.abs(new_delta).max()
    
    # 更新 DataFrame
    df[action_key] = list(new_actions)
    
    # 确保输出目录存在
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存 parquet 文件
    df.to_parquet(output_parquet, index=False)
    
    return {
        "num_frames": num_frames,
        "original_delta_max": float(original_delta_max),
        "new_delta_max": float(new_delta_max),
    }


def process_episode(args: tuple) -> tuple:
    """
    处理单个 episode（用于并行处理）。
    
    Args:
        args: (episode_idx, input_parquet, output_parquet, state_key, action_key, joint_dim)
    
    Returns:
        (episode_idx, stats_dict)
    """
    episode_idx, input_parquet, output_parquet, state_key, action_key, joint_dim = args
    
    try:
        stats = convert_episode_parquet(
            input_parquet, output_parquet, state_key, action_key, joint_dim
        )
        return episode_idx, stats, None
    except Exception as e:
        return episode_idx, None, str(e)


def convert_dataset(
    input_path: Path,
    output_path: Path,
    num_workers: int = 4,
    state_key: str = "observation.state",
    action_key: str = "action",
    joint_dim: int = 7,
) -> None:
    """
    转换整个数据集。
    
    Args:
        input_path: 输入数据集路径
        output_path: 输出数据集路径
        num_workers: 并行 worker 数量
        state_key: state 字段名
        action_key: action 字段名
        joint_dim: 关节维度（默认 7，不包括夹爪）
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # 检查输入路径
    if not input_path.exists():
        raise FileNotFoundError(f"输入数据集不存在: {input_path}")
    
    # 检查输出路径
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"输出目录已存在且非空: {output_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 读取 info.json
    info_path = input_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    
    total_episodes = info["total_episodes"]
    data_path_template = info["data_path"]
    
    print(f"[INFO] 输入数据集: {input_path}")
    print(f"[INFO] 输出数据集: {output_path}")
    print(f"[INFO] 总 episode 数: {total_episodes}")
    print(f"[INFO] 并行 worker 数: {num_workers}")
    print()
    
    # 复制 meta 目录（排除 episodes_stats.jsonl，因为 action 已更改，需要重新计算）
    meta_input = input_path / "meta"
    meta_output = output_path / "meta"
    meta_output.mkdir(parents=True, exist_ok=True)
    
    # 需要跳过的文件：action 统计量相关的文件
    skip_files = {"episodes_stats.jsonl"}
    copied_files = []
    skipped_files = []
    
    for meta_file in meta_input.iterdir():
        if meta_file.is_file():
            if meta_file.name in skip_files:
                skipped_files.append(meta_file.name)
                continue
            shutil.copy2(meta_file, meta_output / meta_file.name)
            copied_files.append(meta_file.name)
    
    print(f"[INFO] 已复制 meta 目录（{len(copied_files)} 个文件）")
    if skipped_files:
        print(f"[WARN] 已跳过以下文件（action 已更改，需重新计算）: {skipped_files}")
        print(f"[WARN] 请在转换完成后运行以下命令重新计算统计量:")
        print(f"       python -m lerobot.scripts.compute_stats --dataset-path {output_path}")
    
    # 创建视频目录的符号链接（节省空间）
    videos_input = input_path / "videos"
    videos_output = output_path / "videos"
    
    if videos_input.exists():
        # 使用相对路径创建符号链接
        videos_output.symlink_to(videos_input.resolve())
        print(f"[INFO] 已创建视频目录符号链接: {videos_output} -> {videos_input}")
    
    # 收集所有需要处理的 parquet 文件
    tasks = []
    data_input = input_path / "data"
    data_output = output_path / "data"
    
    for episode_idx in range(total_episodes):
        # 计算 chunk 索引
        chunk_idx = episode_idx // info.get("chunks_size", 1000)
        
        # 构建文件路径
        rel_path = data_path_template.format(
            episode_chunk=chunk_idx,
            episode_index=episode_idx
        )
        
        input_parquet = input_path / rel_path
        output_parquet = output_path / rel_path
        
        if not input_parquet.exists():
            print(f"[WARN] Episode {episode_idx} 的 parquet 文件不存在: {input_parquet}")
            continue
        
        tasks.append((
            episode_idx,
            input_parquet,
            output_parquet,
            state_key,
            action_key,
            joint_dim,
        ))
    
    print(f"[INFO] 共 {len(tasks)} 个 episode 需要处理")
    print()
    
    # 并行处理
    stats_list = []
    errors = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_episode, task): task[0] for task in tasks}
        
        completed = 0
        for future in as_completed(futures):
            episode_idx = futures[future]
            completed += 1
            
            ep_idx, stats, error = future.result()
            
            if error:
                errors.append((ep_idx, error))
                print(f"\r[ERROR] Episode {ep_idx}: {error}")
            else:
                stats_list.append(stats)
                
                # 进度显示
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"\r[INFO] 进度: {completed}/{len(tasks)} episodes", end="", flush=True)
    
    print()
    print()
    
    # 统计信息
    if stats_list:
        total_frames = sum(s["num_frames"] for s in stats_list)
        avg_original_delta = np.mean([s["original_delta_max"] for s in stats_list])
        avg_new_delta = np.mean([s["new_delta_max"] for s in stats_list])
        max_original_delta = max(s["original_delta_max"] for s in stats_list)
        max_new_delta = max(s["new_delta_max"] for s in stats_list)
        
        print(f"[INFO] 转换完成!")
        print(f"[INFO] 总帧数: {total_frames}")
        print(f"[INFO] 原始 delta (action - state) 最大值:")
        print(f"       平均: {avg_original_delta:.6f} rad")
        print(f"       最大: {max_original_delta:.6f} rad")
        print(f"[INFO] 新 delta (next_state - state) 最大值:")
        print(f"       平均: {avg_new_delta:.6f} rad")
        print(f"       最大: {max_new_delta:.6f} rad")
    
    if errors:
        print()
        print(f"[WARN] 有 {len(errors)} 个 episode 处理失败:")
        for ep_idx, error in errors[:5]:
            print(f"       Episode {ep_idx}: {error}")
        if len(errors) > 5:
            print(f"       ... 还有 {len(errors) - 5} 个错误")
    
    print()
    print(f"[INFO] 输出数据集: {output_path}")
    print(f"[INFO] 请运行 compute_norm_stats.py 重新计算归一化统计量")


def main():
    parser = argparse.ArgumentParser(
        description="将 LeRobot 数据集的 action 从目标指令转换为下一帧状态",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本用法
    python convert_action_to_next_state.py \\
        --input-path /path/to/original/dataset \\
        --output-path /path/to/converted/dataset
    
    # 指定并行 worker 数
    python convert_action_to_next_state.py \\
        --input-path /path/to/original/dataset \\
        --output-path /path/to/converted/dataset \\
        --num-workers 8
        """
    )
    
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="原始数据集路径（LeRobot v2.1 格式）",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="输出数据集路径（不会覆盖原始数据集）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="并行处理的 worker 数量（默认 4）",
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default="observation.state",
        help="state 字段名（默认 observation.state）",
    )
    parser.add_argument(
        "--action-key",
        type=str,
        default="action",
        help="action 字段名（默认 action）",
    )
    parser.add_argument(
        "--joint-dim",
        type=int,
        default=7,
        help="关节维度（默认 7），只转换前 N 维，夹爪保持原样",
    )
    
    args = parser.parse_args()
    
    try:
        convert_dataset(
            input_path=Path(args.input_path),
            output_path=Path(args.output_path),
            num_workers=args.num_workers,
            state_key=args.state_key,
            action_key=args.action_key,
            joint_dim=args.joint_dim,
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
