#!/usr/bin/env python3
"""计算 OpenPI 归一化统计量 (norm_stats)

使用方法:
    cd /share/0xyj/model3_openpi0.5/openpi-main
    JAX_PLATFORMS=cpu /share/0xyj/model3_openpi0.5/openpi-main/.venv/bin/python3.11 \\
        /share/0xyj/model3_openpi0.5/my_pi0_training/compute_norm_stats.py \\
        --config-name pi0_right_arm_pytorch \\
        --dataset-path /share/0xyj/model3_openpi0.5/lerobot_dataset

输出路径:
    /share/0xyj/model3_openpi0.5/openpi-main/assets/pi0_right_arm_pytorch/local/right_arm_box_pick/norm_stats.json
"""

import sys
import os
import argparse
from pathlib import Path

# 路径设置
_MY_DIR = Path("/share/0xyj/model3_openpi0.5/my_pi0_training")
_OPENPI_SRC = Path("/share/0xyj/model3_openpi0.5/openpi-main/src")
for _p in [str(_MY_DIR), str(_OPENPI_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# 复用 train_right_arm.py 中的 patch
import train_right_arm
train_right_arm.apply_patches()
_local_root_registry = train_right_arm._local_root_registry

import openpi.training.config as _config
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
from my_config_right_arm import get_my_configs


def register_configs():
    for cfg in get_my_configs():
        if cfg.name not in _config._CONFIGS_DICT:
            _config._CONFIGS.append(cfg)
            _config._CONFIGS_DICT[cfg.name] = cfg


register_configs()


def compute_norm_stats(config_name: str, dataset_path: str, output_path: str | None = None):
    print(f"[INFO] 计算 norm_stats: config={config_name}")
    print(f"[INFO] 数据集路径: {dataset_path}")

    if config_name not in _config._CONFIGS_DICT:
        raise ValueError(f"配置 '{config_name}' 未找到。可用: {list(_config._CONFIGS_DICT.keys())}")

    config = _config._CONFIGS_DICT[config_name]
    data_config_factory = config.data

    # 注册本地数据集路径
    tmp_dc = data_config_factory.create(config.assets_dirs, config.model)
    repo_id = tmp_dc.repo_id
    _data_loader._local_root_registry[repo_id] = dataset_path
    _local_root_registry[repo_id] = dataset_path
    print(f"[INFO] 注册本地路径: {repo_id} -> {dataset_path}")

    # 创建数据加载器（不 shuffle，遍历全数据集）
    from lerobot.common.datasets import lerobot_dataset
    import torch
    import numpy as np

    root = Path(dataset_path)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id, root=root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(config.model.action_horizon)]
            for key in tmp_dc.action_sequence_keys
        },
        video_backend="pyav",
    )

    print(f"[INFO] 数据集大小: {len(dataset)} 样本")

    # 应用数据转换并收集统计量
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=False
    )

    # 收集 state 和 action 数据用于统计
    all_states = []
    all_actions = []

    print("[INFO] 遍历数据集收集统计量...")
    import tqdm
    for batch in tqdm.tqdm(data_loader, desc="Computing stats"):
        # 直接提取 state 和 action
        state = batch.get("observation.state")
        action = batch.get("action")
        if state is not None:
            s = state.numpy() if isinstance(state, torch.Tensor) else np.array(state)
            all_states.append(s[..., :8])  # 只取前8维
        if action is not None:
            a = action.numpy() if isinstance(action, torch.Tensor) else np.array(action)
            if a.ndim == 3:
                a = a[:, 0, :]  # 取第一帧
            all_actions.append(a[..., :8])  # 只取前8维

    if not all_states:
        raise RuntimeError("未能收集到数据，请检查数据集路径和格式")

    states  = np.concatenate(all_states,  axis=0)
    actions = np.concatenate(all_actions, axis=0)

    def compute_stats(data):
        return {
            "mean": data.mean(axis=0).tolist(),
            "std":  data.std(axis=0).tolist(),
            "min":  data.min(axis=0).tolist(),
            "max":  data.max(axis=0).tolist(),
            "q01":  np.quantile(data, 0.01, axis=0).tolist(),
            "q99":  np.quantile(data, 0.99, axis=0).tolist(),
        }

    norm_stats = {
        "norm_stats": {
            "state":   compute_stats(states),
            "actions": compute_stats(actions),
        }
    }

    # 确定输出路径
    if output_path is None:
        assets_dir = Path("/share/0xyj/model3_openpi0.5/openpi-main/assets")
        out_dir = assets_dir / config_name / repo_id
    else:
        out_dir = Path(output_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "norm_stats.json"

    import json
    with open(out_file, "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"[INFO] norm_stats 已保存: {out_file}")
    print(f"[INFO] state  mean: {[f'{v:.4f}' for v in norm_stats['norm_stats']['state']['mean']]}")
    print(f"[INFO] action mean: {[f'{v:.4f}' for v in norm_stats['norm_stats']['actions']['mean']]}")
    return str(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算 OpenPI 归一化统计量")
    parser.add_argument("--config-name", default="pi0_right_arm_pytorch", help="训练配置名称")
    parser.add_argument("--dataset-path", default="/share/0xyj/model3_openpi0.5/lerobot_dataset", help="数据集路径")
    parser.add_argument("--output-path", default=None, help="自定义输出路径（可选）")
    args = parser.parse_args()
    compute_norm_stats(args.config_name, args.dataset_path, args.output_path)
