#!/usr/bin/env python3
"""Compute OpenPI norm_stats for the right-arm head-camera config.

This version intentionally matches the official OpenPI stats path:
raw LeRobot sample -> repack transforms -> data transforms -> RunningStats.

For pi0_right_arm_head_cam, data transforms include DeltaActions, so the
saved action statistics are computed in delta action space for the first
7 joint dimensions, while the gripper dimension remains absolute.
"""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_MY_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MY_DIR.parent
_OPENPI_SRC = _PROJECT_ROOT / "openpi-main" / "src"

for _p in [str(_MY_DIR), str(_OPENPI_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the training-time patches for local LeRobot loading and offline mode.
import train_right_arm_head_cam

train_right_arm_head_cam.apply_patches()
_local_root_registry = train_right_arm_head_cam._local_root_registry

import numpy as np
import tqdm

import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms
from my_config_right_arm_head_cam import get_my_configs


class RemoveStrings(_transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def register_configs() -> None:
    for cfg in get_my_configs():
        if cfg.name not in _config._CONFIGS_DICT:
            _config._CONFIGS.append(cfg)
            _config._CONFIGS_DICT[cfg.name] = cfg


register_configs()


def compute_norm_stats(config_name: str, dataset_path: str, output_path: str | None = None) -> str:
    print(f"[INFO] Computing norm_stats: config={config_name}")
    print(f"[INFO] Dataset path: {dataset_path}")

    if config_name not in _config._CONFIGS_DICT:
        raise ValueError(f"Config '{config_name}' not found. Available: {list(_config._CONFIGS_DICT.keys())}")

    config = _config._CONFIGS_DICT[config_name]
    data_config = config.data.create(config.assets_dirs, config.model)
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Data config repo_id is None")

    _data_loader._local_root_registry[repo_id] = dataset_path
    _local_root_registry[repo_id] = dataset_path
    print(f"[INFO] Registered local dataset: {repo_id} -> {dataset_path}")

    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )

    batch_size = config.batch_size
    num_batches = len(dataset) // batch_size
    if num_batches < 1:
        raise RuntimeError(f"Dataset has {len(dataset)} samples, smaller than batch_size={batch_size}")

    print(f"[INFO] Dataset size: {len(dataset)} samples")
    print("[INFO] Stats are computed after repack/data transforms; actions are in delta space.")

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        num_batches=num_batches,
        framework="torch",
    )

    keys = ["state", "actions"]
    stats = {key: _normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stat.get_statistics() for key, stat in stats.items()}

    if output_path is None:
        out_dir = config.assets_dirs / repo_id
    else:
        out_dir = Path(output_path)

    _normalize.save(out_dir, norm_stats)
    out_file = out_dir / "norm_stats.json"

    print(f"[INFO] Saved norm_stats: {out_file}")
    print(f"[INFO] state  mean: {[f'{v:.4f}' for v in norm_stats['state'].mean]}")
    print(f"[INFO] action mean: {[f'{v:.4f}' for v in norm_stats['actions'].mean]}")
    print("[INFO] action dims 0-6 are delta stats; dim 7 remains absolute gripper stats.")
    return str(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute OpenPI norm_stats for the head-camera config")
    parser.add_argument("--config-name", default="pi0_right_arm_head_cam", help="Training config name")
    parser.add_argument("--dataset-path", default="/share/0xyj/model3_openpi0.5/lerobot_dataset", help="Dataset path")
    parser.add_argument("--output-path", default=None, help="Optional output directory")
    args = parser.parse_args()
    compute_norm_stats(args.config_name, args.dataset_path, args.output_path)
