#!/usr/bin/env python3
"""训练启动器 - 在调用 train_pytorch.py 之前注册自定义配置和 patch"""

import sys
import os
import json
import dataclasses
from pathlib import Path

# 添加路径
_MY_DIR = "/share/0xyj/model3_openpi0.5/my_pi0_training"
_OPENPI_DIR = "/share/0xyj/model3_openpi0.5/openpi-main"
_OPENPI_SRC = f"{_OPENPI_DIR}/src"
_DATASET_DIR = "/share/0xyj/model3_openpi0.5/lerobot_dataset"

for _p in [_MY_DIR, _OPENPI_SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ============================================================================
# Patch: 修复旧版 lerobot 数据集格式兼容问题
# ============================================================================
_patch_applied = False
def _apply_lerobot_patch():
    """Patch lerobot_dataset 来支持旧版数据路径格式"""
    global _patch_applied
    if _patch_applied:
        return
    
    try:
        import lerobot.common.datasets.lerobot_dataset as _lr
        
        def _patched_get_data_file_path(self, ep_index: int) -> Path:
            cs = self.info.get('chunks_size', 1000)
            ep_chunk = ep_index // cs
            file_index = ep_index % cs
            return Path(f"data/chunk-{ep_chunk:03d}/file-{file_index:03d}.parquet")
        
        def _patched_get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
            cs = self.info.get('chunks_size', 1000)
            ep_chunk = ep_index // cs
            file_index = ep_index % cs
            return Path(f"videos/{vid_key}/chunk-{ep_chunk:03d}/file-{file_index:03d}.mp4")
        
        _lr.LeRobotDatasetMetadata.get_data_file_path = _patched_get_data_file_path
        _lr.LeRobotDatasetMetadata.get_video_file_path = _patched_get_video_file_path
        print("[PATCH] lerobot_dataset 已修复：支持旧版数据路径格式")
        _patch_applied = True
    except Exception as e:
        print(f"[PATCH] lerobot_dataset patch 失败: {e}")

_apply_lerobot_patch()

# ============================================================================
# Patch: transform_dataset 以支持从 stats.json 加载 norm_stats
# ============================================================================
_patch_stats_applied = False
_norm_stats = None

def _apply_stats_patch():
    """Patch transform_dataset 来从 stats.json 加载 norm_stats"""
    global _patch_stats_applied, _norm_stats
    if _patch_stats_applied:
        return
    
    # 加载 norm_stats
    stats_path = Path(_DATASET_DIR) / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            raw = json.load(f)
        
        import numpy as np
        import openpi.shared.normalize as _normalize
        
        _norm_stats = {}
        for key, stats in raw.items():
            if isinstance(stats, dict) and "mean" in stats:
                mean = np.array(stats["mean"], dtype=np.float32)
                std = np.array(stats.get("std", [0.0] * len(mean)), dtype=np.float32)
                min_val = np.array(stats.get("min", mean), dtype=np.float32)
                max_val = np.array(stats.get("max", mean), dtype=np.float32)
                std = np.where(std == 0, 1e-6, std)
                _norm_stats[key] = _normalize.NormStats(
                    mean=mean, std=std, min=min_val, max=max_val,
                    q01=min_val, q99=max_val,
                )
        print(f"[INFO] 从 stats.json 加载了 {len(_norm_stats)} 个 norm_stats")
    
    import openpi.training.data_loader as _data_loader
    import openpi.training.config as _config
    
    _orig_transform = _data_loader.transform_dataset
    
    def _patched_transform(dataset, data_config, *, skip_norm_stats=False):
        if _norm_stats is not None and data_config.norm_stats is None:
            # 使用 dataclasses.replace 来创建新的 DataConfig
            data_config = dataclasses.replace(data_config, norm_stats=_norm_stats)
        return _orig_transform(dataset, data_config, skip_norm_stats=skip_norm_stats)
    
    _data_loader.transform_dataset = _patched_transform
    print("[PATCH] transform_dataset 已修复：支持从 stats.json 加载 norm_stats")
    _patch_stats_applied = True

_apply_stats_patch()

# 注册自定义配置
import openpi.training.config as _config
from my_config import get_my_configs

for _cfg in get_my_configs():
    if _cfg.name not in _config._CONFIGS_DICT:
        _config._CONFIGS.append(_cfg)
        _config._CONFIGS_DICT[_cfg.name] = _cfg
print(f"[INFO] 已注册自定义配置: {_cfg.name}")

# 执行原始 train_pytorch.py
_original_script = f"{_OPENPI_DIR}/scripts/train_pytorch.py"
exec(open(_original_script).read())
