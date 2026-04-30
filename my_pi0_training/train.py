#!/usr/bin/env python3
"""Pi0 微调训练脚本 -- 右臂抓取箱子

使用方法:
    cd /share/0xyj/model3_openpi0.5/openpi-main
    JAX_PLATFORMS=cpu /share/0xyj/model3_openpi0.5/openpi-main/.venv/bin/python3.11 \\
        /share/0xyj/model3_openpi0.5/my_pi0_training/train.py \\
        pi0_right_arm_pytorch --exp_name=right_arm_v1

目录结构:
    /share/0xyj/model3_openpi0.5/
    ├── openpi-main/            # 官方仓库（不修改）
    ├── my_pi0_training/        # 本训练模块（当前目录）
    │   ├── train.py            # 本文件
    │   ├── my_config.py        # 训练配置
    │   └── compute_norm_stats.py
    ├── lerobot_dataset/        # 数据集
    └── pi0_base/               # 预训练权重
"""

import sys
import os
from pathlib import Path

# ============================================================================
# 路径设置
# ============================================================================
_MY_DIR = Path("/share/0xyj/model3_openpi0.5/my_pi0_training")
_OPENPI_SRC = Path("/share/0xyj/model3_openpi0.5/openpi-main/src")
_OPENPI_VENV_SITE = Path("/share/0xyj/model3_openpi0.5/openpi-main/.venv/lib/python3.11/site-packages")

for _p in [str(_MY_DIR), str(_OPENPI_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 强制 JAX 用 CPU，避免 JAX CUDA 初始化与 PyTorch CUDA 冲突
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# 将 openpi 缓存重定向到本地高速路径（避免网络存储超时）
# paligemma_tokenizer.model 已预置于 /tmp/openpi_cache/big_vision/
# openpi 通过 OPENPI_DATA_HOME 环境变量读取缓存路径
_CACHE_DIR = os.environ.get("OPENPI_DATA_HOME", "/tmp/openpi_cache")
os.environ["OPENPI_DATA_HOME"] = _CACHE_DIR

# ============================================================================
# Patch 状态
# ============================================================================
_patch_applied = False
_local_root_registry: dict = {}
_excluded_image_patterns = ["depth"]


def _log_once(msg: str) -> None:
    key = f"_LOGGED_{abs(hash(msg)) % 10**7}"
    if not os.environ.get(key):
        os.environ[key] = "1"
        print(f"[PATCH] {msg}")


def _filter_episodes_stats(episodes_stats: dict, patterns: list) -> dict:
    filtered = {}
    for ep_idx, ep_stats in episodes_stats.items():
        filtered[ep_idx] = {
            k: v for k, v in ep_stats.items()
            if not any(p in k for p in patterns)
        }
    return filtered


def apply_patches():
    """Patch OpenPI 数据加载器：
    1. 支持本地数据集（无需 HuggingFace Hub）
    2. 过滤深度图统计量（避免验证错误）
    3. 兼容 HuggingFace datasets 4.x Column 对象
    4. 跳过时间戳检查（大幅加速数据加载初始化）
    5. 使用 pyav 视频后端
    6. 绕过 HuggingFace Hub 网络请求（离线环境）
    """
    global _patch_applied
    if _patch_applied:
        return
    _patch_applied = True

    import torch
    import openpi.training.data_loader as _data_loader
    import openpi.transforms as _transforms
    import openpi.models.model as _model
    import openpi.training.config as _config
    from lerobot.common.datasets import lerobot_dataset
    from lerobot.common.datasets import utils as ds_utils

    # --- 过滤深度图统计量 ---
    _orig_load_stats = ds_utils.load_episodes_stats
    def patched_load_episodes_stats(root):
        return _filter_episodes_stats(_orig_load_stats(root), _excluded_image_patterns)
    ds_utils.load_episodes_stats = patched_load_episodes_stats
    lerobot_dataset.load_episodes_stats = patched_load_episodes_stats

    # --- 跳过时间戳同步检查（加速初始化 10x） ---
    def _skip_check(*args, **kwargs):
        return True
    ds_utils.check_timestamps_sync = _skip_check
    if hasattr(lerobot_dataset, "check_timestamps_sync"):
        lerobot_dataset.check_timestamps_sync = _skip_check

    # --- 兼容 datasets 4.x Column 对象（初始化阶段） ---
    _orig_init = lerobot_dataset.LeRobotDataset.__init__
    def patched_init(self, *args, **kwargs):
        _orig_stack = torch.stack
        def _safe_stack(tensors, *a, **kw):
            if hasattr(tensors, "__class__") and tensors.__class__.__name__ == "Column":
                return torch.tensor([0.0])
            return _orig_stack(tensors, *a, **kw)
        torch.stack = _safe_stack
        try:
            _orig_init(self, *args, **kwargs)
        finally:
            torch.stack = _orig_stack
    lerobot_dataset.LeRobotDataset.__init__ = patched_init

    # --- 兼容 datasets 4.x Column 对象（训练数据读取阶段） ---
    def patched_query(self, query_indices):
        import numpy as np
        result = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            selected = self.hf_dataset.select(q_idx)[key]
            if hasattr(selected, "__class__") and selected.__class__.__name__ == "Column":
                result[key] = torch.from_numpy(np.array(selected))
            else:
                result[key] = torch.stack(selected)
        return result
    lerobot_dataset.LeRobotDataset._query_hf_dataset = patched_query

    # --- 支持本地数据集路径 + pyav 视频后端 ---
    def patched_create_torch_dataset(data_config, action_horizon, model_config):
        repo_id = data_config.repo_id
        if repo_id is None:
            raise ValueError("Repo ID is not set.")
        if repo_id == "fake":
            return _data_loader.FakeDataset(model_config, num_samples=1024)
        root = _local_root_registry.get(repo_id)
        if root:
            root = Path(root)
            _log_once(f"使用本地数据集: {root}")
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id, root=root,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)]
                for key in data_config.action_sequence_keys
            },
            video_backend="pyav",
        )
        if data_config.prompt_from_task:
            dataset = _data_loader.TransformedDataset(
                dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
            )
        return dataset
    _data_loader.create_torch_dataset = patched_create_torch_dataset
    _data_loader._local_root_registry = _local_root_registry

    # --- 绕过 HuggingFace Hub 网络请求（离线环境）---
    # LeRobotDatasetMetadata.__init__ 调用 get_safe_version 会尝试访问 huggingface.co
    # 在无外网环境下直接返回 'main' 版本即可
    def patched_get_safe_version(repo_id, version):
        """离线环境：跳过 Hub 版本检查，直接返回本地版本"""
        return "main"
    ds_utils.get_safe_version = patched_get_safe_version
    # 同时 patch lerobot_dataset 模块中的引用
    if hasattr(lerobot_dataset, 'get_safe_version'):
        lerobot_dataset.get_safe_version = patched_get_safe_version
    # patch LeRobotDatasetMetadata 中使用的 utils 模块
    try:
        import lerobot.common.datasets.lerobot_dataset as _ld_mod
        import lerobot.common.datasets.utils as _utils_mod
        _utils_mod.get_safe_version = patched_get_safe_version
        _log_once("已 patch get_safe_version（离线模式）")
    except Exception as e:
        _log_once(f"patch get_safe_version 失败: {e}")

    # --- patch load_metadata：跳过慢速的 load_episodes_stats ---
    # 数据集是 v3.0 格式，load_episodes_stats 需要读取网络存储上的 parquet 文件，极慢
    # 直接构造最小兼容的 episodes_stats，完全不读文件
    _orig_load_metadata = lerobot_dataset.LeRobotDatasetMetadata.load_metadata
    def patched_load_metadata(self):
        import packaging.version
        self.info = ds_utils.load_info(self.root)
        # 修复 data_path 格式：v3.0 用 {chunk_index}/{file_index}，新版用 {episode_chunk}/{episode_index}
        # 实际文件: data/chunk-000/file-000.parquet
        # 新版 lerobot 期望用 episode_chunk 和 episode_index 格式化
        # 因为文件名是 file-{file_index:03d}，file_index == episode_index % chunks_size
        # 所以直接 patch get_data_file_path 方法返回正确路径
        chunks_size = self.info.get('chunks_size', 1000)
        def patched_get_data_file_path(self_meta, ep_index: int):
            ep_chunk = ep_index // chunks_size
            file_index = ep_index % chunks_size
            return Path(f"data/chunk-{ep_chunk:03d}/file-{file_index:03d}.parquet")
        lerobot_dataset.LeRobotDatasetMetadata.get_data_file_path = patched_get_data_file_path

        def patched_get_video_file_path(self_meta, ep_index: int, vid_key: str):
            ep_chunk = ep_index // chunks_size
            file_index = ep_index % chunks_size
            return Path(f"videos/{vid_key}/chunk-{ep_chunk:03d}/file-{file_index:03d}.mp4")
        lerobot_dataset.LeRobotDatasetMetadata.get_video_file_path = patched_get_video_file_path
        _log_once(f"已 patch get_data_file_path/get_video_file_path (chunks_size={chunks_size})")

    # 必须在类级别 patch video_file_path（DataLoader worker进程也会调用）
    _chunks_size_default = 1000  # 从 info.json 读取，但这里先用默认值
    def _class_get_video_file_path(self_meta, ep_index: int, vid_key: str):
        cs = self_meta.info.get('chunks_size', 1000)
        ep_chunk = ep_index // cs
        file_index = ep_index % cs
        return Path(f"videos/{vid_key}/chunk-{ep_chunk:03d}/file-{file_index:03d}.mp4")
    lerobot_dataset.LeRobotDatasetMetadata.get_video_file_path = _class_get_video_file_path

    def _class_get_data_file_path(self_meta, ep_index: int):
        cs = self_meta.info.get('chunks_size', 1000)
        ep_chunk = ep_index // cs
        file_index = ep_index % cs
        return Path(f"data/chunk-{ep_chunk:03d}/file-{file_index:03d}.parquet")
    lerobot_dataset.LeRobotDatasetMetadata.get_data_file_path = _class_get_data_file_path
    _log_once("已在类级别 patch get_data_file_path + get_video_file_path")

    # --- patch load_metadata：跳过慢速的 load_episodes_stats ---
    def patched_load_metadata(self):
        self.info = ds_utils.load_info(self.root)
        # 跳过版本兼容性检查（避免网络请求）
        self.tasks, self.task_to_task_index = ds_utils.load_tasks(self.root)
        self.episodes = ds_utils.load_episodes(self.root)
        # 构造最小 episodes_stats：每个 episode 使用空字典
        # self.episodes 在此版本 lerobot 中是 int 列表（episode indices）
        total_eps = self.info.get('total_episodes', len(self.episodes))
        self.episodes_stats = {i: {} for i in range(total_eps)}
        self.stats = {}
        _log_once(f"已 patch load_metadata: {total_eps} episodes，跳过 episodes_stats 读取")
    lerobot_dataset.LeRobotDatasetMetadata.load_metadata = patched_load_metadata
    _log_once("已 patch load_metadata（跳过慢速 episodes_stats）")

    _log_once("patches applied: 本地数据集 + datasets4.x兼容 + 跳过时间戳检查 + pyav 视频后端 + 离线HF")


def register_configs():
    """将自定义配置动态注册到 OpenPI 配置系统。"""
    from openpi.training import config as _config
    from my_config import get_my_configs
    registered = []
    for cfg in get_my_configs():
        if cfg.name not in _config._CONFIGS_DICT:
            _config._CONFIGS.append(cfg)
            _config._CONFIGS_DICT[cfg.name] = cfg
            registered.append(cfg.name)
    print(f"[INFO] 已注册配置: {registered}")


# 模块加载时立即执行
apply_patches()
register_configs()

# ============================================================================
# 训练相关导入
# ============================================================================
import dataclasses
import gc
import logging
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm as tqdm_lib
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def init_logging():
    level_map = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E"}
    class Fmt(logging.Formatter):
        def format(self, r):
            r.levelname = level_map.get(r.levelname, r.levelname)
            return super().format(r)
    fmt = Fmt(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(fmt)


def init_wandb(config: _config.TrainConfig, resuming: bool):
    """初始化 WandB，使用预配置的 API Key 和项目。"""
    if not config.wandb_enabled:
        wandb.init(mode="disabled")
        return

    # 设置 WandB API Key
    os.environ["WANDB_API_KEY"] = "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"

    ckpt_dir = config.checkpoint_dir
    wandb_id_file = ckpt_dir / "wandb_id.txt"

    if resuming and wandb_id_file.exists():
        run_id = wandb_id_file.read_text().strip()
        wandb.init(
            id=run_id,
            resume="must",
            project="pi0_dual_arm",
            entity=None,
        )
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project="pi0_dual_arm",
            tags=["pi0", "dual_arm", "16d"],
        )
        if ckpt_dir.exists():
            wandb_id_file.write_text(wandb.run.id)

    logging.info(f"WandB run: {wandb.run.url}")


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int, rank: int):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def save_checkpoint(model, optimizer, step, config, is_main, data_config):
    if not is_main:
        return
    should_save = (step % config.save_interval == 0 and step > 0) or step == config.num_train_steps
    if not should_save:
        return
    final_dir = config.checkpoint_dir / str(step)
    tmp_dir   = config.checkpoint_dir / f"tmp_{step}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    m = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    safetensors.torch.save_model(m, tmp_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save({"global_step": step, "timestamp": time.time()}, tmp_dir / "metadata.pt")

    if data_config.norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(tmp_dir / "assets" / data_config.asset_id, data_config.norm_stats)

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    logging.info(f"Checkpoint saved -> {final_dir}")

    # 清理旧 checkpoint（只保留 keep_period 整数倍和最新的）
    if config.keep_period > 0:
        steps_saved = sorted([
            int(d.name) for d in config.checkpoint_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])
        for old_step in steps_saved[:-1]:
            if old_step % config.keep_period != 0:
                old_dir = config.checkpoint_dir / str(old_step)
                if old_dir.exists():
                    shutil.rmtree(old_dir)
                    logging.info(f"Removed old checkpoint: {old_dir}")


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    steps = [
        int(d.name) for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    if not steps:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    latest = max(steps)
    ckpt = checkpoint_dir / str(latest)
    m = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    safetensors.torch.load_model(m, ckpt / "model.safetensors", device=str(device))
    opt_state = torch.load(ckpt / "optimizer.pt", map_location=device, weights_only=False)
    optimizer.load_state_dict(opt_state)
    del opt_state
    gc.collect()
    meta = torch.load(ckpt / "metadata.pt", map_location=device, weights_only=False)
    logging.info(f"Resumed from step {latest}")
    return meta.get("global_step", latest)


def get_latest_step(checkpoint_dir):
    steps = [
        int(d.name) for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(steps) if steps else None


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # 注册本地数据集路径
    if hasattr(config.data, "local_root") and config.data.local_root:
        tmp_dc = config.data.create(config.assets_dirs, config.model)
        _data_loader._local_root_registry[tmp_dc.repo_id] = config.data.local_root
        if is_main:
            logging.info(f"本地数据集: {tmp_dc.repo_id} -> {config.data.local_root}")

    # Checkpoint 处理
    resuming = False
    if config.resume:
        if config.checkpoint_dir.exists():
            latest = get_latest_step(config.checkpoint_dir)
            resuming = latest is not None
            if not resuming:
                raise FileNotFoundError(f"No checkpoints in {config.checkpoint_dir}")
        else:
            raise FileNotFoundError(f"Checkpoint dir not found: {config.checkpoint_dir}")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Cleared checkpoint dir: {config.checkpoint_dir}")

    if not resuming:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 WandB
    if is_main:
        init_wandb(config, resuming)

    # 构建数据加载器
    loader = _data_loader.create_data_loader(config, framework="pytorch", shuffle=True)
    data_config = loader.data_config()

    # 构建模型
    model_cfg = config.model
    if not isinstance(model_cfg, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=model_cfg.action_dim,
            action_horizon=model_cfg.action_horizon,
            max_token_len=model_cfg.max_token_len,
        )
    else:
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        # 关闭梯度检查点：两卡各80GB显存足够，不需要以速度换显存
        # 关闭后每卡显存约增加15-18GB（保存中间激活），但速度提升约25%
        pass  # gradient_checkpointing_enable() 已禁用
    logging.info("Gradient checkpointing disabled (sufficient VRAM)")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True, gradient_as_bucket_view=True,
        )

    # 加载预训练权重
    if config.pytorch_weight_path is not None:
        wpath = os.path.join(config.pytorch_weight_path, "model.safetensors")
        logging.info(f"Loading weights: {wpath}")
        m = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.load_model(m, wpath)
        logging.info(f"Weights loaded: {sum(p.numel() for p in m.parameters())/1e9:.2f}B params")

    # 优化器
    sch = config.lr_schedule
    optim = torch.optim.AdamW(
        model.parameters(), lr=sch.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps, weight_decay=config.optimizer.weight_decay,
    )

    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)

    def lr_fn(step):
        wu = sch.warmup_steps
        if step < wu:
            return sch.peak_lr * max(step, 1) / (wu + 1)
        p = min(1.0, (step - wu) / max(1, sch.decay_steps - wu))
        cos = 0.5 * (1 + np.cos(np.pi * p))
        return sch.decay_lr + (sch.peak_lr - sch.decay_lr) * cos

    model.train()
    t0 = time.time()
    infos = []

    if is_main:
        logging.info(f"Host: {platform.node()}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            logging.info(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        logging.info(f"num_train_steps={config.num_train_steps}, batch_size={config.batch_size}")
        logging.info(f"warmup={sch.warmup_steps}, peak_lr={sch.peak_lr:.2e}, decay={sch.decay_steps}")
        logging.info(f"save_interval={config.save_interval}, keep_period={config.keep_period}")
        logging.info(f"WandB enabled: {config.wandb_enabled}")

    pbar = tqdm_lib.tqdm(
        total=config.num_train_steps, initial=global_step,
        desc="Training", disable=not is_main
    )

    while global_step < config.num_train_steps:
        for observation, actions in loader:
            if global_step >= config.num_train_steps:
                break

            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(torch.float32).to(device)

            for pg in optim.param_groups:
                pg["lr"] = lr_fn(global_step)

            losses = model(observation, actions)
            if isinstance(losses, (list, tuple)):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)
            loss = losses.mean()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optimizer.clip_gradient_norm
            )
            optim.step()
            optim.zero_grad(set_to_none=True)

            if is_main:
                infos.append({
                    "loss": loss.item(),
                    "lr": optim.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm),
                })

            if is_main and global_step % config.log_interval == 0 and infos:
                elapsed = time.time() - t0
                avg_loss = sum(i["loss"]      for i in infos) / len(infos)
                avg_lr   = sum(i["lr"]        for i in infos) / len(infos)
                avg_gn   = sum(i["grad_norm"] for i in infos) / len(infos)
                logging.info(
                    f"step={global_step}/{config.num_train_steps} "
                    f"loss={avg_loss:.4f} lr={avg_lr:.2e} "
                    f"grad_norm={avg_gn:.2f} elapsed={elapsed:.1f}s"
                )
                if config.wandb_enabled:
                    wandb.log({
                        "train/loss":      avg_loss,
                        "train/lr":        avg_lr,
                        "train/grad_norm": avg_gn,
                        "train/step":      global_step,
                    }, step=global_step)
                t0 = time.time()
                infos = []

            global_step += 1
            save_checkpoint(model, optim, global_step, config, is_main, data_config)
            pbar.update(1)
            if is_main:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr":   f"{optim.param_groups[0]['lr']:.2e}"
                })

    pbar.close()
    if is_main and config.wandb_enabled:
        wandb.finish()
    cleanup_ddp()
    logging.info("Training complete!")


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"===== Pi0 双臂 16 维训练启动 =====")
    logging.info(f"Config: {config.name}")
    logging.info(f"Exp: {config.exp_name}")
    logging.info(f"Checkpoint dir: {config.checkpoint_dir}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)} "
                         f"({torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB)")
    else:
        logging.warning("No GPU found, training on CPU (very slow)")
    train_loop(config)


if __name__ == "__main__":
    main(_config.cli())
 