#!/usr/bin/env python3
"""
Realman 机械臂 OpenPI PyTorch 训练脚本

使用方法:
    # 在 openpi 目录下运行
    cd openpi
    uv run ../realman_openpi_training/train.py pi0_realman_pytorch --exp_name=my_exp

特点:
    - 使用 PyTorch 进行训练（显存更低，支持梯度检查点）
    - 完全独立于 OpenPI 官方源码，无需修改任何官方文件
    - 通过动态注册机制将自定义配置注入到 OpenPI 配置系统
    - 支持本地数据集加载（无需从 HuggingFace Hub 下载）
    - 自动过滤深度图统计量，避免验证错误
    - 支持单卡/多卡训练（DDP）
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

# 日志去重（使用环境变量跨进程共享状态）
import multiprocessing

def _is_main_process() -> bool:
    """检查是否为主进程（非 DataLoader worker）"""
    # DataLoader worker 进程会设置这个环境变量
    worker_info = None
    try:
        from torch.utils.data import get_worker_info
        worker_info = get_worker_info()
    except ImportError:
        pass
    return worker_info is None

def _log_once(message: str, level: str = "INFO") -> None:
    """只在主进程打印一次的日志函数"""
    if not _is_main_process():
        return
    # 使用环境变量标记已打印的消息
    env_key = f"_LOGGED_{hash(message) % 1000000}"
    if os.environ.get(env_key):
        return
    os.environ[env_key] = "1"
    print(f"[{level}] {message}")

# Patch 相关变量
_excluded_image_patterns = ["depth"]
_local_root_registry = {}

# 使用模块级变量标记 patch 状态（每个进程独立）
# 注意：不使用环境变量，因为 fork 子进程会继承环境变量，
# 但不会继承 Python 模块级别的对象修改
_patch_applied_in_this_process = False

def _is_patch_applied() -> bool:
    """检查 patch 是否已在当前进程中应用"""
    return _patch_applied_in_this_process

def _mark_patch_applied():
    """标记 patch 已在当前进程中应用"""
    global _patch_applied_in_this_process
    _patch_applied_in_this_process = True


def _filter_episodes_stats(episodes_stats: dict, excluded_patterns: list[str]) -> dict:
    """过滤 episodes_stats，移除匹配排除模式的 key"""
    if not excluded_patterns:
        return episodes_stats
    
    filtered = {}
    for ep_idx, ep_stats in episodes_stats.items():
        filtered_ep = {}
        for key, stats in ep_stats.items():
            should_exclude = any(pattern in key for pattern in excluded_patterns)
            if not should_exclude:
                filtered_ep[key] = stats
        filtered[ep_idx] = filtered_ep
    
    return filtered


def patch_data_loader_for_local_datasets():
    """Patch OpenPI 数据加载器，支持本地数据集 + 过滤深度图统计量 + 修复 datasets 4.x 兼容性
    
    注意：此 patch 必须在所有进程中执行（包括 DataLoader worker 进程），
    因为 _query_hf_dataset 会在 worker 进程中被调用。
    """
    if _is_patch_applied():
        return
    _mark_patch_applied()
    
    import torch
    import openpi.training.data_loader as _data_loader
    from lerobot.common.datasets import lerobot_dataset
    from lerobot.common.datasets import utils as ds_utils
    from torch.utils.data import Dataset
    import openpi.transforms as _transforms
    import openpi.models.model as _model
    import openpi.training.config as _config
    
    _original_create_torch_dataset = _data_loader.create_torch_dataset
    _original_load_episodes_stats = ds_utils.load_episodes_stats
    _original_lerobot_init = lerobot_dataset.LeRobotDataset.__init__
    _filtered_keys_logged = set()
    
    def patched_load_episodes_stats(root):
        episodes_stats = _original_load_episodes_stats(root)
        filtered = _filter_episodes_stats(episodes_stats, _excluded_image_patterns)
        
        if episodes_stats and filtered:
            first_ep_original = list(episodes_stats.values())[0]
            first_ep_filtered = list(filtered.values())[0]
            removed_keys = set(first_ep_original.keys()) - set(first_ep_filtered.keys())
            if removed_keys:
                key_str = str(removed_keys)
                if key_str not in _filtered_keys_logged:
                    _filtered_keys_logged.add(key_str)
                    _log_once(f"已过滤统计量 key: {removed_keys}")
        
        return filtered
    
    ds_utils.load_episodes_stats = patched_load_episodes_stats
    lerobot_dataset.load_episodes_stats = patched_load_episodes_stats
    
    # 跳过时间戳检查（加速数据加载）
    _original_check_timestamps_sync = ds_utils.check_timestamps_sync
    
    def patched_check_timestamps_sync(*args, **kwargs):
        """跳过时间戳同步检查，大幅加速数据加载"""
        return True
    
    ds_utils.check_timestamps_sync = patched_check_timestamps_sync
    # 同时 patch lerobot_dataset 模块中的引用
    if hasattr(lerobot_dataset, 'check_timestamps_sync'):
        lerobot_dataset.check_timestamps_sync = patched_check_timestamps_sync
    
    def patched_lerobot_init(self, *args, **kwargs):
        """修复 LeRobotDataset.__init__ 中的问题：
        
        1. 兼容 HuggingFace datasets 4.x 的 Column 对象
        2. 跳过时间戳检查相关的数据加载（大幅加速初始化）
        """
        import numpy as np
        
        # 保存原始的 torch.stack
        original_torch_stack = torch.stack
        
        def patched_torch_stack(tensors, *stack_args, **stack_kwargs):
            """兼容 HuggingFace datasets 4.x 的 Column 对象 + 跳过时间戳相关加载"""
            # 如果是 Column 对象
            if hasattr(tensors, '__class__') and tensors.__class__.__name__ == 'Column':
                # 由于我们已经跳过了 check_timestamps_sync，
                # 返回一个空 tensor 即可，不需要真正加载数据
                # 这是最大的性能优化点
                return torch.tensor([0.0])
            
            # 否则使用原始行为
            return original_torch_stack(tensors, *stack_args, **stack_kwargs)
        
        # 临时替换 torch.stack
        torch.stack = patched_torch_stack
        try:
            _original_lerobot_init(self, *args, **kwargs)
        finally:
            # 恢复原始 torch.stack
            torch.stack = original_torch_stack
    
    lerobot_dataset.LeRobotDataset.__init__ = patched_lerobot_init
    
    # ========== 关键修复：Patch _query_hf_dataset 方法 ==========
    # 这个方法在训练循环中的 __getitem__ 被调用，必须正确处理 Column 对象
    _original_query_hf_dataset = lerobot_dataset.LeRobotDataset._query_hf_dataset
    
    def patched_query_hf_dataset(self, query_indices):
        """修复 _query_hf_dataset 以兼容 HuggingFace datasets 4.x 的 Column 对象
        
        原始代码: torch.stack(self.hf_dataset.select(q_idx)[key])
        问题: datasets 4.x 返回 Column 对象而不是 list of tensors
        修复: 将 Column 转换为 numpy 再转为 tensor
        """
        import numpy as np
        result = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            
            selected_data = self.hf_dataset.select(q_idx)[key]
            
            # 处理 Column 对象（HuggingFace datasets 4.x）
            if hasattr(selected_data, '__class__') and selected_data.__class__.__name__ == 'Column':
                # Column 对象需要转换为 numpy 再转为 tensor
                np_data = np.array(selected_data)
                result[key] = torch.from_numpy(np_data)
            else:
                # 原始行为：假设是 list of tensors
                result[key] = torch.stack(selected_data)
        
        return result
    
    lerobot_dataset.LeRobotDataset._query_hf_dataset = patched_query_hf_dataset
    
    def patched_create_torch_dataset(
        data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
    ) -> Dataset:
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
            repo_id,
            root=root,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)] 
                for key in data_config.action_sequence_keys
            },
            # 使用 pyav 作为视频后端，避免 torchcodec 与 PyTorch 版本不兼容的问题
            video_backend="pyav",
        )
        
        if data_config.prompt_from_task:
            dataset = _data_loader.TransformedDataset(
                dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
            )
        
        return dataset
    
    _data_loader.create_torch_dataset = patched_create_torch_dataset
    _data_loader._local_root_registry = _local_root_registry
    _log_once("已 patch 数据加载器：本地数据集 + 过滤深度图 + datasets 4.x 兼容 + 跳过时间戳检查 + _query_hf_dataset 修复 + pyav 视频后端")


def register_realman_configs():
    """将 Realman 配置动态注册到 OpenPI 配置系统"""
    # 只在主进程执行
    if not _is_main_process():
        return
    
    from openpi.training import config as _config
    from realman_config import get_realman_configs
    
    realman_configs = get_realman_configs()
    registered_count = 0
    for cfg in realman_configs:
        if cfg.name not in _config._CONFIGS_DICT:
            _config._CONFIGS.append(cfg)
            _config._CONFIGS_DICT[cfg.name] = cfg
            registered_count += 1
    
    if registered_count > 0:
        _log_once(f"已注册 {len(realman_configs)} 个 Realman 配置")


# 模块加载时执行 patch 和注册
patch_data_loader_for_local_datasets()
register_realman_configs()


# ============================================================================
# PyTorch 训练逻辑
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
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def init_logging():
    """初始化日志格式"""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """初始化 wandb"""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    """设置分布式训练"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    """清理分布式训练"""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    """设置随机种子"""
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    """构建数据集"""
    data_loader = _data_loader.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """保存检查点"""
    if not is_main:
        return

    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """加载检查点"""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    logging.info("Loading model state...")
    safetensors_path = ckpt_dir / "model.safetensors"

    if safetensors_path.exists():
        model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
        logging.info("Loaded model state from safetensors format")
    else:
        raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

    logging.info("Loading optimizer state...")
    optimizer_path = ckpt_dir / "optimizer.pt"

    if optimizer_path.exists():
        optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
    else:
        raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

    metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
    global_step = metadata.get("global_step", latest_step)

    torch.cuda.empty_cache()
    gc.collect()

    logging.info(f"Successfully loaded checkpoint from step {latest_step}")
    return global_step


def get_latest_checkpoint_step(checkpoint_dir):
    """获取最新检查点步数"""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """记录显存使用情况"""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9

    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    """PyTorch 训练主循环"""
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # 注册 local_root
    if hasattr(config.data, 'local_root') and config.data.local_root:
        data_config_temp = config.data.create(config.assets_dirs, config.model)
        _data_loader._local_root_registry[data_config_temp.repo_id] = config.data.local_root
        if is_main:
            logging.info(f"注册本地数据集路径: {data_config_temp.repo_id} -> {config.data.local_root}")

    # 初始化检查点目录
    resuming = False
    if config.resume:
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(f"Resuming from checkpoint at step {latest_step}")
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir}")
        else:
            raise FileNotFoundError(f"Checkpoint directory {exp_checkpoint_dir} does not exist")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    if not resuming:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created checkpoint directory: {config.checkpoint_dir}")

    # 初始化 wandb
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 构建数据加载器
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(f"Batch size per GPU: {effective_batch_size} (total: {config.batch_size})")

    loader, data_config = build_datasets(config)

    # 构建模型
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    # 启用梯度检查点
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")

    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    # 加载预训练权重
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(
            (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model), model_path
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # 优化器和学习率调度
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # 加载检查点
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []

    if is_main:
        logging.info(f"Running on: {platform.node()} | world_size={world_size}")
        logging.info(f"Training: num_train_steps={config.num_train_steps}, batch_size={config.batch_size}")
        logging.info(f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}")
        logging.info(f"Training precision: {model_cfg.dtype}")

    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            if global_step >= config.num_train_steps:
                break

            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(torch.float32).to(device)

            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            losses = model(observation, actions)
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            optim.step()
            optim.zero_grad(set_to_none=True)

            if is_main:
                infos.append({
                    "loss": loss.item(),
                    "learning_rate": optim.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                })

            if is_main and (global_step % config.log_interval == 0) and len(infos) > 0:
                elapsed = time.time() - start_time
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                avg_grad_norm = sum(info["grad_norm"] for info in infos) / len(infos)

                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                )

                if config.wandb_enabled:
                    wandb.log({
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "grad_norm": avg_grad_norm,
                        "step": global_step,
                    }, step=global_step)

                start_time = time.time()
                infos = []

            global_step += 1
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}"})

    if pbar is not None:
        pbar.close()

    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()
    logging.info("Training completed!")


def main(config: _config.TrainConfig):
    """主函数"""
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info(f"Config: {config.name}")
    
    # 检查 GPU 可用性
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logging.warning("No GPU available, training on CPU (this will be very slow)")
    
    train_loop(config)


if __name__ == "__main__":
    main(_config.cli())
