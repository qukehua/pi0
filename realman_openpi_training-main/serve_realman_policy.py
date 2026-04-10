#!/usr/bin/env python
"""
Realman OpenPI 推理服务启动脚本

此脚本在启动 OpenPI 推理服务前，自动注册 Realman 相关的配置。
同时自动从 checkpoint 目录中检测 asset_id 并加载 norm_stats。

用法：
    cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/openpi
    source .venv/bin/activate
    
    python ../realman_openpi_training/serve_realman_policy.py \
        --config=pi0_realman_inference \
        --checkpoint=/path/to/checkpoint/30000 \
        --prompt="the gripper pick up the yellow banana" \
        --port=8000
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

# 确保 realman_openpi_training 目录在 Python 路径中
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def register_realman_configs():
    """将 Realman 配置动态注册到 OpenPI 配置系统"""
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
        logging.info(f"已注册 {len(realman_configs)} 个 Realman 配置")
    
    return realman_configs


def auto_detect_asset_id(checkpoint_dir: Path) -> Optional[str]:
    """从 checkpoint 目录自动检测 asset_id。
    
    OpenPI 的 checkpoint 结构如下：
        checkpoint_dir/
        ├── model.safetensors
        ├── metadata.pt
        └── assets/
            └── <asset_id>/
                └── norm_stats.json
    
    此函数会扫描 assets 目录下的子目录，找到包含 norm_stats.json 的目录名作为 asset_id。
    """
    assets_dir = checkpoint_dir / "assets"
    if not assets_dir.exists():
        return None
    
    # 扫描 assets 目录下的子目录
    for subdir in assets_dir.iterdir():
        if subdir.is_dir():
            norm_stats_file = subdir / "norm_stats.json"
            if norm_stats_file.exists():
                logging.info(f"自动检测到 asset_id: {subdir.name}")
                return subdir.name
    
    return None


def load_norm_stats_from_checkpoint(checkpoint_dir: Path, asset_id: str) -> dict:
    """从 checkpoint 目录加载 norm_stats。
    
    Args:
        checkpoint_dir: checkpoint 目录路径
        asset_id: 资产 ID（子目录名）
    
    Returns:
        norm_stats 字典
    """
    import openpi.transforms as transforms
    
    norm_stats_path = checkpoint_dir / "assets" / asset_id / "norm_stats.json"
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"找不到 norm_stats.json: {norm_stats_path}")
    
    with open(norm_stats_path, "r") as f:
        raw_data = json.load(f)
    
    # norm_stats.json 格式可能是：
    # 1. {"norm_stats": {"state": {...}, "actions": {...}}}  (新格式)
    # 2. {"state": {...}, "actions": {...}}  (旧格式)
    if "norm_stats" in raw_data:
        raw_stats = raw_data["norm_stats"]
    else:
        raw_stats = raw_data
    
    # 转换为 NormStats 对象
    norm_stats = {}
    for key, value in raw_stats.items():
        norm_stats[key] = transforms.NormStats(
            mean=value["mean"],
            std=value["std"],
            q01=value.get("q01"),
            q99=value.get("q99"),
        )
    
    logging.info(f"已加载 norm_stats (keys: {list(norm_stats.keys())})")
    return norm_stats


def main():
    parser = argparse.ArgumentParser(
        description="Realman OpenPI 推理服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python serve_realman_policy.py \\
        --config=pi0_realman_inference \\
        --checkpoint=/path/to/checkpoint/30000 \\
        --prompt="pick up the banana" \\
        --port=8000
        
可用配置:
    - pi0_realman_inference: π₀ 推理配置
    - pi05_realman_inference: π₀.5 推理配置
"""
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="配置名称 (如 pi0_realman_inference)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="checkpoint 目录路径"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="默认任务指令"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="WebSocket 服务端口 (默认: 8000)"
    )
    parser.add_argument(
        "--record", action="store_true",
        help="记录策略行为用于调试"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, force=True)
    
    # 注册 Realman 配置
    logging.info("正在注册 Realman 配置...")
    register_realman_configs()
    
    # 导入 OpenPI 模块
    from openpi.policies import policy_config as _policy_config
    from openpi.policies import policy as _policy
    from openpi.training import config as _config
    from openpi.serving import websocket_policy_server
    import socket
    
    # 获取配置
    logging.info(f"加载配置: {args.config}")
    train_config = _config.get_config(args.config)
    
    # 自动检测 asset_id 并加载 norm_stats
    checkpoint_dir = Path(args.checkpoint)
    asset_id = auto_detect_asset_id(checkpoint_dir)
    
    norm_stats = None
    if asset_id:
        try:
            norm_stats = load_norm_stats_from_checkpoint(checkpoint_dir, asset_id)
        except Exception as e:
            logging.warning(f"加载 norm_stats 失败: {e}")
            norm_stats = None
    
    if norm_stats is None:
        logging.warning("未找到 norm_stats，将尝试从配置中加载（可能会失败）")
    
    # 创建策略
    logging.info(f"加载 checkpoint: {args.checkpoint}")
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        default_prompt=args.prompt,
        norm_stats=norm_stats,  # 传递手动加载的 norm_stats
    )
    
    # 扩展 metadata，添加训练配置信息
    policy_metadata = dict(policy.metadata) if policy.metadata else {}
    
    # 从 train_config 提取关键配置
    use_delta = False
    default_prompt_from_config = None
    
    if hasattr(train_config, 'data'):
        data_config = train_config.data
        # 提取 use_delta_joint_actions
        if hasattr(data_config, 'use_delta_joint_actions'):
            use_delta = data_config.use_delta_joint_actions
        # 提取 default_prompt（如果命令行没指定，使用配置中的）
        if hasattr(data_config, 'default_prompt'):
            default_prompt_from_config = data_config.default_prompt
    
    # 确定最终使用的 prompt（优先级：命令行 > 配置文件）
    final_prompt = args.prompt if args.prompt else default_prompt_from_config
    
    # 添加到 metadata
    policy_metadata.update({
        "default_prompt": final_prompt,
        "use_delta_actions": use_delta,
        "config_name": args.config,
    })
    
    logging.info(f"策略配置:")
    logging.info(f"  default_prompt: {final_prompt}")
    logging.info(f"  use_delta_actions: {use_delta}")
    logging.info(f"  config_name: {args.config}")
    
    # 记录模式
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")
    
    # 启动服务
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"启动服务 (host: {hostname}, ip: {local_ip}, port: {args.port})")
    
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    
    logging.info(f"OpenPI 推理服务已启动，监听端口 {args.port}")
    logging.info(f"RoboCOIN 可通过 --openpi_remote_host=localhost:{args.port} 连接")
    
    server.serve_forever()


if __name__ == "__main__":
    main()
