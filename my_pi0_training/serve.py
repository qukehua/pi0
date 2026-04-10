#!/usr/bin/env python3
"""Pi0 右臂抓取 —— WebSocket 推理服务

用法:
    cd /share/0xyj/model3_openpi0.5/openpi-main
    .venv/bin/python ../my_pi0_training/serve.py --checkpoint_step 15000 --port 8000
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pi0 推理服务")
    parser.add_argument("--checkpoint_step", type=int, default=15000,
                        help="使用哪个步数的 checkpoint，默认 15000")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket 端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    args = parser.parse_args()

    # ── 注册自定义配置 ──────────────────────────────────────────────────────
    from my_config import get_my_configs
    from openpi.training import config as _config
    for cfg in get_my_configs():
        _config.register(cfg)
    logger.info("已注册配置: pi0_right_arm_pytorch")

    # ── 加载训练配置 ────────────────────────────────────────────────────────
    train_config = _config.get_config("pi0_right_arm_pytorch")

    checkpoint_dir = (
        Path(train_config.checkpoint_base_dir)
        / train_config.name
        / "right_arm_box_pick_v1"
        / str(args.checkpoint_step)
    )
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_dir}")
    logger.info(f"加载 checkpoint: {checkpoint_dir}")

    # ── 构建推理 Policy ─────────────────────────────────────────────────────
    from openpi.policies import policy_config as _policy_config
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir=str(checkpoint_dir),
    )
    logger.info("Policy 加载完成，启动服务...")

    # ── 启动 WebSocket 服务 ─────────────────────────────────────────────────
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            "config_name": "pi0_right_arm_pytorch",
            "checkpoint_step": args.checkpoint_step,
            "prompt": "the right arm picks up the box and places it back to its original position",
            "action_dim": 8,
            "action_horizon": 50,
        },
    )
    logger.info(f"推理服务已启动: ws://{args.host}:{args.port}")
    logger.info("等待客户端连接...")
    server.serve_forever()


if __name__ == "__main__":
    main()
