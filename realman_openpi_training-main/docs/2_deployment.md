# Realman π₀/π₀.5 部署推理指南

本文档介绍如何部署和运行 OpenPI 模型进行实时推理控制。

> **文档版本**: v1.3  
> **最后更新**: 2026-01-21  
> **适用框架**: OpenPI + RoboCOIN unified_deploy

---

## 📚 文档导航

- **[← 返回索引](./README.md)**
- **[上一步：训练流程](./1_training.md)**
- **[常见问题](./3_faq.md)**

---

## 1. 部署推理概述

由于 OpenPI 和 RoboCOIN 使用不同的 Python 环境（OpenPI 需要 JAX，RoboCOIN 是 PyTorch），OpenPI 模型需要使用**三终端部署架构**：

### 1.1 与 ACT/Diffusion 模型的对比

| 模型类型 | 所需终端数 | 原因 | 启动复杂度 |
|----------|-----------|------|-----------|
| ACT / Diffusion | **2 个** | 单环境部署，policy_server 直接加载模型 | 简单 |
| OpenPI (π₀/π₀.5) | **3 个** | 双环境部署，JAX 和 PyTorch 环境隔离 | 中等 |

### 1.2 三终端部署架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          三终端部署架构                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  终端 1 (openpi)          终端 2 (robocoin)           终端 3 (robocoin)      │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐      │
│  │ serve_realman_  │      │ policy_server   │      │ realman_client  │      │
│  │ policy.py       │      │                 │      │                 │      │
│  │                 │      │                 │      │                 │      │
│  │  π₀ 模型推理     │◄────►│  gRPC Server    │◄────►│  机器人控制      │      │
│  │  :8000 (WS)     │      │  :50051 (gRPC)  │      │  相机采集        │      │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘      │
│         ▲                         ▲                         ▲               │
│         │                         │                         │               │
│    openpi .venv              robocoin conda            robocoin conda       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| 终端 | 环境 | 组件 | 端口 | 作用 |
|------|------|------|------|------|
| 终端 1 | openpi (.venv) | `serve_realman_policy.py` | :8000 (WebSocket) | π₀/π₀.5 模型推理 |
| 终端 2 | robocoin (conda) | `policy_server.py` | :50051 (gRPC) | 协议转换、动作队列 |
| 终端 3 | robocoin (conda) | `realman_client.py` | - | 机器人控制、相机采集 |

## 2. 三终端启动命令

### 2.1 完整启动示例

以下是包含所有必要参数的完整启动命令，可直接复制使用：

**终端 1：启动 OpenPI 推理服务（openpi 环境）**

```bash
# 进入 openpi 目录，激活 .venv
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/openpi
source .venv/bin/activate

# ⚠️ 重要：禁用 Triton 编译（避免首次推理时长时间编译导致 WebSocket 超时）
export PYTORCH_DISABLE_TRITON=1
export TORCH_COMPILE_DISABLE=1

# 使用 Realman 推理脚本（自动注册配置、自动加载 norm_stats）
python ../realman_openpi_training/serve_realman_policy.py \
    --config=pi0_realman_inference \
    --checkpoint=/home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/openpi/checkpoints/pi0_realman_pytorch/realman_0111-130_0120_pi0_pytorch/30000 \
    --prompt="the gripper pick up the yellow banana and put it in the white plate" \
    --port=8000
```

**终端 2：启动 RoboCOIN PolicyServer（robocoin 环境）**

```bash
# 激活 robocoin 环境
conda activate robocoin
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/RoboCOIN

# 启动 policy_server（远程模式，连接 OpenPI 服务）
python -m lerobot.extensions.unified_deploy.server.policy_server \
    --policy_type=openpi \
    --pretrained_path=dummy \
    --openpi_remote_host=localhost:8000 \
    --device=cuda \
    --n_action_steps=3
```

**终端 3：启动 Realman Client（机器人端）**

```bash
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/RoboCOIN

python -m lerobot.extensions.unified_deploy.client.realman_client \
    --robot_ip="192.168.1.18" \
    --server_address=127.0.0.1:50051 \
    --frequency=5 \
    --camera_configs="{ cam0_rgb: {type: intelrealsense, serial_number_or_name: '230422272673', fps: 30, width: 640, height: 480}, cam1_rgb: {type: intelrealsense, serial_number_or_name: '420122071413', fps: 30, width: 640, height: 480} }" \
    --camera_rotations="{ cam0_rgb: 180 }" \
    --visualize \
    --n_action_steps=3 \
    --max_steps=5000
```

### 2.2 参数说明

**终端 1 参数说明**：

| 参数 | 必填 | 说明 | 示例 |
|------|------|------|------|
| `--config` | ✅ | 推理配置名称 | `pi0_realman_inference` |
| `--checkpoint` | ✅ | checkpoint 完整路径 | `/path/to/checkpoints/30000` |
| `--prompt` | ⭐ | 任务指令（推荐指定） | `"pick up the banana"` |
| `--port` | ❌ | WebSocket 端口 | `8000`（默认） |

**环境变量说明**：

| 环境变量 | 作用 | 推荐值 |
|---------|------|--------|
| `PYTORCH_DISABLE_TRITON` | 禁用 Triton JIT 编译 | `1` |
| `TORCH_COMPILE_DISABLE` | 禁用 torch.compile 优化 | `1` |

**为什么需要禁用 Triton？**
- PyTorch 2.0+ 默认在首次推理时进行 Triton JIT 编译（需要 1-2 分钟）
- 编译期间 WebSocket 服务无法响应，导致客户端 keepalive ping 超时
- 禁用后服务立即可用，推理速度仅损失 10-20%
- 详见 [bug_records.md - BUG-002](./bug_records.md#bug-002-pytorch-triton-编译导致-websocket-连接超时)

---

**终端 2 参数说明**：

| 参数 | 必填 | 说明 | 推荐值 |
|------|------|------|--------|
| `--policy_type` | ✅ | 策略类型 | `openpi` |
| `--pretrained_path` | ✅ | 模型路径（远程模式填 `dummy`） | `dummy` |
| `--openpi_remote_host` | ✅ | OpenPI 服务地址 | `localhost:8000` |
| `--device` | ❌ | 推理设备 | `cuda`（默认） |
| `--n_action_steps` | ⭐ | 每次返回的动作步数 | `3-10` |

---

**终端 3 参数说明**：

| 参数 | 必填 | 说明 | 推荐值 |
|------|------|------|--------|
| `--robot_ip` | ✅ | 机械臂 IP 地址 | `192.168.1.18` |
| `--server_address` | ✅ | PolicyServer 地址 | `127.0.0.1:50051` |
| `--frequency` | ⭐ | 控制循环频率（Hz） | `5-10` |
| `--camera_configs` | ✅ | 相机配置（YAML 格式） | 见上方示例 |
| `--camera_rotations` | ❌ | 相机旋转角度 | `{ cam0_rgb: 180 }` |
| `--visualize` | ❌ | 是否显示相机画面 | 调试时启用 |
| `--n_action_steps` | ⭐ | 每次使用的动作步数 | `3-10` |
| `--max_steps` | ❌ | 最大执行步数 | `5000`（默认 2000） |

**⚠️ 重要提示**：
- `n_action_steps` 可以在终端 2 或终端 3 指定，终端 2 优先级更高
- `camera_configs` 中的序列号需要替换为你的实际相机序列号
- `frequency` 和 `n_action_steps` 需要配合调整（推理频率 ≈ frequency / n_action_steps）

## 3. serve_realman_policy.py 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--config` | 配置名称 | `pi0_realman_inference`, `pi05_realman_inference` |
| `--checkpoint` | checkpoint 目录路径 | `/path/to/checkpoints/30000` |
| `--prompt` | 默认任务指令 | `"pick up the banana"` |
| `--port` | WebSocket 服务端口 | `8000` |
| `--record` | 记录策略行为用于调试 | （flag）|

**脚本特性**：
- 自动注册 `realman_config.py` 中定义的所有 Realman 配置
- 自动从 checkpoint 目录检测 `asset_id` 并加载 `norm_stats.json`
- 无需修改 OpenPI 官方源码

## 4. 环境准备

在 RoboCOIN 环境中需要安装 `openpi-client` 包：

```bash
conda activate robocoin
pip install -e /path/to/openpi/packages/openpi-client/
```

这是一个轻量级包，只包含 WebSocket 客户端，不引入 JAX 等重依赖。

## 5. 使用 OpenPI 官方客户端（直接调用）

如果不需要 RoboCOIN 框架，也可以直接使用 OpenPI 官方客户端：

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy

# 连接到 serve_realman_policy.py 服务
policy = WebsocketClientPolicy(host="localhost", port=8000)

# 发送观测，获取动作
action = policy.infer({
    "state": current_state,  # [8,] float32
    "images": {
        "base_0_rgb": base_image,        # [H, W, 3] uint8
        "left_wrist_0_rgb": wrist_image, # [H, W, 3] uint8
    },
    "prompt": "pick up the yellow banana and put it in the white plate"
})
```

## 6. n_action_steps 参数详解

`n_action_steps` 参数控制每次推理后实际执行的动作步数，是平衡响应性和效率的关键参数。

### 6.1 基本概念

| 参数 | 说明 | OpenPI 默认值 | 推荐值 |
|------|------|--------------|--------|
| `action_horizon` | 模型预测的动作序列长度（训练时固定） | 50 | - |
| `action_dim` | 模型内部动作维度（固定） | 32 | - |
| `n_action_steps` | 每次推理后实际返回的动作步数 | 50 | 10-20 |

### 6.2 工作原理

```
┌─────────────────────────────────────────────────────────────┐
│                   n_action_steps 工作流程                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 模型推理                                                 │
│     └─ 输出: [action_horizon, action_dim] = [50, 32]       │
│                                                             │
│  2. 维度截取                                                 │
│     └─ 截取: [:, :8] → [50, 8] (7 joints + 1 gripper)      │
│                                                             │
│  3. 步数截取 (n_action_steps)                               │
│     └─ 截取: [:n_action_steps] → [10, 8]                   │
│                                                             │
│  4. 动作队列                                                 │
│     └─ 放入 action_queue (FIFO 无界队列)                    │
│                                                             │
│  5. 执行循环 (frequency Hz)                                 │
│     ├─ 每个周期从队列取 1 个动作                            │
│     ├─ 调用 SDK: rm_movej_canfd() 或 rm_movep_canfd()      │
│     └─ 队列较空时触发新推理                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 参数选择建议

| n_action_steps | 推理频率 | 响应性 | 效率 | 适用场景 |
|---------------|---------|--------|------|---------|
| 1 | 10 Hz | 最高 | 低 | 需要极快响应的任务 |
| 5 | 2 Hz | 高 | 中 | 快速操作任务 |
| 10 | 1 Hz | 中 | 高 | **推荐：平衡响应性和效率** |
| 20 | 0.5 Hz | 低 | 高 | 慢速、稳定的任务 |
| 50 | 0.2 Hz | 最低 | 最高 | 完全执行预测序列（不推荐） |

**推理频率计算**：
```
推理频率 ≈ frequency / n_action_steps

例如：
- frequency=10, n_action_steps=10 → 推理频率 ≈ 1 Hz
- frequency=10, n_action_steps=5  → 推理频率 ≈ 2 Hz
- frequency=30, n_action_steps=10 → 推理频率 ≈ 3 Hz
```

### 6.4 队列管理机制

**队列触发条件**（`realman_client.py`）：
```python
# 当队列中的动作数 < n_action_steps * 0.5 时，触发新推理
if queue_size < n_action_steps * 0.5:
    send_observation()  # 请求新的推理
```

**队列特性**：
- 类型：`Queue.Queue()`，无界 FIFO 队列
- 线程安全：使用 `action_queue_lock` 保护
- 堆积保护：通过 `chunk_size_threshold=0.5` 控制队列大小

**队列稳定状态**：
- 队列大小稳定在 `n_action_steps * 0.5` 到 `n_action_steps` 之间
- 例如 `n_action_steps=10`，队列大小稳定在 5-10 个动作

### 6.5 与其他模型的对比

| 模型 | action_horizon | 默认 n_action_steps | 推荐 n_action_steps |
|------|---------------|-------------------|-------------------|
| OpenPI (π₀/π₀.5) | 50 | 50 | 10-20 |
| ACT | 100 | 100 | 10-20 |
| Diffusion Policy | 16 | 8 | 4-8 |

### 6.6 指定方式

**方式 1：在 policy_server 中指定（推荐）**
```bash
python -m lerobot.extensions.unified_deploy.server.policy_server \
    --policy_type=openpi \
    --n_action_steps=10  # ← 在 server 端指定
```

**方式 2：在 realman_client 中指定**
```bash
python -m lerobot.extensions.unified_deploy.client.realman_client \
    --n_action_steps=10  # ← 在 client 端指定
```

**优先级**：server 端设置 > client 端设置

### 6.7 调试建议

1. **初次部署**：使用 `n_action_steps=10`，观察动作流畅度
2. **动作卡顿**：减小 `n_action_steps`（如 5），提高推理频率
3. **动作过于激进**：增大 `n_action_steps`（如 20），降低推理频率
4. **监控队列大小**：在日志中观察 `queue_size`，确保不为空也不堆积

## 7. VLA 任务指令（Text Instruction）配置

OpenPI 等 VLA 模型需要文本指令来指导机器人执行任务。任务指令可以在多个位置指定，遵循以下优先级规则：

### 7.1 优先级规则

```
Client --task > Server --prompt > train_config.default_prompt > "complete the task"
     (最高)                                                            (最低)
```

### 7.2 指定方式

**方式 1：在 serve_realman_policy.py 中指定（推荐）**

适用场景：单一任务的长时间测试，多个 Client 共享同一任务

```bash
python ../realman_openpi_training/serve_realman_policy.py \
    --config=pi0_realman_inference \
    --checkpoint=/path/to/checkpoint \
    --prompt="pick up the yellow banana and put it in the white plate" \  # ← 在这里指定
    --port=8000
```

**方式 2：在 realman_client 中指定（动态切换）**

适用场景：需要频繁切换任务，多个 Client 执行不同任务

```bash
python -m lerobot.extensions.unified_deploy.client.realman_client \
    --task="pick up the red cube" \  # ← 在这里指定
    --robot_ip="192.168.1.18" \
    --server_address=127.0.0.1:50051 \
    ...
```

**方式 3：使用训练时的 default_prompt（自动）**

如果在 `realman_config.py` 中设置了 `default_prompt`，启动时不需要指定 `--prompt`：

```python
# realman_config.py
data=LeRobotRealmanDataConfig(
    default_prompt="pick up the yellow banana and put it in the white plate",
    ...
)
```

```bash
# 启动时不需要 --prompt，会自动使用训练配置中的 prompt
python ../realman_openpi_training/serve_realman_policy.py \
    --config=pi0_realman_pytorch \
    --checkpoint=/path/to/checkpoint \
    --port=8000
```

### 7.3 验证当前使用的 prompt

**查看 OpenPI 服务日志**：
```
策略配置:
  default_prompt: pick up the yellow banana and put it in the white plate
  use_delta_actions: True
```

**查看 PolicyServer 日志**：
```
[OpenPIAdapter] 服务端元数据:
  default_prompt: pick up the yellow banana and put it in the white plate
```

**查看 Client 日志**：
```
[Client] 已从 Server 获取配置:
  任务: pick up the yellow banana and put it in the white plate
```

### 7.4 注意事项

1. **Prompt 风格一致性**：推理时的 prompt 应与训练时的风格一致
2. **Prompt 长度**：建议 < 20 个单词，简洁明确
3. **语言一致性**：使用训练时的语言（英文/中文），不要混用
4. **传统策略兼容性**：ACT/Diffusion 等传统策略不需要 task 参数，框架会自动处理

---

