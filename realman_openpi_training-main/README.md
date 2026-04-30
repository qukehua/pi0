# Realman OpenPI 训练模块

这是一个独立于 OpenPI 官方源码的训练模块，用于在 Realman RM75-6FB 机械臂上训练 π₀/π₀.5 模型。
详细说明可见docs。

## 特点

- **PyTorch 训练**：使用 PyTorch 进行训练，显存占用更低，支持梯度检查点
- **零侵入**：完全不修改 OpenPI 官方源码
- **可移植**：可以在任何服务器上直接 clone 官方 openpi 仓库使用
- **易维护**：只需维护这个独立目录中的配置文件
- **多卡支持**：支持单卡和多卡（DDP）训练

## 目录结构

```
realman_openpi_training/
├── train.py                          # PyTorch 训练入口脚本
├── serve_realman_policy.py           # 推理服务启动脚本（双进程部署）
├── realman_config.py                 # Realman 机械臂配置
├── compute_norm_stats.py             # 计算归一化统计量
├── convert_action_to_next_state.py   # 数据集 action 转换脚本
├── docs/
│   └── realman_pi_training.md        # 详细训练文档
└── README.md                         # 本文档
```

## 数据集准备

### 1. Action 字段转换（重要！）

遥操数据集中的 `action` 字段通常是当前帧发送给机械臂的目标指令，而不是 OpenPI 期望的下一帧状态。需要先进行转换：

```bash
# 转换数据集
python realman_openpi_training/convert_action_to_next_state.py \
    --input-path /path/to/original/dataset \
    --output-path /path/to/converted/dataset \
    --num-workers 8
```

**转换逻辑**：
- 前 7 维（关节）：`action_t[:7] = state_{t+1}[:7]`
- 第 8 维（夹爪）：保持原样（absolute 模式，不转换）
- 最后一帧：`action_t[:7] = state_t[:7]`（无下一帧）

**为什么需要转换**：
- OpenPI 使用 `delta_mask = make_bool_mask(7, -1)`，前 7 维用 delta 模式，夹爪用 absolute 模式
- DeltaActions 计算：`delta_action_t = action_t - state_t`
- 原始遥操数据计算出的是"跟随误差"（约 0.01-0.02 rad）
- 转换后计算的是真正的状态变化量

### 2. 计算归一化统计量

```bash
cd openpi
uv run ../realman_openpi_training/compute_norm_stats.py \
    --config-name pi0_realman_pytorch \
    --dataset-path /path/to/converted/dataset
```

## 使用方法

### 1. 准备环境

```bash
# 克隆官方 openpi 仓库（如果还没有）
git clone https://github.com/Physical-Intelligence/openpi.git

# 安装依赖
cd openpi
uv sync
```

### 2. 运行训练

```bash
# 在 openpi 目录下运行
cd openpi

# 单卡训练
uv run ../realman_openpi_training/train.py pi0_realman_pytorch --exp_name=my_exp

# 多卡训练（DDP）
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    ../realman_openpi_training/train.py pi0_realman_pytorch --exp_name=my_exp

# 从检查点恢复训练
uv run ../realman_openpi_training/train.py pi0_realman_pytorch --exp_name=my_exp --resume
```

### 3. 可用配置

| 配置名称 | 模型 | 说明 |
|----------|------|------|
| `pi0_realman_pytorch` | π₀ | 使用本地 PyTorch 权重（推荐） |
| `pi05_realman_pytorch` | π₀.5 | 使用本地 PyTorch 权重 |
| `pi0_realman` | π₀ | 全量微调，使用 GCS 权重（需要网络） |
| `pi0_realman_lora` | π₀ | LoRA 微调，低显存 |
| `pi05_realman` | π₀.5 | 全量微调 |
| `pi0_fast_realman` | π₀-FAST | 快速训练版本 |
| `pi0_realman_base_only_pytorch` | π₀ | 只使用基座相机 |
| `pi0_realman_wrist_only_pytorch` | π₀ | 只使用腕部相机 |

### 4. 图像输入配置

通过 `image_keys` 参数指定需要使用的图像观测：

```python
# 双相机（默认）
image_keys=(
    "observation.images.cam1_rgb",  # → base_0_rgb
    "observation.images.cam0_rgb",  # → left_wrist_0_rgb
)

# 单相机
image_keys=("observation.images.cam1_rgb",)  # 只使用基座相机
```

图像按照 `image_keys` 中的顺序映射到 OpenPI 模型的相机槽位：
- 第 1 个 → `base_0_rgb`
- 第 2 个 → `left_wrist_0_rgb`
- 第 3 个 → `right_wrist_0_rgb`

**注意**：深度图会被自动过滤，不会导致统计量验证错误。

## 显存优化

PyTorch 训练脚本默认启用以下优化：

1. **梯度检查点**：自动启用，显著降低显存占用
2. **混合精度训练**：使用 bfloat16 精度
3. **梯度累积**：可通过调整 batch_size 实现

如果仍然遇到显存不足，可以尝试：

```bash
# 减小 batch_size（在 realman_config.py 中修改）
batch_size=16  # 默认 32

# 或设置环境变量
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run ../realman_openpi_training/train.py ...
```

## 云服务器部署

在云服务器上使用时，只需：

1. Clone 官方 openpi 仓库
2. 将 `realman_openpi_training/` 目录复制到服务器
3. 运行训练命令

```bash
# 服务器上的目录结构
workspace/
├── openpi/                    # 官方仓库（不修改）
└── realman_openpi_training/   # 你的配置（复制过来）

# 运行训练
cd workspace/openpi
uv sync
uv run ../realman_openpi_training/train.py pi0_realman_pytorch --exp_name=my_exp
```

## 自定义配置

编辑 `realman_config.py` 修改：

- `repo_id`: 数据集路径
- `local_root`: 本地数据集绝对路径
- `default_prompt`: 任务描述
- `num_train_steps`: 训练步数
- `batch_size`: 批次大小
- `lr_schedule`: 学习率调度
- 其他超参数

## 检查点格式

训练产生的检查点目录结构：

```
checkpoints/pi0_realman_pytorch/<exp_name>/<step>/
├── model.safetensors          # PyTorch 模型权重
├── optimizer.pt               # 优化器状态
├── metadata.pt                # 训练元数据
└── assets/
    └── <asset_id>/
        └── norm_stats.json    # 归一化统计量
```

此格式与 `RoboCOIN/unified_deploy` 框架完全兼容，可直接用于真机部署。

## 工作原理

训练脚本通过动态注册机制将自定义配置注入到 OpenPI 的配置系统中：

```python
def register_realman_configs():
    from openpi.training import config as _config
    from realman_config import get_realman_configs
    
    for cfg in get_realman_configs():
        _config._CONFIGS_DICT[cfg.name] = cfg
```

同时 patch 数据加载器以支持：
- 本地数据集加载（无需从 HuggingFace Hub 下载）
- 自动过滤深度图统计量（避免验证错误）

## 部署推理

由于 OpenPI 和 RoboCOIN 使用不同的 Python 环境，推荐使用**三终端部署架构**：

| 终端 | 环境 | 作用 |
|------|------|------|
| 终端 1 | openpi (.venv) | `serve_realman_policy.py` - π₀ 模型推理 |
| 终端 2 | robocoin (conda) | `policy_server.py` - gRPC 服务 |
| 终端 3 | robocoin (conda) | `realman_client.py` - 机器人控制 |

**详细部署说明请参阅 [`docs/realman_pi_training.md`](docs/realman_pi_training.md#5-部署推理)**，包含：
- 完整的三终端启动命令
- 架构图解
- 参数说明
- 与 ACT/Diffusion 模型的对比
