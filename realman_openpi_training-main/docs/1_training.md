# Realman π₀/π₀.5 训练指南

本文档介绍如何使用 OpenPI 框架在 Realman RM75-6FB 机械臂上训练 π₀/π₀.5 模型。

> **文档版本**: v1.3  
> **最后更新**: 2026-01-21  
> **适用框架**: OpenPI + RoboCOIN unified_deploy

---

## 📚 文档导航

- **[← 返回索引](./README.md)**
- **[下一步：部署推理 →](./2_deployment.md)**
- **[常见问题](./3_faq.md)**

---

## 快速开始

如果你已经熟悉流程，以下是完整的命令序列：

```bash
# 1. 转换数据集 action 字段
python realman_openpi_training/convert_action_to_next_state.py \
    --input-path /path/to/original/dataset \
    --output-path /path/to/converted/dataset \
    --num-workers 8

# 2. 重新计算 episodes_stats（可选但推荐）
conda activate robocoin
python realman_openpi_training/recompute_episodes_stats.py \
    --dataset-path /path/to/converted/dataset \
    --original-dataset-path /path/to/original/dataset \
    --num-workers 8

# 3. 计算 OpenPI 归一化统计量
cd openpi
uv run ../realman_openpi_training/compute_norm_stats.py \
    --config-name pi0_realman_pytorch \
    --dataset-path /path/to/converted/dataset

# 4. 开始训练
uv run ../realman_openpi_training/train.py pi0_realman_pytorch --exp_name=my_exp
```

---

## 术语表

| 术语 | 含义 |
|------|------|
| **cam0_rgb** | 腕部相机 (wrist camera)，安装在机械臂末端 |
| **cam1_rgb** | 基座相机 (base camera)，固定在工作台上 |
| **base_0_rgb** | OpenPI 模型的第一个相机槽位（通常用于基座视角） |
| **left_wrist_0_rgb** | OpenPI 模型的第二个相机槽位（通常用于腕部视角） |
| **delta action** | 相对动作，表示当前状态与目标状态的差值 |
| **absolute action** | 绝对动作，表示目标状态的绝对值 |
| **action_horizon** | 动作序列长度（每次推理预测多少步动作） |
| **action_dim** | 动作维度（模型内部固定为 32，实际使用 8 维） |

---

## 相机命名约定

在 Realman 数据集和整个训练/部署流程中，相机命名遵循以下约定：

```
cam0_rgb: 腕部相机 (wrist camera)
    - 安装位置：机械臂末端/夹爪附近
    - 视角特点：近距离观察操作对象
    - 映射到 OpenPI：left_wrist_0_rgb

cam1_rgb: 基座相机 (base camera)
    - 安装位置：工作台上的固定位置
    - 视角特点：全局视角，观察整个工作空间
    - 映射到 OpenPI：base_0_rgb

cam2_rgb+: 其他视角的基座相机（如有）
    - 额外的固定视角相机
    - 映射到 OpenPI：right_wrist_0_rgb 或其他
```

**重要**：训练时和推理时的映射必须保持一致！

---

## 1. 环境准备

### 1.1 安装 OpenPI

```bash
cd openpi
uv sync
```

### 1.2 训练脚本说明

**重要**：为了避免修改 OpenPI 官方源码，我们使用独立的训练模块 `realman_openpi_training/`。

```
workspace/
├── openpi/                      # 官方仓库（不修改）
└── realman_openpi_training/     # 独立训练模块
    ├── train.py                 # 训练入口
    ├── realman_config.py        # Realman 配置
    ├── compute_norm_stats.py    # 计算归一化统计量
    ├── convert_action_to_next_state.py  # 数据集转换
    ├── recompute_episodes_stats.py      # 重新计算统计量
    └── README.md
```

训练命令：
```bash
cd openpi
uv run ../realman_openpi_training/train.py <config_name> --exp_name=<exp_name>
```

### 1.3 离线训练准备（无网络环境）

如果你的训练服务器没有网络访问，需要提前下载以下文件：

#### 1.3.1 必需文件

| 文件 | 用途 | 下载路径 |
|------|------|----------|
| PaliGemma Tokenizer | 文本分词器 | `gs://big_vision/paligemma_tokenizer.model` |
| π₀ Base 权重 | π₀ 基础模型 | `gs://openpi-assets/checkpoints/pi0_base/params` |
| π₀.5 Base 权重 | π₀.5 基础模型 | `gs://openpi-assets/checkpoints/pi05_base/params` |
| π₀.5 DROID 权重 | DROID 预训练（推荐微调起点） | `gs://openpi-assets/checkpoints/pi05_droid/params` + `/assets` |

#### 1.3.2 下载方法

**方法一：使用 OpenPI 下载工具（推荐）**

```bash
cd openpi
source .venv/bin/activate

# 下载 Tokenizer
python -c "
from openpi.shared.download import maybe_download
maybe_download('gs://big_vision/paligemma_tokenizer.model')
"

# 下载 π₀ base 权重
python -c "
from openpi.shared.download import maybe_download
maybe_download('gs://openpi-assets/checkpoints/pi0_base/params')
"

# 下载 π₀.5 base 权重
python -c "
from openpi.shared.download import maybe_download
maybe_download('gs://openpi-assets/checkpoints/pi05_base/params')
"

# 下载 π₀.5 DROID 权重（推荐用于微调）
python -c "
from openpi.shared.download import maybe_download
maybe_download('gs://openpi-assets/checkpoints/pi05_droid/params')
maybe_download('gs://openpi-assets/checkpoints/pi05_droid/assets')
"
```

文件将下载到 `~/.cache/openpi/` 目录。

**方法二：使用 gsutil 直接下载**

```bash
# 安装 gsutil
pip install gsutil

# 下载文件
gsutil -m cp -r gs://big_vision/paligemma_tokenizer.model ~/.cache/openpi/big_vision/
gsutil -m cp -r gs://openpi-assets/checkpoints/pi0_base ~/.cache/openpi/openpi-assets/checkpoints/
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base ~/.cache/openpi/openpi-assets/checkpoints/
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_droid ~/.cache/openpi/openpi-assets/checkpoints/
```

#### 1.3.3 转换为 PyTorch 格式

OpenPI 官方权重是 JAX 格式，需要转换为 PyTorch 格式才能用于 PyTorch 训练：

```bash
cd openpi
source .venv/bin/activate

# 转换 π₀.5 DROID 权重（推荐）
python examples/convert_jax_model_to_pytorch.py \
    --config=pi05_droid \
    --checkpoint_dir=~/.cache/openpi/openpi-assets/checkpoints/pi05_droid/params \
    --output_dir=data/openpi-assets/checkpoints/pi05_droid_pytorch

# 转换 π₀ base 权重
python examples/convert_jax_model_to_pytorch.py \
    --config=pi0_base \
    --checkpoint_dir=~/.cache/openpi/openpi-assets/checkpoints/pi0_base/params \
    --output_dir=data/openpi-assets/checkpoints/pi0_base_pytorch
```

转换后的 PyTorch 权重将保存在 `data/openpi-assets/checkpoints/` 目录。

#### 1.3.4 离线环境部署

将以下文件打包传输到离线服务器：

```
~/.cache/openpi/
├── big_vision/
│   └── paligemma_tokenizer.model
└── openpi-assets/
    └── checkpoints/
        ├── pi0_base/
        ├── pi05_base/
        └── pi05_droid/

openpi/data/openpi-assets/checkpoints/
├── pi0_base_pytorch/
│   └── model.safetensors
└── pi05_droid_pytorch/
    └── model.safetensors
```

### 1.4 数据集格式要求

数据集需要符合 LeRobot v2.1 格式：

```
dataset/
├── meta/
│   ├── info.json           # 数据集元信息
│   ├── episodes.jsonl      # Episode 索引
│   ├── episodes_stats.jsonl # Episode 统计量
│   └── tasks.jsonl         # 任务描述
├── data/
│   └── chunk-000/          # Parquet 数据文件
└── videos/
    └── chunk-000/          # MP4 视频文件
```

**数据特征定义**（`info.json`）：

```json
{
    "features": {
        "observation.state": {
            "dtype": "float32",
            "shape": [14],
            "names": [
                "joint_1_rad", "joint_2_rad", "joint_3_rad", "joint_4_rad",
                "joint_5_rad", "joint_6_rad", "joint_7_rad", "gripper_open",
                "eef_pos_x_m", "eef_pos_y_m", "eef_pos_z_m",
                "eef_rot_euler_x_rad", "eef_rot_euler_y_rad", "eef_rot_euler_z_rad"
            ]
        },
        "action": {
            "dtype": "float32",
            "shape": [8],
            "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"]
        },
        "observation.images.cam0_rgb": {
            "dtype": "video",
            "shape": [480, 640, 3]
        }
    }
}
```

---

## 2. 数据预处理（关键步骤）

### 2.1 数据集 Action 转换（遥操数据集必须）

**重要**：如果你的数据集是通过遥操作采集的，需要先进行 action 字段转换。

#### 2.1.1 为什么需要转换？

遥操数据集中的 `action` 字段通常是当前帧发送给机械臂的目标指令（command），而不是 OpenPI 期望的下一帧状态：

| 数据类型 | action 含义 | DeltaActions 计算结果 |
|----------|-------------|----------------------|
| 原始遥操数据 | `action_t = command_t` | 跟随误差（约 0.01-0.02 rad） |
| 转换后数据 | `action_t[:7] = state_{t+1}[:7]` | 真正的动作（状态变化量） |

OpenPI 使用 `delta_mask = make_bool_mask(7, -1)`：前 7 维（关节）用 delta 模式，第 8 维（夹爪）用 absolute 模式。

- 原始数据：计算出的是"跟随误差"，不是真正的动作意图
- 转换后：计算出的是 `state_{t+1} - state_t`，即真正的状态变化量

#### 2.1.2 执行转换

```bash
# 转换数据集（不会覆盖原始数据）
python realman_openpi_training/convert_action_to_next_state.py \
    --input-path /path/to/original/dataset \
    --output-path /path/to/converted/dataset \
    --num-workers 8
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--input-path` | 原始数据集路径（LeRobot v2.1 格式） |
| `--output-path` | 输出数据集路径（必须不存在或为空） |
| `--num-workers` | 并行处理的 worker 数量（默认 4） |
| `--joint-dim` | 关节维度（默认 7），只转换前 N 维，夹爪保持原样 |

**转换逻辑**：
- 前 7 维（关节）：`action_t[:7] = state_{t+1}[:7]`
- 第 8 维（夹爪）：保持原样（absolute 模式，不转换）
- 最后一帧：`action_t[:7] = state_t[:7]`（无下一帧，因此 delta = 0）

**输出说明**：
- 视频文件通过符号链接复用，不会复制（节省空间）
- 只修改 parquet 文件中的 action 字段
- 转换完成后会显示原始/新 delta 的统计信息

#### 2.1.3 验证转换结果

转换脚本会输出统计信息：

```
[INFO] 原始 delta (action - state) 最大值:
       平均: 0.015234 rad
       最大: 0.023456 rad
[INFO] 新 delta (next_state - state) 最大值:
       平均: 0.001234 rad
       最大: 0.002345 rad
```

新 delta 应该明显小于原始 delta（约 10 倍差距）。

### 2.2 重新计算 episodes_stats（可选但推荐）

由于 action 字段已更改，需要重新计算统计量：

```bash
# 在 RoboCOIN 环境中运行
conda activate robocoin

python realman_openpi_training/recompute_episodes_stats.py \
    --dataset-path /path/to/converted/dataset \
    --original-dataset-path /path/to/original/dataset \
    --num-workers 8
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--dataset-path` | 转换后的数据集路径 |
| `--original-dataset-path` | 原始数据集路径（用于复制非 action 统计量） |
| `--num-workers` | 并行 worker 数量（默认 8） |

此脚本会：
- 从原始数据集复制其他字段的统计量
- 只重新计算 action 字段的统计量
- 覆盖原有的 `episodes_stats.jsonl` 文件

### 2.3 计算 OpenPI 归一化统计量（norm_stats）

这是训练前必须的步骤：

```bash
cd openpi

uv run ../realman_openpi_training/compute_norm_stats.py \
    --config-name pi0_realman_pytorch \
    --dataset-path /path/to/converted/dataset
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--config-name` | 训练配置名称（必填） |
| `--dataset-path` | 数据集路径（必填） |
| `--output-path` | 自定义输出路径（可选，推荐使用以避免路径不匹配） |
| `--num-workers` | 并行 worker 数量（默认 8） |
| `--state-dim` | state 维度（默认 8） |
| `--action-dim` | action 维度（默认 8） |

**输出文件**：
```
openpi/assets/{config_name}/{repo_id}/norm_stats.json
```

例如：`openpi/assets/pi0_realman_pytorch/local/realman_teleop_0111_right_plate_50/norm_stats.json`

> **⚠️ 重要警告：repo_id 路径匹配问题**
> 
> 输出路径中的 `{repo_id}` 来自配置文件 `realman_config.py` 中的 `repo_id` 字段。
> 如果你修改了数据集但没有同步修改 `repo_id`，会导致：
> - norm_stats 保存到了旧路径
> - 训练时在新路径下找不到 norm_stats.json
> 
> **解决方案**：
> 1. 确保 `realman_config.py` 中的 `repo_id` 与你的数据集名称一致
> 2. 或者使用 `--output-path` 参数显式指定输出路径
> 
> 详见 [Q9: 训练时报错 "norm_stats.json not found"](#q9-训练时报错-norm_statsjson-not-found但我已经计算过了)

**注意事项**：
1. **必须在训练前计算**：如果缺少 norm_stats.json，训练会报错
2. **深度图自动过滤**：如果数据集包含深度图，会自动过滤，不会导致错误
3. **更换数据集需重新计算**：如果更换数据集或修改 transforms，需要重新计算
4. **检查 repo_id 一致性**：确保配置文件中的 `repo_id` 与实际数据集路径匹配

---

## 3. 训练配置

### 3.1 可用配置

| 配置名称 | 模型 | 说明 | 显存需求 |
|----------|------|------|----------|
| `pi0_realman_pytorch` | π₀ | 使用本地 PyTorch 权重（推荐） | ~24GB |
| `pi05_realman_pytorch` | π₀.5 | 使用本地 PyTorch 权重 | ~24GB |
| `pi0_realman` | π₀ | 全量微调，使用 GCS 权重（需要网络） | ~40GB |
| `pi0_realman_lora` | π₀ | LoRA 微调，低显存 | ~16GB |
| `pi05_realman` | π₀.5 | 全量微调 | ~48GB |
| `pi0_fast_realman` | π₀-FAST | 快速推理版本 | ~40GB |
| `pi0_realman_base_only_pytorch` | π₀ | 只使用基座相机 | ~24GB |
| `pi0_realman_wrist_only_pytorch` | π₀ | 只使用腕部相机 | ~24GB |

### 3.2 修改数据集路径

编辑 `realman_openpi_training/realman_config.py`，修改 `DEFAULT_DATASET_PATH` 和相关配置：

```python
# 修改默认数据集路径
DEFAULT_DATASET_PATH = "/path/to/your/converted/dataset"

# 在配置中修改
data=LeRobotRealmanDataConfig(
    # 数据集标识符
    repo_id="local/your_dataset_name",
    # 本地数据集路径（绝对路径）
    local_root=DEFAULT_DATASET_PATH,
    use_delta_joint_actions=True,
    default_prompt="pick and place the object",
    # 图像观测 key 列表
    image_keys=("observation.images.cam1_rgb", "observation.images.cam0_rgb"),
),
```

### 3.3 图像输入配置（image_keys）

通过 `image_keys` 参数指定需要使用的图像观测：

```python
image_keys=("observation.images.cam1_rgb", "observation.images.cam0_rgb")
```

**映射规则**：图像按照 `image_keys` 中的顺序映射到 OpenPI 模型的相机槽位：

| 顺序 | OpenPI 槽位 | 说明 |
|------|-------------|------|
| 第 1 个 | `base_0_rgb` | 基座相机 |
| 第 2 个 | `left_wrist_0_rgb` | 左腕部相机 |
| 第 3 个 | `right_wrist_0_rgb` | 右腕部相机 |

**配置示例**：

```python
# 双相机：基座 + 腕部（推荐）
image_keys=(
    "observation.images.cam1_rgb",  # → base_0_rgb
    "observation.images.cam0_rgb",  # → left_wrist_0_rgb
)

# 单相机：只使用基座相机
image_keys=("observation.images.cam1_rgb",)  # → base_0_rgb

# 单相机：只使用腕部相机
image_keys=("observation.images.cam0_rgb",)  # → base_0_rgb（第一个总是映射到 base）
```

**注意**：深度图（如 `cam0_depth`, `cam1_depth`）会被自动过滤，不会导致统计量验证错误。

### 3.4 任务指令配置（Prompt）

OpenPI 支持两种 prompt 来源：

| 来源 | 配置方式 | 适用场景 |
|------|----------|----------|
| `default_prompt` | 配置文件中固定字符串 | 单任务数据集 |
| 数据集 task 字段 | `prompt_from_task=True` | 多任务数据集 |

**单任务数据集配置**：

```python
data=LeRobotRealmanDataConfig(
    repo_id="local/realman_teleop_0111_right_plate_50",
    use_delta_joint_actions=True,
    default_prompt="pick up the yellow banana and put it in the white plate",
    prompt_from_task=False,  # 默认值，可省略
),
```

**多任务数据集配置**：

```python
data=LeRobotRealmanDataConfig(
    repo_id="local/realman_multi_task_dataset",
    use_delta_joint_actions=True,
    prompt_from_task=True,  # 从数据集 task 字段读取
    default_prompt=None,
),
```

### 3.5 LoRA 微调说明

LoRA（Low-Rank Adaptation）是一种参数高效微调技术：
- **优势**：可训练参数从 ~3B 降到 ~10M，显存需求从 ~40GB 降到 ~16GB
- **适用场景**：显存有限（如 3090/4090）、快速实验、小数据集微调

```python
# realman_config.py 中的 LoRA 配置
model=pi0_config.Pi0Config(
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora"
),
```

---

## 4. 开始训练

### 4.1 训练环境说明

**重要**：训练和推理使用不同的 Python 环境：

| 阶段 | 环境 | 激活方式 |
|------|------|----------|
| 训练 | OpenPI 环境 | `cd openpi && source .venv/bin/activate` 或 `uv run` |
| 推理 | RoboCOIN 环境 | `conda activate robocoin` |

### 4.2 π₀ 训练（使用 PyTorch 权重）

```bash
cd openpi

# 单卡训练
uv run ../realman_openpi_training/train.py pi0_realman_pytorch \
    --exp_name=realman_pi0_v1 \
    --num_train_steps=30000 \
    --batch_size=8

# 多卡训练（DDP）
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    ../realman_openpi_training/train.py pi0_realman_pytorch \
    --exp_name=realman_pi0_v1 \
    --batch_size=16
```

### 4.3 π₀.5 训练（使用 PyTorch 权重）

```bash
cd openpi

uv run ../realman_openpi_training/train.py pi05_realman_pytorch \
    --exp_name=realman_pi05_v1 \
    --num_train_steps=30000 \
    --batch_size=8
```

### 4.4 LoRA 微调（低显存）

```bash
uv run ../realman_openpi_training/train.py pi0_realman_lora --exp_name=realman_pi0_lora_v1
```

### 4.5 常用训练参数

```bash
uv run ../realman_openpi_training/train.py pi0_realman_pytorch \
    --exp_name=my_experiment \
    --num_train_steps=50000 \
    --batch_size=8 \
    --save_interval=2000 \
    --log_interval=100 \
    --wandb_enabled=True
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--exp_name` | 必填 | 实验名称，用于 checkpoint 目录 |
| `--num_train_steps` | 30000 | 总训练步数 |
| `--batch_size` | 8 | 全局 batch size（多 GPU 会均分） |
| `--save_interval` | 1000 | 每多少步保存 checkpoint |
| `--log_interval` | 100 | 每多少步记录日志 |
| `--keep_period` | 5000 | 保留 step % keep_period == 0 的 checkpoint |
| `--wandb_enabled` | True | 是否启用 WandB 日志 |
| `--resume` | False | 是否从上次 checkpoint 恢复训练 |
| `--overwrite` | False | 是否覆盖已有 checkpoint 目录 |

### 4.6 多 GPU 训练

**batch_size 是全局总数**，会自动均分到每个 GPU：

```bash
# 2 GPU 训练，每个 GPU 实际 batch_size = 16 / 2 = 8
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    ../realman_openpi_training/train.py pi0_realman_pytorch \
    --exp_name=my_exp --batch_size=16
```

### 4.7 Checkpoint 保存

Checkpoint 保存在：
```
openpi/checkpoints/{config_name}/{exp_name}/{step}/
├── model.safetensors          # PyTorch 模型权重
├── optimizer.pt               # 优化器状态
├── metadata.pt                # 训练元数据
└── assets/
    └── <asset_id>/
        └── norm_stats.json    # 归一化统计量
```

---

