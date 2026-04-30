# openpi 项目详细解析文档

> 本文档对 `/share/0xyj/model3_openpi0.5/openpi-main` 项目进行全面、系统的中文解析。

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体目录结构](#2-整体目录结构)
3. [核心模型详解](#3-核心模型详解)
4. [数据流水线](#4-数据流水线)
5. [训练系统](#5-训练系统)
6. [推理与部署](#6-推理与部署)
7. [支持的机器人平台](#7-支持的机器人平台)
8. [PyTorch 支持](#8-pytorch-支持)
9. [配置系统](#9-配置系统)
10. [依赖与环境](#10-依赖与环境)
11. [关键代码路径速查](#11-关键代码路径速查)

---

## 1. 项目概述

`openpi` 是由 **Physical Intelligence (PI)** 团队开源的机器人视觉-语言-动作（VLA）模型库，提供三种系列模型：

| 模型 | 类型 | 特点 |
|------|------|------|
| **pi0** | 流匹配 VLA | 连续动作生成，推理速度快 |
| **pi0-FAST** | 自回归 VLA | 基于 FAST 动作分词器，语言遵循能力强 |
| **pi0.5** | 升级版 pi0 | 更好的开放世界泛化，采用知识隔离训练 |

### 1.1 核心能力

- 基于 10k+ 小时机器人数据预训练的基础权重，可直接下载推理
- 支持全量微调和 LoRA 低内存微调
- 适配 ALOHA、DROID（Franka）、LIBERO、UR5 等主流平台
- 支持远程推理：模型运行于服务器，通过 WebSocket 将动作流式传输给机器人
- 同时支持 JAX（原生）和 PyTorch 两种计算后端

### 1.2 硬件要求

| 模式 | 显存需求 | 示例 GPU |
|------|----------|----------|
| 推理 | > 8 GB | RTX 4090 |
| LoRA 微调 | > 22.5 GB | RTX 4090 |
| 全量微调 | > 70 GB | A100 80GB / H100 |

### 1.3 预训练检查点

所有检查点托管于 `gs://openpi-assets`，首次使用时自动下载缓存至 `~/.cache/openpi`：

| 名称 | 用途 |
|------|------|
| `pi0_base` | pi0 基础模型（供微调）|
| `pi0_fast_base` | pi0-FAST 基础模型（供微调）|
| `pi05_base` | pi0.5 基础模型（供微调）|
| `pi0_fast_droid` | pi0-FAST 在 DROID 上微调（零样本桌面操作）|
| `pi05_droid` | pi0.5 在 DROID 上微调（知识隔离）|
| `pi05_libero` | pi0.5 在 LIBERO 上微调（SOTA）|

---

## 2. 整体目录结构

```
openpi-main/
├── src/openpi/
│   ├── models/              # 模型定义（JAX）
│   │   ├── model.py         # 基类：Observation、BaseModel、BaseModelConfig
│   │   ├── pi0.py           # pi0/pi0.5 实现（流匹配）
│   │   ├── pi0_config.py    # pi0/pi0.5 配置
│   │   ├── pi0_fast.py      # pi0-FAST 实现（自回归）
│   │   ├── gemma.py         # Gemma LLM 骨干
│   │   ├── siglip.py        # SigLIP 视觉编码器
│   │   ├── tokenizer.py     # PaliGemma/FAST 分词器
│   │   ├── lora.py          # LoRA 低秩适应
│   │   └── models_pytorch/  # PyTorch 版本模型
│   ├── policies/            # 策略层（平台适配）
│   │   ├── policy.py        # Policy 推理封装
│   │   ├── policy_config.py # 策略工厂函数
│   │   ├── aloha_policy.py  # ALOHA 数据适配
│   │   ├── droid_policy.py  # DROID 数据适配
│   │   └── libero_policy.py # LIBERO 数据适配
│   ├── training/            # 训练系统
│   │   ├── config.py        # 所有训练配置
│   │   ├── data_loader.py   # 数据加载器
│   │   ├── checkpoints.py   # 检查点管理（Orbax）
│   │   ├── optimizer.py     # 优化器配置
│   │   ├── sharding.py      # JAX FSDP 分片
│   │   └── weight_loaders.py# 预训练权重加载
│   ├── serving/
│   │   └── websocket_policy_server.py  # WebSocket 服务器
│   ├── shared/              # 共享工具
│   │   ├── normalize.py     # 归一化统计
│   │   ├── download.py      # GCS 自动下载
│   │   └── image_tools.py   # 图像处理工具
│   └── transforms.py        # 数据变换系统
├── scripts/
│   ├── train.py             # JAX 训练入口
│   ├── train_pytorch.py     # PyTorch 训练入口
│   ├── serve_policy.py      # 启动策略服务器
│   └── compute_norm_stats.py# 计算归一化统计
├── packages/openpi-client/  # 独立 WebSocket 客户端包
├── examples/                # 各平台使用示例
│   ├── aloha_real/          # ALOHA 真实机器人
│   ├── aloha_sim/           # ALOHA 仿真
│   ├── droid/               # DROID 机器人
│   ├── libero/              # LIBERO 仿真基准
│   ├── simple_client/       # 无机器人测试
│   └── inference.ipynb      # 推理演示 Notebook
├── docs/                    # 文档
├── dlimp_pkg/               # RLDS 数据加载包
├── lerobot_pkg/             # LeRobot 数据集包
└── pyproject.toml           # 依赖配置
```

---

## 3. 核心模型详解

### 3.1 整体架构：PaliGemma + 动作专家

所有 pi0 系列模型采用双专家 Transformer 架构：

```
图像(224x224) + 语言指令 + 机器人状态
       |
       +-- [SigLIP So400m/14]  --> 图像 token 序列
       +-- [PaliGemma tokenizer]--> 文本 token 序列
       +-- [Linear 投影]        --> 动作 token 序列
       |
  [前缀 Prefix]：图像 + 语言 token，双向注意力（非因果）
       |
  [后缀 Suffix]：动作 token，只能看到前缀，彼此因果掩码
       |
  [Gemma 2B 视觉语言专家] + [Gemma 300M 动作专家]
  两专家共享注意力计算
       |
  流匹配输出 -> 动作 chunk
```

**KV Cache 加速推理**：前缀只需计算一次并缓存，推理时每个欧拉步只对后缀（动作 token）做前向传播，大幅降低延迟。

### 3.2 pi0 与 pi0.5 的区别

| 对比项 | pi0 | pi0.5 |
|--------|-----|-------|
| 状态输入方式 | 独立状态 token 作为后缀首个 token | 状态离散化后编码进语言 token（前缀）|
| 时间步注入 | 时间步与动作 token 拼接后过 MLP | 时间步通过 adaRMSNorm 注入动作专家 |
| `pi05` 标志 | False | True |
| max_token_len 默认值 | 48 | 200 |
| 典型 action_horizon | 50（ALOHA）| 15（DROID）/ 10（LIBERO）|

### 3.3 模型配置（Pi0Config）

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(BaseModelConfig):
    dtype: str = "bfloat16"                  # 计算精度
    paligemma_variant: str = "gemma_2b"      # 视觉语言专家变体
    action_expert_variant: str = "gemma_300m"# 动作专家变体
    action_dim: int = 32                     # 动作空间维度
    action_horizon: int = 50                 # 动作 chunk 长度
    pi05: bool = False                       # 是否使用 pi0.5 架构
    discrete_state_input: bool = None        # 是否离散化状态输入
```

可用变体：`gemma_2b`（全量微调）、`gemma_2b_lora`（LoRA）、`dummy`（调试）

### 3.4 流匹配（Flow Matching）原理

**训练**（`compute_loss`）：
```
t ~ Beta(1.5, 1) * 0.999 + 0.001   # 偏向噪声较多区域
x_t = t*noise + (1-t)*actions      # 带噪动作
u_t = noise - actions               # 真实速度场
Loss = MSE(predicted_v_t, u_t)
```

**推理**（`sample_actions`，欧拉积分）：
```
1. 前缀一次前向传播，填充 KV Cache
2. 从 x1（纯噪声）循环 num_steps 次（默认 10）：
   v_t = 模型前向（仅后缀，复用 KV Cache）
   x_t += (-1/num_steps) * v_t
3. x0 即为预测的动作 chunk
```

---

## 4. 数据流水线

### 4.1 数据标准格式

```python
{
    "image": {
        "base_0_rgb":        uint8[H, W, 3],  # 第三视角 RGB
        "left_wrist_0_rgb":  uint8[H, W, 3],  # 左腕摄像头
        "right_wrist_0_rgb": uint8[H, W, 3],  # 右腕摄像头（无则零填充）
    },
    "image_mask": {"base_0_rgb": bool, "left_wrist_0_rgb": bool, "right_wrist_0_rgb": bool},
    "state":                 float32[state_dim],
    "tokenized_prompt":      int32[max_token_len],
    "tokenized_prompt_mask": bool[max_token_len],
    "actions":               float32[action_horizon, action_dim],  # 仅训练时
}
```

`Observation.from_dict()` 自动将 uint8 图像转为 [-1,1] float32；PyTorch (C,H,W) 转为 (H,W,C)。

### 4.2 三层 Transform 流水线

```
原始数据
  | [1] repack_transforms（仅训练）：重命名 key 对齐推理格式
  | [2] data_transforms（训练+推理）：平台适配，可选 DeltaActions
  | [3] Normalize：使用 norm_stats 归一化 state/action
  | [4] model_transforms（训练+推理）：InjectDefaultPrompt, ResizeImages(224x224),
  |     TokenizePrompt, PadStatesAndActions
  v 最终模型输入
```

### 4.3 内置 Transform 清单

| Transform | 作用 |
|-----------|------|
| `RepackTransform` | 按映射字典重命名/重组键值 |
| `InjectDefaultPrompt` | 无 prompt 时注入默认语言指令 |
| `Normalize` / `Unnormalize` | z-score 或分位数归一化/反归一化 |
| `ResizeImages` | 图像 resize+pad 至目标分辨率 |
| `DeltaActions` / `AbsoluteActions` | 绝对/增量动作互转 |
| `TokenizePrompt` | 字符串指令转 token 序列 |
| `TokenizeFASTInputs` | pi0-FAST：指令+状态+动作联合分词 |
| `ExtractFASTActions` | pi0-FAST：输出 token 解码为动作 |
| `PadStatesAndActions` | state/action 零填充至 action_dim |
| `PromptFromLeRobotTask` | 从 LeRobot task 字段提取语言指令 |

### 4.4 归一化策略

- **z-score**（pi0）：`x_norm = (x - mean) / (std + 1e-6)`
- **分位数**（pi0.5）：`x_norm = (x - q01) / (q99 - q01 + 1e-6) * 2 - 1`（输出范围 [-1,1]）

预计算命令：`uv run scripts/compute_norm_stats.py --config-name pi05_libero`

---

## 5. 训练系统

### 5.1 训练流程（JAX）

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_exp --overwrite
```

内部流程（`scripts/train.py`）：
```
1. init_logging() + init_wandb()
2. create_data_loader()         # 构建 LeRobot/RLDS 数据加载器
3. init_train_state()           # 随机初始化 + 加载预训练权重 + FSDP 分片
4. 训练循环（num_train_steps 次）：
   - train_step: compute_loss -> grad -> optimizer.update
   - EMA 参数更新
   - 每 log_interval 步记录 loss/grad_norm 到 W&B
   - 每 save_interval 步保存 Orbax 检查点
```

### 5.2 TrainConfig 关键字段

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 32 | 全局 batch size |
| `num_train_steps` | 30000 | 训练总步数 |
| `ema_decay` | 0.99 | EMA 衰减（None 禁用）|
| `freeze_filter` | Nothing | 冻结参数过滤器（LoRA 时使用）|
| `fsdp_devices` | 1 | FSDP 分片设备数 |
| `save_interval` | 1000 | 检查点保存间隔（步）|
| `wandb_enabled` | True | 是否启用 W&B 日志 |
| `pytorch_weight_path` | None | PyTorch 格式基础权重路径 |

### 5.3 LoRA 微调

```python
TrainConfig(
    name="pi0_libero_low_mem_finetune",
    model=Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ),
    freeze_filter=Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter(),
    ema_decay=None,
)
```

---

## 6. 推理与部署

### 6.1 直接代码推理

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
policy = policy_config.create_trained_policy(config, checkpoint_dir)

example = {
    "observation/exterior_image_1_left": ...,  # numpy uint8 [H,W,3]
    "observation/wrist_image_left": ...,
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]  # float32 [action_horizon, action_dim]
```

### 6.2 Policy.infer() 推理流程

```
输入 obs
  -> input_transforms（归一化、resize、tokenize）
  -> Observation.from_dict()
  -> model.sample_actions()（流匹配欧拉积分，10步）
  -> output_transforms（Unnormalize、AbsoluteActions）
  -> {"actions": float32[action_horizon, robot_action_dim]}
```

推理耗时记录在 `policy_timing.infer_ms` 字段。

### 6.3 策略服务器（WebSocket）

```bash
# 使用训练好的检查点
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=checkpoints/pi05_libero/my_experiment/20000

# 使用默认预训练模型
uv run scripts/serve_policy.py --env droid
uv run scripts/serve_policy.py --env aloha_sim
```

服务器默认监听 `0.0.0.0:8000`，可用 `--record` 记录输入输出到磁盘。

### 6.4 远程推理客户端

```python
from openpi_client import websocket_client_policy
policy = websocket_client_policy.WebsocketClientPolicy(host="192.168.1.100", port=8000)
action_chunk = policy.infer(obs)["actions"]
```

`ActionChunkBroker` 用于高频控制循环中平滑消费动作 chunk，避免每步等待网络。

---

## 7. 支持的机器人平台

### 7.1 ALOHA（双臂机器人）

- **适配文件**：`src/openpi/policies/aloha_policy.py`
- **动作空间**：14 维（左右各 6 关节 + 1 夹爪），支持增量动作
- **图像输入**：顶部摄像头（cam_high）+ 可选左右腕摄像头
- **adapt_to_pi**：将标准 ALOHA 关节空间转换为 PI 内部坐标系（标准数据应设为 True）
- **示例**：`examples/aloha_real/`、`examples/aloha_sim/`

### 7.2 DROID（Franka 机器人）

- **适配文件**：`src/openpi/policies/droid_policy.py`
- **动作空间**：8 维（7 关节速度/位置 + 1 夹爪）
- **图像输入**：外部摄像头（exterior_image_1_left）+ 腕部摄像头（wrist_image_left）
- **大规模训练**：支持 RLDS 格式完整 DROID 数据集（约 10k 小时）
- **示例**：`examples/droid/`

### 7.3 LIBERO（仿真基准）

- **适配文件**：`src/openpi/policies/libero_policy.py`
- **动作空间**：7 维（模型输出取前 7 维，其余为零填充）
- **状态空间**：8 维
- **图像输入**：正面摄像头 + 腕部摄像头，右腕用零填充
- **评测**：pi0.5 在 LIBERO 上达到 SOTA，详见 `examples/libero/README.md`

### 7.4 UR5

- 参考 `examples/ur5/README.md`，需自定义数据适配层

---

## 8. PyTorch 支持

### 8.1 功能支持矩阵

| 功能 | JAX | PyTorch |
|------|-----|---------|
| pi0/pi0.5 推理 | 支持 | 支持 |
| pi0-FAST | 支持 | 不支持 |
| 全量微调 | 支持 | 支持 |
| LoRA 微调 | 支持 | 不支持 |
| FSDP 多卡 | 支持 | 不支持 |
| DDP 多卡 | 不支持 | 支持 |
| 多节点训练 | 不支持 | 支持（torchrun）|
| EMA | 支持 | 不支持 |

### 8.2 安装 PyTorch 支持

```bash
uv sync
uv pip show transformers   # 确认版本为 4.53.2
# 应用必要 patch（支持 AdaRMS、精度控制、KV Cache）
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
    .venv/lib/python3.11/site-packages/transformers/
# 撤销：uv cache clean transformers
```

### 8.3 JAX 转 PyTorch 检查点

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax_ckpt \
    --config_name pi05_libero \
    --output_path /path/to/pytorch_ckpt
```

### 8.4 PyTorch 训练

```bash
# 单 GPU
uv run scripts/train_pytorch.py pi05_libero --exp_name my_run

# 多 GPU DDP（单节点）
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py pi05_libero --exp_name my_run

# 恢复训练
uv run scripts/train_pytorch.py pi05_libero --exp_name my_run --resume
```

---

## 9. 配置系统

### 9.1 已定义配置列表

| 配置名 | 模型 | 用途 |
|--------|------|------|
| `pi0_aloha` | pi0 | ALOHA 推理 |
| `pi05_aloha` | pi0.5 | ALOHA 推理 |
| `pi0_droid` | pi0 | DROID 推理 |
| `pi0_fast_droid` | pi0-FAST | DROID 推理 |
| `pi05_droid` | pi0.5 | DROID 推理（知识隔离）|
| `pi0_libero` | pi0 | LIBERO 全量微调 |
| `pi0_libero_low_mem_finetune` | pi0 LoRA | LIBERO LoRA 微调 |
| `pi0_fast_libero` | pi0-FAST | LIBERO 全量微调 |
| `pi05_libero` | pi0.5 | LIBERO 全量微调 |
| `pi0_aloha_sim` | pi0 | ALOHA 仿真微调 |
| `pi05_droid_finetune` | pi0.5 | 自定义 DROID 数据集微调 |
| `debug` | pi0（dummy）| 快速调试（无需 GPU）|

### 9.2 自定义数据集适配步骤

**步骤 1**：将数据转换为 LeRobot 格式（参考 `examples/libero/convert_libero_data_to_lerobot.py`）

**步骤 2**：创建平台适配类（参考 `LiberoInputs` / `LiberoOutputs`）

```python
@dataclasses.dataclass(frozen=True)
class MyRobotInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    def __call__(self, data: dict) -> dict:
        return {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": parse_image(data["observation/image"]),
                "left_wrist_0_rgb": parse_image(data["observation/wrist"]),
                "right_wrist_0_rgb": np.zeros((H, W, 3), dtype=np.uint8),
            },
            "image_mask": {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False},
            "actions": data.get("actions"),
            "prompt": data.get("prompt"),
        }
```

**步骤 3**：在 `config.py` 中添加 `TrainConfig` 并计算归一化统计：

```bash
uv run scripts/compute_norm_stats.py --config-name my_robot_config
```

**步骤 4**：启动训练：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py my_robot_config \
    --exp-name=exp1 --overwrite
```

---

## 10. 依赖与环境

### 10.1 核心依赖

| 依赖包 | 版本 | 用途 |
|--------|------|------|
| `jax[cuda12]` | 0.5.3 | 主计算框架 |
| `flax` | 0.10.2 | JAX 神经网络库（NNX）|
| `torch` | 2.7.1 | PyTorch 后端 |
| `transformers` | 4.53.2 | PyTorch 模型骨干 |
| `orbax-checkpoint` | 0.11.13 | JAX 检查点管理 |
| `lerobot` | 本地路径 | LeRobot 数据集加载 |
| `wandb` | >=0.19.1 | 实验监控 |
| `tyro` | >=0.9.5 | CLI 参数解析 |
| `sentencepiece` | >=0.2.0 | PaliGemma tokenizer |
| `augmax` | >=0.3.4 | 图像增强（训练时）|

### 10.2 安装方式

```bash
# 克隆仓库（含子模块）
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi

# 安装依赖
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 10.3 可选依赖

```bash
# RLDS 数据加载（完整 DROID 数据集训练）
uv sync --group rlds

# 开发工具
uv sync --group dev
```

### 10.4 常见问题排查

| 问题 | 解决方案 |
|------|----------|
| `uv sync` 依赖冲突 | `rm -rf .venv && uv sync` 或 `uv self update` |
| 训练显存不足 | 设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`，或 `--fsdp-devices N` |
| 策略服务器连接失败 | 检查端口 8000 是否开放，确认防火墙设置 |
| 缺少 norm_stats | 先运行 `scripts/compute_norm_stats.py` |
| CUDA 错误 | 尝试卸载系统级 CUDA 库，使用 uv 安装的版本 |
| 训练 loss 发散 | 检查 norm_stats.json 中 q01/q99/std 是否异常小 |
| LeRobot 导入失败 | 确认运行了 `GIT_LFS_SKIP_SMUDGE=1 uv sync` |

---

## 11. 关键代码路径速查

### 11.1 模型结构

| 目标 | 文件 |
|------|------|
| pi0/pi0.5 前向、流匹配损失、欧拉推理 | `src/openpi/models/pi0.py` |
| pi0-FAST 自回归推理 | `src/openpi/models/pi0_fast.py` |
| 模型基类、Observation 结构 | `src/openpi/models/model.py` |
| 模型超参数配置 | `src/openpi/models/pi0_config.py` |
| Gemma LLM 骨干 | `src/openpi/models/gemma.py` |
| SigLIP 视觉编码器 | `src/openpi/models/siglip.py` |
| LoRA 实现 | `src/openpi/models/lora.py` |
| PyTorch 版本模型 | `src/openpi/models_pytorch/` |

### 11.2 数据处理

| 目标 | 文件 |
|------|------|
| 所有 Transform 实现 | `src/openpi/transforms.py` |
| LIBERO 数据适配 | `src/openpi/policies/libero_policy.py` |
| DROID 数据适配 | `src/openpi/policies/droid_policy.py` |
| ALOHA 数据适配 | `src/openpi/policies/aloha_policy.py` |
| 数据加载器 | `src/openpi/training/data_loader.py` |

### 11.3 训练配置

| 目标 | 文件 |
|------|------|
| 所有预定义 TrainConfig | `src/openpi/training/config.py` |
| 优化器和学习率调度 | `src/openpi/training/optimizer.py` |
| 检查点保存/恢复 | `src/openpi/training/checkpoints.py` |
| 预训练权重加载器 | `src/openpi/training/weight_loaders.py` |
| JAX FSDP 分片 | `src/openpi/training/sharding.py` |

### 11.4 推理部署

| 目标 | 文件 |
|------|------|
| Policy 推理封装 | `src/openpi/policies/policy.py` |
| 策略工厂函数 | `src/openpi/policies/policy_config.py` |
| WebSocket 服务器 | `src/openpi/serving/websocket_policy_server.py` |
| 启动服务器脚本 | `scripts/serve_policy.py` |
| WebSocket 客户端 | `packages/openpi-client/src/openpi_client/websocket_client_policy.py` |
| 动作块代理 | `packages/openpi-client/src/openpi_client/action_chunk_broker.py` |

---

## 附录：快速上手命令清单

```bash
# 安装
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 计算归一化统计（微调前必须执行）
uv run scripts/compute_norm_stats.py --config-name pi05_libero

# 启动微调训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero \
    --exp-name=my_experiment --overwrite

# 启动推理服务器（训练好的检查点）
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=checkpoints/pi05_libero/my_experiment/20000

# 使用默认预训练模型
uv run scripts/serve_policy.py --env droid

# 转换 JAX 检查点为 PyTorch
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax_ckpt \
    --config_name pi05_libero \
    --output_path /path/to/pytorch_ckpt

# PyTorch 单卡训练
uv run scripts/train_pytorch.py pi05_libero --exp_name my_run

# 调试（无 GPU）
uv run scripts/train.py debug
```

---

*文档生成时间：2026-03-25*
*基于 openpi-main（Physical Intelligence 开源版本，包含 pi0 / pi0-FAST / pi0.5 模型）*
