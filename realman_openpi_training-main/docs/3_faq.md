# Realman π₀/π₀.5 常见问题与解决方案

本文档收集了训练和部署过程中的常见问题及解决方案。

> **文档版本**: v1.3  
> **最后更新**: 2026-01-21  
> **适用框架**: OpenPI + RoboCOIN unified_deploy

---

## 📚 文档导航

- **[← 返回索引](./README.md)**
- **[训练流程](./1_training.md)**
- **[部署推理](./2_deployment.md)**

---

## 1. 动作空间说明

### 1.1 State 维度

Realman RM75-6FB 的 state 包含 14 维：

| 索引 | 名称 | 单位 | 说明 |
|------|------|------|------|
| 0-6 | joint_1 ~ joint_7 | rad | 7 个关节角度 |
| 7 | gripper_open | [0, 1] | 夹爪开合度（0=闭合，1=完全打开） |
| 8-10 | eef_pos_x/y/z | m | 末端执行器位置 |
| 11-13 | eef_rot_euler_x/y/z | rad | 末端执行器姿态（欧拉角） |

**注意**：OpenPI 模型只使用前 8 维（7 joints + 1 gripper）。

### 1.2 Action 维度

模型输出的 action 为 8 维：

| 索引 | 名称 | 模式 | 说明 |
|------|------|------|------|
| 0-6 | joint_1 ~ joint_7 | Delta | 关节角度变化量（rad） |
| 7 | gripper | Absolute | 目标夹爪开合度 [0, 1] |

### 1.3 Delta vs Absolute 模式

OpenPI 使用混合模式：

```python
# Delta mask 定义
delta_mask = make_bool_mask(7, -1)  # [True, True, True, True, True, True, True, False]
```

| 维度 | 模式 | 计算方式 | 执行方式 |
|------|------|----------|----------|
| 0-6 (joints) | Delta | `delta = action - state` | `target = state + delta` |
| 7 (gripper) | Absolute | `target = action` | 直接使用 action 值 |

**为什么夹爪用 Absolute？**
- 夹爪状态是离散的（开/闭），delta 模式容易累积误差
- Absolute 模式可以直接指定目标状态，更稳定

### 1.4 单位说明

| 量 | 训练/推理单位 | 机械臂 SDK 单位 | 转换 |
|----|--------------|----------------|------|
| 关节角度 | rad | rad | 无需转换 |
| 夹爪开合 | [0, 1] | [0, 1000] | × 1000 |
| 末端位置 | m | mm | × 1000 |
| 末端姿态 | rad | rad | 无需转换 |

**RoboCOIN unified_deploy 会自动处理单位转换**，无需手动转换。

### 1.5 Action Horizon 与 Action Dim

| 参数 | 值 | 说明 |
|------|-----|------|
| `action_horizon` | 50 | 每次推理预测 50 步动作 |
| `action_dim` | 32 | 模型内部动作维度（固定） |
| 实际使用维度 | 8 | 只使用前 8 维（7 joints + 1 gripper） |

模型输出 shape: `[action_horizon, action_dim]` = `[50, 32]`，实际使用 `[:, :8]`。

---

## 2. 常见问题

### Q1: 训练时报错 "norm_stats.json not found"

**原因**：未计算归一化统计量。

**解决**：
```bash
cd openpi
uv run ../realman_openpi_training/compute_norm_stats.py \
    --config-name pi0_realman_pytorch \
    --dataset-path /path/to/your/dataset
```

### Q2: 训练时报错 "KeyError: 'observation.images.cam0_depth'"

**原因**：数据集包含深度图，但 norm_stats 验证时尝试检查深度图统计量。

**解决**：这个问题已在最新版本中修复。深度图会被自动过滤，不会影响训练。如果仍然遇到，请确保使用最新的 `realman_config.py`。

### Q3: 推理时动作幅度过大或过小

**可能原因**：
1. 训练时和推理时的 `use_delta_joint_actions` 设置不一致
2. 数据集未进行 action 转换（见 2.1 节）
3. norm_stats 计算错误

**排查步骤**：
1. 确认训练配置中 `use_delta_joint_actions=True`
2. 确认推理配置中也是 `use_delta_joint_actions=True`
3. 检查数据集是否已转换（原始 delta 应该很小，约 0.01-0.02 rad）
4. 重新计算 norm_stats

### Q4: 显存不足 (OOM)

**解决方案**：

1. **减小 batch_size**：
   ```python
   # 在 realman_config.py 中修改
   batch_size=4  # 从 8 减小到 4
   ```

2. **使用 LoRA 微调**：
   ```bash
   uv run ../realman_openpi_training/train.py pi0_realman_lora --exp_name=my_exp
   ```

3. **设置显存优化环境变量**：
   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run ...
   ```

### Q5: 如何从 checkpoint 恢复训练？

```bash
uv run ../realman_openpi_training/train.py pi0_realman_pytorch \
    --exp_name=my_exp \
    --resume
```

训练会自动从最新的 checkpoint 恢复。

### Q6: 如何使用多任务数据集？

1. 确保数据集的 `meta/tasks.jsonl` 包含任务描述
2. 配置 `prompt_from_task=True`：
   ```python
   data=LeRobotRealmanDataConfig(
       repo_id="local/multi_task_dataset",
       prompt_from_task=True,
       default_prompt=None,
   ),
   ```

### Q7: 训练和推理的相机顺序不一致怎么办？

**关键**：训练和推理时的 `image_keys` 顺序必须一致！

如果训练时：
```python
image_keys=("observation.images.cam1_rgb", "observation.images.cam0_rgb")
```

推理时也必须按相同顺序传入图像：
```python
images={
    "base_0_rgb": cam1_image,      # 基座相机
    "left_wrist_0_rgb": cam0_image,  # 腕部相机
}
```

### Q8: 如何评估训练效果？

1. **查看训练曲线**：使用 WandB 查看 loss 曲线
2. **离线评估**：使用 RoboCOIN 的离线评估功能
3. **真机测试**：部署到真机进行实际测试

### Q9: 训练时报错 "norm_stats.json not found"，但我已经计算过了？

**⚠️ 这是一个常见的配置陷阱！**

**问题现象**：
- 你运行了 `compute_norm_stats.py`，脚本显示成功保存
- 但训练时仍然报错找不到 `norm_stats.json`

**根本原因**：`repo_id` 配置不一致导致路径不匹配。

**路径生成逻辑**：
```python
# compute_norm_stats.py 的输出路径逻辑
output_dir = config.assets_dirs / data_config.repo_id
# 例如：openpi/assets/pi0_realman_pytorch/local/realman_teleop_xxx
```

**典型错误场景**：

| 配置文件中的 repo_id | 实际保存路径 | 结果 |
|---------------------|-------------|------|
| `realman_teleop_130_openpi` | `assets/.../realman_teleop_130_openpi/` | ✅ 匹配 |
| `realman_teleop_130_openpi` | `assets/.../local/realman_teleop_0111_xxx/` | ❌ 不匹配！ |

**排查步骤**：

1. **检查 norm_stats 实际保存位置**：
   ```bash
   find openpi/assets -name "norm_stats.json"
   ```

2. **检查配置文件中的 repo_id**：
   ```bash
   grep "repo_id" realman_openpi_training/realman_config.py
   ```

3. **确认两者路径一致**

**解决方案**：

**方案 1：修改配置文件中的 repo_id（推荐）**

确保 `realman_config.py` 中的 `repo_id` 与你期望的路径一致：

```python
data=LeRobotRealmanDataConfig(
    # repo_id 决定了 norm_stats.json 的查找路径
    repo_id="local/realman_teleop_0111_right_plate_50",  # ← 必须与保存路径一致
    local_root="/path/to/your/dataset",
    ...
),
```

**方案 2：使用 --output-path 参数明确指定**

运行 `compute_norm_stats.py` 时显式指定输出路径：

```bash
uv run ../realman_openpi_training/compute_norm_stats.py \
    --config-name pi0_realman_pytorch \
    --dataset-path /path/to/your/dataset \
    --output-path openpi/assets/pi0_realman_pytorch/local/your_dataset_name
```

**最佳实践**：

1. **统一命名**：让 `repo_id` 与数据集目录名保持一致
2. **使用 `local/` 前缀**：本地数据集建议使用 `local/dataset_name` 格式
3. **修改配置后重新计算**：如果修改了 `repo_id`，需要重新运行 `compute_norm_stats.py`

### Q10: PyTorch 训练报错 "transformers_replace is not installed correctly"

**问题现象**：
```
ValueError: transformers_replace is not installed correctly. 
Please install it with `uv pip install transformers==4.53.2` and 
`cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`.
```

**原因**：OpenPI 的 PyTorch 训练模式需要修改过的 transformers 库文件（用于支持 PaliGemma 等模型的特殊功能）。

**解决方案**：

1. **确认 transformers 版本**：
   ```bash
   conda activate openpi
   python -c "import transformers; print(transformers.__version__)"
   # 应该输出: 4.53.2
   ```

2. **复制修改过的模型文件到 site-packages**：
   ```bash
   # 如果使用 conda 环境
   cp -rv /workspaces/lerobot_training/openpi/src/openpi/models_pytorch/transformers_replace/models/* \
       /workspaces/_conda/envs/openpi/lib/python3.11/site-packages/transformers/models/
   
   # 如果使用 uv 虚拟环境
   cp -rv ./src/openpi/models_pytorch/transformers_replace/* \
       .venv/lib/python3.11/site-packages/transformers/
   ```

**⚠️ 注意**：
- 这个操作需要在**每个新的服务器/环境**上执行一次
- 如果你在多台服务器上训练，需要分别在每台服务器上执行此操作
- 执行后会覆盖 `gemma`、`paligemma`、`siglip` 三个模型目录

### Q11: 数据加载时报错 "TypeError: stack(): argument 'tensors' must be tuple of Tensors, not Column"

**问题现象**：
```
TypeError: stack(): argument 'tensors' (position 1) must be tuple of Tensors, not Column
```

**原因**：HuggingFace `datasets` 库 4.x 版本的 API 变化。`ds["column"]` 返回 `Column` 对象而非 list，与 `torch.stack()` 不兼容。

**解决方案**：

此问题已在 `realman_openpi_training/train.py` 中通过 Monkey Patch 修复。确保使用最新版本的训练脚本即可。

**技术细节**：
- `train.py` 中的 `patch_data_loader_for_local_datasets()` 函数会自动处理此兼容性问题
- 修复方式：临时替换 `torch.stack` 函数，使其能正确处理 `Column` 对象

### Q12: 数据集加载非常慢（10+ 分钟）

**问题现象**：
- 训练启动后，显示 `Loading dataset shards: 100%` 后长时间无响应
- 进程 CPU 占用 100%，但没有日志输出

**原因**：LeRobot 数据集初始化时会进行时间戳同步检查（`check_timestamps_sync`），对于大数据集需要加载全部时间戳数据。

**解决方案**：

此问题已在 `realman_openpi_training/train.py` 中优化。训练脚本会自动跳过时间戳检查，大幅加速数据加载。

**优化效果**：

| 阶段 | 优化前 | 优化后 |
|------|--------|--------|
| 时间戳数据加载 | 5-10 分钟 | 跳过 |
| 时间戳检查 | 1-2 分钟 | 跳过 |
| 总初始化时间 | 10-15 分钟 | 1-2 分钟 |

**⚠️ 注意**：
- 时间戳检查是 LeRobot 官方的数据质量验证机制
- 跳过检查后，训练仍能正常进行（每个样本的时间戳仍可按需获取）
- 如果需要验证数据集质量，可以单独运行验证脚本

### Q13: 多卡训练报错 "Normalization stats not found"

**问题现象**：
```
ValueError: Normalization stats not found. 
Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`.
```

**原因**：不同配置（如 `pi0_realman_pytorch` vs `pi05_realman_pytorch`）的 norm_stats 存储路径不同。

**解决方案**：

1. **检查 norm_stats 存储位置**：
   ```bash
   # 查看已有的 norm_stats
   find openpi/assets -name "norm_stats.json"
   ```

2. **为对应配置计算 norm_stats**：
   ```bash
   cd openpi
   
   # 为 pi05_realman_pytorch 计算
   python ../realman_openpi_training/compute_norm_stats.py \
       --config-name pi05_realman_pytorch \
       --dataset-path /path/to/your/dataset \
       --output-path openpi/assets/pi05_realman_pytorch/your_dataset_repo_id
   ```

3. **或者复制已有的 norm_stats**（如果数据格式相同）：
   ```bash
   mkdir -p openpi/assets/pi05_realman_pytorch/realman_teleop_130_openpi_0120_pi05
   cp openpi/assets/pi0_realman_pytorch/realman_teleop_130_openpi_0120/norm_stats.json \
      openpi/assets/pi05_realman_pytorch/realman_teleop_130_openpi_0120_pi05/
   ```

**路径对应关系**：

| 配置名 | norm_stats 路径 |
|--------|----------------|
| `pi0_realman_pytorch` | `assets/pi0_realman_pytorch/<repo_id>/norm_stats.json` |
| `pi05_realman_pytorch` | `assets/pi05_realman_pytorch/<repo_id>/norm_stats.json` |

---

## 3. 参考资料

### 3.1 相关仓库

| 仓库 | 说明 | 链接 |
|------|------|------|
| OpenPI | π₀/π₀.5 官方实现 | https://github.com/Physical-Intelligence/openpi |
| LeRobot | HuggingFace 机器人学习库 | https://github.com/huggingface/lerobot |
| RoboCOIN | 本项目的统一部署框架 | 本地 `RoboCOIN/` 目录 |

### 3.2 论文

- **π₀**: [π₀: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)
- **π₀.5**: π₀ 的改进版本，支持更长的动作序列和更好的泛化能力

### 3.3 数据集格式

- [LeRobot Dataset Format v2.1](https://huggingface.co/docs/lerobot/dataset_format)

### 3.4 本项目文档

| 文档 | 路径 | 说明 |
|------|------|------|
| 训练模块 README | `realman_openpi_training/README.md` | 训练模块快速入门 |
| RoboCOIN 部署文档 | `RoboCOIN/docs/training_pipeline.md` | 统一部署框架说明 |
| OpenPI 远程推理 | `openpi/docs/remote_inference.md` | OpenPI 官方推理文档 |

---

## 4. 附录：完整配置示例

### 4.1 单任务训练配置

```python
_config.TrainConfig(
    name="pi0_realman_single_task",
    model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
    data=LeRobotRealmanDataConfig(
        repo_id="local/realman_pick_place",
        local_root="/data/datasets/realman_pick_place",
        use_delta_joint_actions=True,
        default_prompt="pick up the red cube and place it in the blue box",
        prompt_from_task=False,
        image_keys=(
            "observation.images.cam1_rgb",  # 基座相机
            "observation.images.cam0_rgb",  # 腕部相机
        ),
    ),
    pytorch_weight_path="data/openpi-assets/checkpoints/pi0_droid_pytorch",
    num_train_steps=30000,
    batch_size=8,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1000, 
        peak_lr=5e-5, 
        decay_steps=30000, 
        decay_lr=5e-6
    ),
    ema_decay=0.999,
)
```

### 4.2 多任务训练配置

```python
_config.TrainConfig(
    name="pi05_realman_multi_task",
    model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
    data=LeRobotRealmanDataConfig(
        repo_id="local/realman_multi_task",
        local_root="/data/datasets/realman_multi_task",
        use_delta_joint_actions=True,
        prompt_from_task=True,  # 从数据集读取任务描述
        default_prompt=None,
        image_keys=(
            "observation.images.cam1_rgb",
            "observation.images.cam0_rgb",
        ),
    ),
    pytorch_weight_path="data/openpi-assets/checkpoints/pi05_droid_pytorch",
    num_train_steps=50000,
    batch_size=8,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=2000, 
        peak_lr=3e-5, 
        decay_steps=50000, 
        decay_lr=3e-6
    ),
    ema_decay=0.999,
)
```

---

*文档结束*
