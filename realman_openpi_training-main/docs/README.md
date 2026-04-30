# Realman RM75-6FB 机械臂 π₀/π₀.5 文档索引

> **文档版本**: v1.3  
> **最后更新**: 2026-01-21  
> **适用框架**: OpenPI + RoboCOIN unified_deploy

---

## 📚 文档结构

本目录包含 Realman 机械臂 OpenPI 训练和部署的完整文档，按功能模块组织：

### 核心文档

| 文档 | 说明 | 适用人群 |
|------|------|---------|
| [1_training.md](./1_training.md) | 训练流程完整指南 | 训练工程师 |
| [2_deployment.md](./2_deployment.md) | 部署推理完整指南 | 部署工程师 |
| [3_faq.md](./3_faq.md) | 常见问题与解决方案 | 所有用户 |

### 辅助文档

| 文档 | 说明 |
|------|------|
| [bug_records.md](./bug_records.md) | Bug 记录与修复历史 |
| [CHANGELOG_task_display_fix.md](./CHANGELOG_task_display_fix.md) | Task 显示修复的详细记录 |

---

## 🚀 快速开始

### 训练流程（4 步）

```bash
# 1. 转换数据集 action 字段
python realman_openpi_training/convert_action_to_next_state.py \
    --input-path /path/to/original/dataset \
    --output-path /path/to/converted/dataset

# 2. 重新计算 episodes_stats（可选但推荐）
conda activate robocoin
python realman_openpi_training/recompute_episodes_stats.py \
    --dataset-path /path/to/converted/dataset \
    --original-dataset-path /path/to/original/dataset

# 3. 计算 OpenPI 归一化统计量
cd openpi
uv run ../realman_openpi_training/compute_norm_stats.py \
    --config-name pi0_realman_pytorch \
    --dataset-path /path/to/converted/dataset

# 4. 开始训练
uv run ../realman_openpi_training/train.py pi0_realman_pytorch --exp_name=my_exp
```

详见 [1_training.md](./1_training.md)

---

### 部署流程（3 终端）

**终端 1：OpenPI 推理服务**
```bash
cd openpi
source .venv/bin/activate
export PYTORCH_DISABLE_TRITON=1

python ../realman_openpi_training/serve_realman_policy.py \
    --config=pi0_realman_inference \
    --checkpoint=/path/to/checkpoint/30000 \
    --prompt="pick up the yellow banana" \
    --port=8000
```

**终端 2：RoboCOIN PolicyServer**
```bash
conda activate robocoin
cd RoboCOIN

python -m lerobot.extensions.unified_deploy.server.policy_server \
    --policy_type=openpi \
    --pretrained_path=dummy \
    --openpi_remote_host=localhost:8000 \
    --n_action_steps=10
```

**终端 3：Realman Client**
```bash
python -m lerobot.extensions.unified_deploy.client.realman_client \
    --robot_ip="192.168.1.18" \
    --server_address=127.0.0.1:50051 \
    --frequency=10 \
    --camera_configs="..." \
    --visualize
```

详见 [2_deployment.md](./2_deployment.md)

---

## 📖 术语表

| 术语 | 含义 |
|------|------|
| **cam0_rgb** | 腕部相机 (wrist camera)，安装在机械臂末端 |
| **cam1_rgb** | 基座相机 (base camera)，固定在工作台上 |
| **base_0_rgb** | OpenPI 模型的第一个相机槽位（通常用于基座视角） |
| **left_wrist_0_rgb** | OpenPI 模型的第二个相机槽位（通常用于腕部视角） |
| **delta action** | 相对动作，表示当前状态与目标状态的差值 |
| **absolute action** | 绝对动作，表示目标状态的绝对值 |
| **action_horizon** | 动作序列长度（每次推理预测多少步动作，OpenPI 为 50） |
| **n_action_steps** | 每次推理实际返回的动作步数（推荐 10-20） |
| **action_dim** | 动作维度（模型内部固定为 32，实际使用 8 维） |

---

## 🔗 相关资源

### 官方仓库

- [OpenPI](https://github.com/Physical-Intelligence/openpi) - π₀/π₀.5 官方实现
- [LeRobot](https://github.com/huggingface/lerobot) - HuggingFace 机器人学习库

### 论文

- [π₀: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)

### 数据集格式

- [LeRobot Dataset Format v2.1](https://huggingface.co/docs/lerobot/dataset_format)

---

## 💡 常见问题速查

| 问题 | 解决方案 | 详细文档 |
|------|---------|---------|
| 训练时报错 "norm_stats.json not found" | 运行 `compute_norm_stats.py` | [3_faq.md#q1](./3_faq.md#q1-训练时报错-norm_statsjson-not-found) |
| 推理时动作幅度异常 | 检查 delta 配置和数据转换 | [3_faq.md#q3](./3_faq.md#q3-推理时动作幅度过大或过小) |
| 显存不足 (OOM) | 减小 batch_size 或使用 LoRA | [3_faq.md#q4](./3_faq.md#q4-显存不足-oom) |
| 如何修改推理时的任务指令 | 修改 `serve_realman_policy.py` 的 `--prompt` | [2_deployment.md#58](./2_deployment.md#58-vla-任务指令text-instruction配置) |
| 如何调整推理响应性 | 调整 `n_action_steps` 参数 | [2_deployment.md#57](./2_deployment.md#57-n_action_steps-参数详解) |

---

## 📞 获取帮助

1. **查看文档**：先查阅对应的文档章节
2. **常见问题**：检查 [3_faq.md](./3_faq.md) 中是否有类似问题
3. **Bug 记录**：查看 [bug_records.md](./bug_records.md) 了解已知问题

---

*文档索引结束*
