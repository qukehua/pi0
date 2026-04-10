# Bug 记录文档

本文档记录 Realman OpenPI 训练/推理过程中遇到的典型 bug，包括原因分析、数据链路追踪和解决方案。

---

## BUG-001: 推理时图像 Key 映射冲突导致 ValueError

### 问题日期
2026-01-21

### 错误信息

**终端 3（serve_realman_policy.py）报错**：
```
ValueError: 图像 key 'cam1_rgb' 在数据集中未找到。可用的 key: ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']
```

**终端 2（policy_server.py）报错**：
```
websockets.exceptions.ConnectionClosedError: received 1011 (internal error) Internal server error.
```

### 问题根源

`openpi_adapter.py` 与 `RealmanInputs` transform 的职责重叠，导致图像 key 被错误地**双重映射**。

### 详细分析

#### 训练时的数据流（正确）

```
LeRobot 数据集
    │
    ├── observation.images.cam0_rgb (腕部相机)
    ├── observation.images.cam1_rgb (基座相机)
    ├── observation.state [14]
    └── action [8]
    │
    ▼
repack_transforms
    │  提取 key 并简化路径
    │
    ├── images.cam0_rgb  (从 observation.images.cam0_rgb)
    ├── images.cam1_rgb  (从 observation.images.cam1_rgb)
    ├── state [14]
    └── actions [8]
    │
    ▼
RealmanInputs (data_transforms.inputs)
    │  根据 image_keys=("cam1_rgb", "cam0_rgb") 顺序提取:
    │    第1个 cam1_rgb → base_0_rgb (基座相机)
    │    第2个 cam0_rgb → left_wrist_0_rgb (腕部相机)
    │
    ├── image.base_0_rgb        (来自 cam1_rgb)
    ├── image.left_wrist_0_rgb  (来自 cam0_rgb)
    ├── image.right_wrist_0_rgb (零填充占位符)
    ├── image_mask.base_0_rgb = True
    ├── image_mask.left_wrist_0_rgb = True
    ├── image_mask.right_wrist_0_rgb = False
    └── state [8] (截取前8维)
    │
    ▼
model_transforms (Resize, Tokenize 等)
    │
    ▼
模型推理 ✓
```

#### 推理时的数据流（修复前 - 错误！）

```
realman_client
    │
    ├── cam0_rgb (腕部相机)
    ├── cam1_rgb (基座相机)
    └── state [14]
    │
    ▼
policy_server → openpi_adapter.prepare_observation()
    │
    │  【问题所在！做了不该做的映射】
    │  根据 key_mapping 将:
    │    cam1_rgb → base_0_rgb
    │    cam0_rgb → left_wrist_0_rgb
    │
    ├── image.base_0_rgb         ← 已经是 OpenPI 格式
    ├── image.left_wrist_0_rgb   ← 已经是 OpenPI 格式
    └── image.right_wrist_0_rgb  (零填充)
    │
    ▼
WebSocket 发送到 serve_realman_policy.py
    │
    ▼
Policy.infer() → _input_transform → RealmanInputs
    │
    │  【报错！】
    │  RealmanInputs 期望的 key: "cam1_rgb", "cam0_rgb"
    │  实际收到的 key: "base_0_rgb", "left_wrist_0_rgb"
    │
    ✗ ValueError: 图像 key 'cam1_rgb' 在数据集中未找到
```

#### 问题总结表

| 组件 | 训练时输入 | 推理时输入 (修复前) | 问题 |
|------|-----------|---------------------|------|
| `RealmanInputs` | `{cam0_rgb, cam1_rgb}` | `{base_0_rgb, left_wrist_0_rgb}` ❌ | key 不匹配 |
| `openpi_adapter` | N/A | 错误地做了 key 映射 | 职责越界 |

**职责划分混乱**：
- `openpi_adapter.prepare_observation()` 做了 `cam_rgb → openpi_slot` 的映射（第 465-508 行）
- `RealmanInputs` 也做了同样的映射（第 112-143 行）
- 推理时两者串联，导致 `RealmanInputs` 收到的是已经映射过的 key

### 解决方案

**修改 `openpi_adapter.py` 的 `prepare_observation()` 方法**：

1. **不做图像 key 映射**：保持原始的 `cam0_rgb`, `cam1_rgb` key
2. **使用 `images` key**：与训练时 `repack_transforms` 的输出格式一致
3. **删除 `image_mask`**：由 `RealmanInputs` 生成

#### 修复后的推理数据流

```
realman_client
    │
    ├── cam0_rgb (腕部相机)
    ├── cam1_rgb (基座相机)
    └── state [14]
    │
    ▼
openpi_adapter.prepare_observation() 【修复后】
    │
    │  保持原始 key，不做映射
    │  使用 "images" key（与训练时一致）
    │
    ├── images.cam0_rgb  (保持原样)
    ├── images.cam1_rgb  (保持原样)
    └── state [14] (保持原样)
    │
    ▼
WebSocket → serve_realman_policy.py → RealmanInputs
    │
    │  与训练时完全一致的映射逻辑
    │  cam1_rgb → base_0_rgb
    │  cam0_rgb → left_wrist_0_rgb
    │
    ├── image.base_0_rgb
    ├── image.left_wrist_0_rgb
    └── image.right_wrist_0_rgb (占位)
    │
    ▼
模型推理 ✓
```

### 修改的文件

1. **`RoboCOIN/src/lerobot/extensions/unified_deploy/server/adapters/openpi_adapter.py`**
   - 重写 `prepare_observation()` 方法
   - 更新文件顶部的设计原则文档

### 设计原则（教训总结）

```
重要：图像 Key 映射职责划分

本适配器的 prepare_observation() 方法 **不负责** 图像 key 的映射！

训练时的数据流：
    LeRobot 数据集 → repack_transforms → RealmanInputs → 模型
    - repack_transforms: observation.images.cam0_rgb → images.cam0_rgb
    - RealmanInputs: images.cam0_rgb → image.left_wrist_0_rgb

推理时的数据流：
    realman_client → openpi_adapter → WebSocket → RealmanInputs → 模型
    - openpi_adapter: 保持原始 key (cam0_rgb, cam1_rgb)
    - RealmanInputs: cam0_rgb → left_wrist_0_rgb（与训练时一致）

这样设计的原因：
    1. 保证训练和推理使用完全相同的数据转换逻辑
    2. 避免 openpi_adapter 和 RealmanInputs 重复映射导致的错误
    3. 符合 OpenPI 的设计理念：transforms 链负责数据格式转换
```

### 验证方法

1. 重新启动三终端部署架构
2. 观察终端 3 是否还有 ValueError 报错
3. 检查模型是否正常输出动作

### 相关文件链接

- `realman_openpi_training/realman_config.py` - `RealmanInputs` 类定义
- `RoboCOIN/.../openpi_adapter.py` - OpenPI 适配器
- `openpi/src/openpi/policies/policy.py` - OpenPI Policy 推理入口

---

## BUG-002: PyTorch Triton 编译导致 WebSocket 连接超时

### 问题日期
2026-01-21

### 错误信息

**场景 1：用户手动中断编译**

**终端 1（serve_realman_policy.py）报错**：
```
subprocess.CalledProcessError: Command '[...ptxas...] died with <Signals.SIGINT: 2>.
triton.runtime.errors.PTXASError: PTXAS error: `ptxas` failed with error code -2
```

**终端 2（policy_server.py）报错**：
```
websockets.exceptions.ConnectionClosedError: sent 1011 (internal error) keepalive ping timeout; no close frame received
```

**场景 2：编译时间过长导致 keepalive 超时（更常见）**

**终端 1（serve_realman_policy.py）日志**：
```
AUTOTUNE bmm(8x51x867, 8x867x256)
  triton_bmm_147 0.0143 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=True, ...
  triton_bmm_150 0.0174 ms 82.4% ACC_TYPE='tl.float32', ALLOW_TF32=True, ...
  ...
SingleProcess AUTOTUNE benchmarking takes 0.1932 seconds and 0.0001 seconds precompiling for 18 choices
```

**终端 2（policy_server.py）报错**：
```
2026-01-21 15:28:37,745 - websockets.client - ERROR - keepalive ping failed
TimeoutError: timed out while closing connection

websockets.exceptions.ConnectionClosedError: sent 1011 (internal error) keepalive ping timeout; no close frame received

[OpenPIAdapter] 推理失败 (尝试 1/3): sent 1011 (internal error) keepalive ping timeout; no close frame received
[OpenPIAdapter] WebSocket 连接断开，尝试重新连接...
[OpenPIAdapter] 等待 OpenPI 服务启动...
Waiting for server at ws://localhost:8000...
```

**终端 3（realman_client）日志**：
```
E0121 15:28:37.077000 231117 torch/_inductor/select_algorithm.py:2100] [7/0] Runtime error during autotuning: 
E0121 15:28:37.077000 231117 torch/_inductor/select_algorithm.py:2100] [7/0] No valid triton configs. OutOfResources: out of resource: shared memory, Required: 114688, Hardware limit: 101376.
```

**症状**：
- 终端 1 在第一次推理时进行 Triton 自动调优（AUTOTUNE），需要 1-2 分钟
- 终端 2 的 WebSocket 客户端 keepalive ping 超时（默认 20 秒）
- 终端 2 疯狂尝试重连，但终端 1 仍在编译，无法响应握手
- 机械臂完全不动（动作队列始终为空）

### 问题根源

**主要原因**：PyTorch 的 Triton JIT 编译器在第一次推理时编译 CUDA kernel，编译时间（1-2 分钟）超过 WebSocket keepalive 超时时间（20 秒）。

**次要原因**：
1. Triton 编译期间，WebSocket 服务无法响应任何请求（包括 ping/pong）
2. WebSocket 客户端的 keepalive 机制检测到超时，主动断开连接
3. 客户端尝试重连，但服务端仍在编译，握手失败

### 详细分析

#### 问题链路

```
第一次推理请求
    │
    ▼
终端 1: PyTorch 开始 Triton JIT 编译
    │
    ├─ 编译 CUDA kernel (需要 1-2 分钟)
    ├─ CPU 占用 100%，看起来像"卡住"
    ├─ WebSocket 服务无法响应任何请求
    │
    ▼
终端 2: WebSocket keepalive ping 超时 (20 秒)
    │
    ├─ 客户端主动断开连接
    ├─ 尝试重连，但服务端仍在编译
    ├─ 握手失败: "timed out while waiting for handshake response"
    │
    ▼
终端 2: 推理失败，返回空动作
    │
    ▼
终端 3: 动作队列为空
    │
    ▼
机械臂不动 ✗
```

#### 为什么会发生 Triton 编译？

PyTorch 2.0+ 默认启用 `torch.compile()` 优化，会在第一次推理时进行 Triton JIT 编译（需要 1-2 分钟）。编译期间：
- CPU 占用 100%，看起来像"卡住"
- WebSocket 服务无法响应任何请求（包括 keepalive ping）
- 日志输出：`AUTOTUNE bmm`, `triton_bmm_xxx` 等
- 编译完成后会缓存，后续推理很快

**关键问题**：编译时间（60-120s）远超 WebSocket keepalive 超时（20s），导致客户端认为连接断开。

### 解决方案

#### 方案 1：禁用 Triton 编译（推荐）

在启动终端 1 之前设置环境变量：

```bash
# 终端 1
cd /home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/openpi
source .venv/bin/activate

# 禁用 Triton 编译
export PYTORCH_DISABLE_TRITON=1
export TORCH_COMPILE_DISABLE=1

python ../realman_openpi_training/serve_realman_policy.py \
    --config=pi0_realman_inference \
    --checkpoint=/path/to/checkpoint/30000 \
    --prompt="pick up the banana" \
    --port=8000
```

**优点**：
- 立即启动，无需等待编译
- 避免 keepalive 超时问题

**缺点**：
- 推理速度稍慢（约 10-20% 性能损失）

#### 方案 2：等待编译完成

如果不禁用 Triton，第一次推理时：
1. **不要中断**，耐心等待 1-2 分钟
2. 观察 CPU 占用率 100% = 正在编译
3. 编译完成后会缓存，后续推理很快

**优点**：
- 编译后推理速度最快

**缺点**：
- 首次启动慢
- 容易误以为卡死而中断

#### 方案 3：增加 WebSocket keepalive 超时（不推荐）

修改 `websockets` 库的 keepalive 超时时间（需要修改代码），但这只是治标不治本。

### 最佳实践

1. **开发/调试阶段**：禁用 Triton（方案 1），快速迭代
2. **生产部署**：首次启动时等待编译完成（方案 2），后续享受最快推理速度
3. **始终启用自动重连**：`openpi_adapter.py` 已实现自动重连机制（最多 3 次）

### 修改的文件

1. **`RoboCOIN/src/lerobot/extensions/unified_deploy/server/adapters/openpi_adapter.py`**
   - 在 `predict()` 方法中添加 WebSocket 重连逻辑（最多重试 3 次）

### 验证方法

1. **测试禁用 Triton**：
   ```bash
   export PYTORCH_DISABLE_TRITON=1
   export TORCH_COMPILE_DISABLE=1
   python serve_realman_policy.py ...
   ```
   预期：服务立即启动，第一次推理无延迟，无 `AUTOTUNE` 日志

2. **测试自动重连**：
   - 启动三终端
   - 在推理过程中手动停止终端 1（Ctrl+C）
   - 重新启动终端 1
   - 观察终端 2 是否自动重连并继续推理

### 相关链接

- PyTorch Triton 文档：https://pytorch.org/docs/stable/torch.compiler.html
- Triton 编译器：https://github.com/openai/triton

---

## BUG-003: Task 显示为 None 和 Action Type 显示错误

### 问题日期
2026-01-21

### 错误信息

**Client 日志显示**：
```
[Client] 已从 Server 获取配置:
  任务: None                   ← 错误！应该显示 prompt
  动作空间: joints
  动作类型: absolute           ← 错误！应该是 delta
```

**Queue 波动异常**：
```
Frame 50, FPS=5.0, Queue=55   ← 队列堆积
Frame 100, FPS=5.0, Queue=5
Frame 150, FPS=5.0, Queue=57
```

### 问题根源

**问题 1：Task 显示为 None**
- 远程模式下，`serve_realman_policy.py` 的 `--prompt` 参数没有通过 WebSocket metadata 传递给 RoboCOIN

**问题 2：Action Type 显示为 Absolute**
- 远程模式下，RoboCOIN 无法访问 checkpoint 目录中的 `metadata.pt`，无法读取 `use_delta_joint_actions` 配置

**问题 3：Queue 波动异常**
- `n_action_steps` 参数没有生效，Server 返回 50 个动作全部放入队列，而不是只取前 N 个

### 详细分析

#### 数据流断裂点

**Task 传递链路**：
```
serve_realman_policy.py (--prompt="...")
    ↓ (传给 Policy，但不在 metadata 中)
WebSocket metadata (缺少 prompt) ❌
    ↓
OpenPIAdapter (无法获取 prompt)
    ↓
policy_server.py metadata["task"] = None ❌
```

**Action Type 推断链路**：
```
realman_config.py (use_delta_joint_actions=True)
    ↓ (训练时配置)
metadata.pt (存储在 checkpoint 中)
    ↓ (远程模式无法访问) ❌
OpenPIAdapter._infer_action_type()
    ↓ (找不到 delta_source)
返回 "absolute" ❌
```

**Queue 堆积原因**：
```python
# Server 返回 50 个动作
action_chunk = [action_0, ..., action_49]

# Client 全部放入队列（错误！）
for action in action_chunk.actions:
    queue.put(action)  # 放入 50 个

# 导致队列堆积
Queue: 0 → 50 → 49 → 48 → ... → 5 → 55 → ...
```

### 解决方案

#### 修改 1：扩展 `serve_realman_policy.py` 的 metadata

**文件**：`realman_openpi_training/serve_realman_policy.py`

```python
# 从 train_config 提取配置信息
use_delta = False
default_prompt_from_config = None

if hasattr(train_config, 'data'):
    data_config = train_config.data
    if hasattr(data_config, 'use_delta_joint_actions'):
        use_delta = data_config.use_delta_joint_actions
    if hasattr(data_config, 'default_prompt'):
        default_prompt_from_config = data_config.default_prompt

# 确定最终使用的 prompt（优先级：命令行 > 配置文件）
final_prompt = args.prompt if args.prompt else default_prompt_from_config

# 扩展 metadata
policy_metadata = dict(policy.metadata) if policy.metadata else {}
policy_metadata.update({
    "default_prompt": final_prompt,
    "use_delta_actions": use_delta,
    "config_name": args.config,
})
```

#### 修改 2：`OpenPIAdapter` 从 metadata 读取配置

**文件**：`RoboCOIN/src/lerobot/extensions/unified_deploy/server/adapters/openpi_adapter.py`

```python
# 在 _connect_websocket() 中读取配置
if self._server_metadata:
    self._default_prompt = self._server_metadata.get("default_prompt", "")
    self._use_delta_actions = self._server_metadata.get("use_delta_actions", False)

# 覆盖 _infer_action_type() 方法
def _infer_action_type(self) -> str:
    if self.is_remote_mode:
        if hasattr(self, '_use_delta_actions'):
            return "delta" if self._use_delta_actions else "absolute"
        return "absolute"
    # ... 本地模式逻辑
```

#### 修改 3：`policy_server.py` 使用 adapter 的 prompt

**文件**：`RoboCOIN/src/lerobot/extensions/unified_deploy/server/policy_server.py`

```python
# 优先使用 adapter 的 default_prompt
adapter_prompt = getattr(self.adapter, '_default_prompt', None)
final_task = adapter_prompt if adapter_prompt else (obs.task if obs.task else "")

metadata = {
    "action_space": self.adapter.action_space,
    "action_type": self.adapter.action_type,
    "task": final_task,
}
```

#### 修改 4：`realman_client.py` 截取 n_action_steps

**文件**：`RoboCOIN/src/lerobot/extensions/unified_deploy/client/realman_client.py`

```python
# 截取前 n_action_steps 个动作
if self.config.n_action_steps > 0 and len(actions) > self.config.n_action_steps:
    actions = actions[:self.config.n_action_steps]
    if self._frame_count == 0:
        print(f"[Client] 每次只使用前 {self.config.n_action_steps} 个动作")
```

### 修改的文件

1. ✅ `realman_openpi_training/serve_realman_policy.py`
2. ✅ `RoboCOIN/src/lerobot/extensions/unified_deploy/server/adapters/openpi_adapter.py`
3. ✅ `RoboCOIN/src/lerobot/extensions/unified_deploy/server/policy_server.py`
4. ✅ `RoboCOIN/src/lerobot/extensions/unified_deploy/client/realman_client.py`

### 验证方法

**预期效果**：
```bash
[Client] 已从 Server 获取配置:
  任务: pick up the yellow banana and put it in the white plate  ✅
  动作空间: joints  ✅
  动作类型: delta  ✅

[Client] 每次只使用前 3 个动作（共返回 50 个）

Frame 50, FPS=5.0, Queue=2    ✅ 队列稳定
Frame 100, FPS=5.0, Queue=1   ✅
Frame 150, FPS=5.0, Queue=3   ✅
```

### 核心改进

1. **✅ Task 显示正确**：Client 显示真实输入网络的 prompt
2. **✅ Action Type 正确**：正确识别 delta 模式
3. **✅ Queue 稳定**：队列大小稳定在 `n_action_steps` 附近
4. **✅ 架构统一**：task、action_space、action_type 都由 Server 决定
5. **✅ 响应速度提升**：不再堆积 50 个动作，推理更及时

---

## 模板：Bug 记录格式

```markdown
## BUG-XXX: 简短描述

### 问题日期
YYYY-MM-DD

### 错误信息
（粘贴完整的错误堆栈）

### 问题根源
（一句话总结）

### 详细分析
（数据流图、表格对比等）

### 解决方案
（具体的代码修改说明）

### 修改的文件
（列出修改的文件路径）

### 验证方法
（如何确认问题已解决）
```
