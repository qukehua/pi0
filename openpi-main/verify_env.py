import sys
import os

print("=" * 50)
print("🚀 OpenPI 服务器环境全面自检 🚀")
print("=" * 50)

# 1. Python 版本检查
v = sys.version_info
print(f"[1] Python 版本 : {v.major}.{v.minor}.{v.micro} \t" + ("✅" if v.major==3 and v.minor>=11 else "❌ (需>=3.11)"))

# 2. PyTorch & CUDA 检查
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"[2] PyTorch 状态: ✅ (版本: {torch.__version__}, CUDA 加速: {'✅ 正常' if cuda_ok else '❌ 失败'})")
except Exception as e:
    print(f"[2] PyTorch 状态: ❌ (加载失败: {e})")

# 3. JAX & GPU 检查 (OpenPI 推理/训练大量依赖 JAX)
try:
    import jax
    gpus = sum(1 for d in jax.devices() if d.platform == 'gpu')
    print(f"[3] JAX 框架状态: ✅ (版本: {jax.__version__}, 识别到 GPU 数量: {gpus})")
except Exception as e:
    print(f"[3] JAX 框架状态: ❌ (加载失败或未探测到显卡: {e})")

# 4. OpenPI 核心组件包检查
print("-" * 50)
packages = ["flax", "openpi", "openpi_client", "lerobot", "dlimp"]
for i, pkg in enumerate(packages, start=4):
    try:
        __import__(pkg)
        print(f"[{i}] {pkg:<13}: ✅ 导入成功")
    except Exception as e:
        print(f"[{i}] {pkg:<13}: ❌ 导入失败 ({e})")
print("-" * 50)

# 5. 模型预训练权重检查
model_dir = os.path.expanduser("~/.cache/openpi/checkpoints/pi05_base")
if os.path.exists(model_dir) and os.listdir(model_dir):
    print(f"[9] 模型权重文件: ✅ (已检测到存放于 {model_dir})")
else:
    print(f"[9] 模型权重文件: ⚠️ (未找到。如果您尚未执行模型下载步骤，请忽略此项)")

print("=" * 50)
