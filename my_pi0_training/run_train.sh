#!/bin/bash
# ============================================================================
# Pi0 右臂抓取训练启动脚本
# 任务: 右臂抓取箱子，提起来，放回原位
# 数据: 86条 × ~15s @ 30fps，总帧 34911
#
# 用法:
#   bash run_train.sh              # 全新训练
#   bash run_train.sh --resume     # 断点续训
#   bash run_train.sh --stop       # 停止训练
# ============================================================================

set -e

PYTHON="/share/0xyj/model3_openpi0.5/openpi-main/.venv/bin/python3.11"
MY_DIR="/share/0xyj/model3_openpi0.5/my_pi0_training"
PID_FILE="${MY_DIR}/train.pid"
LOG_DIR="${MY_DIR}/logs"
CONFIG_NAME="pi0_right_arm_pytorch"
EXP_NAME="right_arm_box_pick_v1"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
LATEST_LOG="${MY_DIR}/train_latest.log"

# ---- 停止训练 ----
if [ "$1" == "--stop" ]; then
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat "${PID_FILE}")
        echo "[INFO] Stopping training process PID=${PID}..."
        kill -SIGTERM "${PID}" 2>/dev/null && echo "[INFO] Sent SIGTERM to ${PID}" || echo "[WARN] Process not found"
        rm -f "${PID_FILE}"
    else
        echo "[INFO] No PID file found, trying pkill..."
        pkill -f "train.py.*${CONFIG_NAME}" 2>/dev/null || echo "[WARN] No matching process"
    fi
    exit 0
fi

# ---- 检查是否已在运行 ----
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[WARN] Training already running (PID=${OLD_PID})"
        echo "[WARN] Use: bash run_train.sh --stop  to stop it first"
        exit 1
    else
        echo "[INFO] Stale PID file found, removing..."
        rm -f "${PID_FILE}"
    fi
fi

# ---- 断点续训标志 ----
RESUME_FLAG=""
if [ "$1" == "--resume" ]; then
    RESUME_FLAG="--resume"
    echo "[INFO] Resume mode: will continue from latest checkpoint"
fi

echo "========================================================"
echo " Pi0 右臂抓取训练"
echo " Config:  ${CONFIG_NAME}"
echo " Exp:     ${EXP_NAME}"
echo " Log:     ${LOG_FILE}"
echo " Resume:  ${RESUME_FLAG:-no}"
echo "========================================================"

# ---- 关键环境变量 ----
export JAX_PLATFORMS=cpu
export PYTHONNOUSERSITE=1
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[ERROR] WANDB_API_KEY is not set. Export it before running this script."
    exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENPI_DATA_HOME=/tmp/openpi_cache

# ---- 检测 GPU 数量，自动选择单卡/双卡 ----
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "${NUM_GPUS}" -ge 2 ]; then
    echo "[INFO] 双卡 DDP 模式 (${NUM_GPUS} GPUs)"
    nohup /share/0xyj/model3_openpi0.5/openpi-main/.venv/bin/torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=29500 \
        "${MY_DIR}/train.py" \
        "${CONFIG_NAME}" \
        --exp_name="${EXP_NAME}" \
        ${RESUME_FLAG} \
        >> "${LOG_FILE}" 2>&1 &
else
    echo "[INFO] 单卡模式 (1 GPU)"
    nohup "${PYTHON}" -u \
        "${MY_DIR}/train.py" \
        "${CONFIG_NAME}" \
        --exp_name="${EXP_NAME}" \
        ${RESUME_FLAG} \
        >> "${LOG_FILE}" 2>&1 &
fi

TRAIN_PID=$!
echo ${TRAIN_PID} > "${PID_FILE}"
echo "[INFO] Training started! PID=${TRAIN_PID}"
echo "[INFO] Log file: ${LOG_FILE}"
echo "[INFO] Latest log link: ${LATEST_LOG}"

# 更新软链接指向最新日志
ln -sf "${LOG_FILE}" "${LATEST_LOG}"

echo ""
echo "[INFO] 实时查看训练日志:"
echo "  tail -f ${LATEST_LOG}"
echo ""
echo "[INFO] 停止训练:"
echo "  bash ${MY_DIR}/run_train.sh --stop"
echo ""
echo "[INFO] 断点续训:"
echo "  bash ${MY_DIR}/run_train.sh --resume"
echo ""
echo "[INFO] WandB 实时监控:"
echo "  https://wandb.ai"
