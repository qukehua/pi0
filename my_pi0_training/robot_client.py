#!/usr/bin/env python3
"""Pi0 右臂抓取 —— 机器人端推理客户端

部署在 Orin AGX 上，对接真机。
用法:
    pip install openpi-client numpy opencv-python pyrealsense2
    python robot_client.py --host <服务器IP> --port 8000
"""

import argparse
import logging
import time

import cv2
import numpy as np


def compress_image(img: np.ndarray, quality: int = 85) -> np.ndarray:
    """JPEG 压缩图像以减少网络传输量。
    输入: (H, W, 3) uint8 RGB
    输出: (H, W, 3) uint8 RGB（经 JPEG 压缩后解压，保持格式兼容）
    传输数据量: ~450KB → ~40KB，WiFi 延迟从 ~300ms 降至 ~30ms
    """
    # RGB → BGR for cv2
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    bgr_dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr_dec, cv2.COLOR_BGR2RGB)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 配置区（根据真机修改）
# ─────────────────────────────────────────────────────────────────────────────

# 推理配置
TASK_PROMPT = "the right arm picks up the box and places it back to its original position"
ACTION_HORIZON = 25        # 每次执行前25帧（共预测50帧，执行一半再重新推理）
CONTROL_FREQ_HZ = 25       # 控制频率 25Hz

# 相机分辨率（ResensenseD435/D405 默认输出，会被 resize 到 224x224）
CAM_WIDTH = 640
CAM_HEIGHT = 480

# 右臂关节数（7关节 + 1灵巧手）
STATE_DIM = 8


# ─────────────────────────────────────────────────────────────────────────────
# 相机接口（使用 pyrealsense2）
# ─────────────────────────────────────────────────────────────────────────────

class RealSenseCamera:
    """RealSense D435/D405 相机接口"""

    def __init__(self, serial_number: str | None = None, name: str = "cam"):
        self.name = name
        self.serial = serial_number
        self.pipeline = None

    def start(self):
        try:
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()
            config = rs.config()
            if self.serial:
                config.enable_device(self.serial)
            config.enable_stream(rs.stream.color, CAM_WIDTH, CAM_HEIGHT, rs.format.rgb8, 30)
            self.pipeline.start(config)
            logger.info(f"相机 [{self.name}] 启动成功 (SN: {self.serial or 'auto'})")
        except Exception as e:
            logger.warning(f"相机 [{self.name}] 启动失败: {e}，将使用随机图像占位")
            self.pipeline = None

    def get_image(self) -> np.ndarray:
        """返回 (224, 224, 3) uint8 RGB 图像"""
        if self.pipeline is None:
            # 相机未连接时返回随机图像（调试用）
            return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        try:
            import cv2
            import pyrealsense2 as rs
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            img = np.asanyarray(color_frame.get_data())  # (H, W, 3) RGB
            img = cv2.resize(img, (224, 224))
            return img
        except Exception as e:
            logger.warning(f"相机 [{self.name}] 读取失败: {e}")
            return np.zeros((224, 224, 3), dtype=np.uint8)

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 机器人接口（根据你的SDK修改）
# ─────────────────────────────────────────────────────────────────────────────

class RobotInterface:
    """真机接口 —— 根据你的机器人SDK实现"""

    def __init__(self):
        # TODO: 初始化你的机器人SDK
        # 例如: self.robot = YourRobotSDK()
        self._mock = True
        logger.warning("使用模拟机器人接口（调试模式），请替换为真实SDK")

    def get_right_arm_state(self) -> np.ndarray:
        """获取右臂关节状态，返回 (8,) float32
        
        前7维：右臂关节角度（弧度）
        第8维：灵巧手状态（0=开，1=闭）
        
        TODO: 替换为真实读取
        """
        if self._mock:
            return np.array([1.273, 0.657, -1.251, -1.479, -0.390, 0.745, 1.453, 0.328],
                           dtype=np.float32)
        # 真实实现示例:
        # joint_angles = self.robot.get_joint_angles()  # 返回14维双臂
        # right_arm = joint_angles[:7]                  # 取右臂前7维
        # hand_state = self.robot.get_hand_state()      # 灵巧手状态
        # return np.concatenate([right_arm, [hand_state]], dtype=np.float32)

    def execute_action(self, action: np.ndarray) -> bool:
        """执行一帧动作，action 为 (8,) float32（绝对关节角度）
        
        前7维：目标关节角度（弧度）
        第8维：灵巧手指令（>0.5=闭合，<=0.5=张开）
        
        返回: True=成功, False=急停/异常
        
        TODO: 替换为真实控制
        """
        if self._mock:
            logger.debug(f"执行动作: {action}")
            return True

        # 真实实现示例:
        # # 安全检查：关节角度限位
        # if not self._check_joint_limits(action[:7]):
        #     logger.error("关节角度超限，停止执行！")
        #     return False
        #
        # # 发送关节目标
        # self.robot.set_joint_angles(action[:7], speed=0.3)
        #
        # # 控制灵巧手
        # if action[7] > 0.5:
        #     self.robot.close_hand()
        # else:
        #     self.robot.open_hand()
        #
        # return True


# ─────────────────────────────────────────────────────────────────────────────
# 主推理循环
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pi0 机器人客户端")
    parser.add_argument("--host", type=str, default="192.168.1.100",
                        help="服务器IP地址")
    parser.add_argument("--port", type=int, default=8000,
                        help="服务器WebSocket端口")
    parser.add_argument("--action_horizon", type=int, default=ACTION_HORIZON,
                        help="每次执行的动作帧数（默认25）")
    parser.add_argument("--max_episodes", type=int, default=10,
                        help="最大执行轮数")
    # 相机序列号（通过 rs-enumerate-devices 查看）
    parser.add_argument("--cam_high_sn", type=str, default=None,
                        help="顶部相机序列号(D435)")
    parser.add_argument("--cam_right_wrist_sn", type=str, default=None,
                        help="右腕相机序列号(D405)")
    parser.add_argument("--cam_left_wrist_sn", type=str, default=None,
                        help="左腕相机序列号(D405)")
    args = parser.parse_args()

    # ── 初始化相机 ──────────────────────────────────────────────────────────
    cam_high = RealSenseCamera(args.cam_high_sn, name="cam_high")
    cam_right_wrist = RealSenseCamera(args.cam_right_wrist_sn, name="cam_right_wrist")
    cam_left_wrist = RealSenseCamera(args.cam_left_wrist_sn, name="cam_left_wrist")

    cam_high.start()
    cam_right_wrist.start()
    cam_left_wrist.start()

    # ── 初始化机器人 ────────────────────────────────────────────────────────
    robot = RobotInterface()

    # ── 连接推理服务 ────────────────────────────────────────────────────────
    from openpi_client import websocket_client_policy as _ws_policy
    from openpi_client import action_chunk_broker

    logger.info(f"连接推理服务: ws://{args.host}:{args.port}")
    policy = _ws_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = policy.get_server_metadata()
    logger.info(f"服务端元数据: {metadata}")

    # ActionChunkBroker: 每次推理得到50帧，执行 action_horizon 帧后再重新推理
    broker = action_chunk_broker.ActionChunkBroker(
        policy=policy,
        action_horizon=args.action_horizon,
    )

    logger.info("=" * 60)
    logger.info("推理客户端就绪，开始执行任务")
    logger.info(f"任务: {TASK_PROMPT}")
    logger.info(f"动作执行频率: {CONTROL_FREQ_HZ} Hz")
    logger.info(f"每次推理执行: {args.action_horizon} 帧")
    logger.info("=" * 60)

    # ── 主循环 ──────────────────────────────────────────────────────────────
    dt = 1.0 / CONTROL_FREQ_HZ

    for episode in range(args.max_episodes):
        logger.info(f"
=== Episode {episode + 1}/{args.max_episodes} ===")
        input("按 Enter 开始本轮任务（先确认机器人在安全位置）...")

        step = 0
        while True:
            t_start = time.time()

            # 1. 采集观测（JPEG压缩减少传输量：450KB→40KB，WiFi延迟从~300ms降至~30ms）
            obs = {
                "state": robot.get_right_arm_state(),  # (8,) float32
                "images": {
                    "cam_high":        compress_image(cam_high.get_image()),
                    "cam_right_wrist": compress_image(cam_right_wrist.get_image()),
                    "cam_left_wrist":  compress_image(cam_left_wrist.get_image()),
                },
                "prompt": TASK_PROMPT,
            }

            # 2. 推理（broker 内部管理重新推理时机）
            result = broker.infer(obs)
            action = result["actions"]  # (8,) float32，绝对关节角度

            # 3. 执行动作
            ok = robot.execute_action(action)
            if not ok:
                logger.error("动作执行失败，终止本轮")
                break

            step += 1
            logger.debug(f"step={step} action={action}")

            # 4. 控制频率
            elapsed = time.time() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.warning(f"控制循环超时: {elapsed*1000:.0f}ms > {dt*1000:.0f}ms")

            # 5. 终止条件（根据任务逻辑修改）
            if step >= 200:  # 最多执行 200步 = 8秒
                logger.info("达到最大步数，本轮结束")
                break

        logger.info(f"Episode {episode + 1} 完成，共执行 {step} 步")

    # ── 清理 ────────────────────────────────────────────────────────────────
    cam_high.stop()
    cam_right_wrist.stop()
    cam_left_wrist.stop()
    logger.info("客户端退出")


if __name__ == "__main__":
    main()
