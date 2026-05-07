# -*- coding: utf-8 -*-
"""
robot_executor.py — 机器人端执行器（实时观测版）

功能：
  - 三线程架构: 观测发送/动作接收/执行
  - 观测发送线程: 30Hz采集摄像头+关节数据，主动发送到4090端
  - 动作接收线程: 接收4090端推理结果
  - 执行线程: 执行接收到的动作

运行（机器人端）:
  cd /path/to/robot
  pip install -r requirements.txt
  python robot_executor.py --inference-host 192.168.1.101 --inference-port 9000

使用说明:
  1. 先启动4090端推理脚本
  2. 再启动此脚本（机器人端主动连接4090）
  3. 机器人端持续发送30Hz观测数据
  4. 4090端按键触发推理后，机器人执行动作
"""
import argparse
import base64
import json
import logging
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import cv2
import numpy as np

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent.absolute()
LOG_FILE = SCRIPT_DIR / "robot_executor.log"

# 配置日志：同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
    ],
)
logger = logging.getLogger(__name__)

# 抑制第三方库的冗余日志
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# ============================================================
# 工具函数
# ============================================================
def send_msg(sock: socket.socket, obj: dict) -> None:
    """发送JSON消息: [4字节长度(big-endian)][JSON字节串]"""
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    header = struct.pack(">I", len(payload))
    sock.sendall(header + payload)

def recv_msg(sock: socket.socket, timeout: float = 10.0) -> Optional[dict]:
    """接收JSON消息"""
    sock.settimeout(timeout)
    try:
        # 先读取4字节长度头
        header = _recv_exact(sock, 4)
        if header is None:
            return None
        length = struct.unpack(">I", header)[0]

        if length > 10 * 1024 * 1024:  # 超过10MB，认为是非法数据
            logger.error(f"[RECV] 非法消息长度: {length}")
            return None

        # 读取payload
        payload = _recv_exact(sock, length)
        if payload is None:
            return None

        msg = json.loads(payload.decode("utf-8"))
        return msg
    except json.JSONDecodeError as e:
        logger.error(f"[RECV] JSON解析失败: {e}")
        return None
    except Exception as e:
        logger.warning(f"[RECV] 接收消息异常: {e}")
        return None

def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """接收指定字节数"""
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        except socket.timeout:
            # 没有数据，连接正常，继续
            continue
        except Exception as e:
            logger.error(f"[RECV] _recv_exact 失败: {e}")
            return None
    return buf

def encode_image(image, quality: int = 85) -> Optional[str]:
    """将numpy图像编码为base64 JPEG"""
    if image is None:
        return None
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")

def format_joints(joints, precision=1):
    """格式化关节角度为短字符串"""
    if joints is None:
        return "None"
    return "[" + ", ".join(f"{x:.1f}" for x in joints[:3]) + ", ...]"

def format_images_info(images):
    """格式化图像状态信息"""
    if not images:
        return "empty"
    parts = []
    for k, v in images.items():
        if v is not None:
            parts.append(f"{k}=OK({len(v)//1024}KB)")
        else:
            parts.append(f"{k}=FAIL")
    return ", ".join(parts)

# ============================================================
# 线程安全的数据缓冲区
# ============================================================
class DataBuffer:
    """线程安全的数据缓冲区（FIFO队列）"""
    def __init__(self, name: str = "buffer", max_size: int = 100, latest_only: bool = False):
        self._queue = []  # 使用队列存储多个数据
        self._lock = threading.Lock()
        self._name = name
        self._put_count = 0
        self._get_count = 0
        self._max_size = max_size
        self._latest_only = latest_only

    def put(self, data) -> bool:
        """存入数据，返回是否成功"""
        with self._lock:
            if self._latest_only:
                # Strict latest-only: new action replaces all pending stale actions.
                self._queue = [data]
            else:
                self._queue.append(data)
            # 如果队列太长，移除最旧的数据
            while len(self._queue) > self._max_size:
                self._queue.pop(0)
            self._put_count += 1
            return True

    def get(self):
        """取出最早的数据，返回(data, timestamp_ns)"""
        with self._lock:
            if not self._queue:
                self._get_count += 1
                return None, 0
            if self._latest_only:
                data = self._queue[-1]
                self._queue.clear()
            else:
                data = self._queue.pop(0)
            self._get_count += 1
            return data, time.time_ns()

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def size(self) -> int:
        """返回队列中的数据数量"""
        with self._lock:
            return len(self._queue)

    def get_stats(self):
        """获取缓冲区统计信息"""
        with self._lock:
            return {
                "name": self._name,
                "size": len(self._queue),
                "put_count": self._put_count,
                "get_count": self._get_count,
            }
        with self._lock:
            age_ms = 0
            if self._data is not None and self._timestamp_ns > 0:
                age_ms = (time.time_ns() - self._timestamp_ns) / 1_000_000
            return {
                "name": self._name,
                "has_data": self._data is not None,
                "age_ms": age_ms,
                "put_count": self._put_count,
                "get_count": self._get_count,
            }

# ============================================================
# 机械臂客户端
# ============================================================
class ArmClient:
    """机械臂客户端 - TCP控制 + UDP状态监听"""
    def __init__(self, key: str, ip: str, udp_port: int):
        self.key = key
        self.ip = ip
        self.udp_port = udp_port
        self._lock = threading.Lock()
        self._tcp = None
        self._udp = None
        self._running = False
        self._thread = None
        self._buf = ""
        self._joints_deg: Optional[List[float]] = None
        self._gripper: Optional[float] = None
        self._gripper_online = False
        self._dexhand_dof: int = 6  # 灵巧手自由度，默认6

    def _recv_json(self, timeout: float) -> Optional[dict]:
        self._tcp.settimeout(timeout)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                chunk = self._tcp.recv(4096).decode("utf-8", errors="ignore")
            except socket.timeout:
                continue
            if not chunk:
                continue
            self._buf += chunk
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip().rstrip("\r")
                if not line:
                    continue
                try:
                    return json.loads(line)
                except Exception:
                    pass
        return None

    def _send(self, cmd: dict, timeout: float = 3.0) -> Optional[dict]:
        self._tcp.send((json.dumps(cmd) + "\r\n").encode("utf-8"))
        return self._recv_json(timeout)

    def _udp_loop(self):
        while self._running:
            try:
                data, _ = self._udp.recvfrom(65535)
                msg = json.loads(data.decode("utf-8", errors="ignore"))
                joints_raw = msg["joint_status"]["joint_position"]
                joints_deg = [x / 1000.0 for x in joints_raw]
                grip = None
                online = False
                dof = 6
                rp = msg.get("rm_plus_state")
                if isinstance(rp, dict) and rp.get("sys_state") != "offline":
                    online = True
                    pos = rp.get("pos", [])
                    if isinstance(pos, list) and len(pos) > 0:
                        grip = float(pos[0])
                        dof = len(pos)
                with self._lock:
                    self._joints_deg = joints_deg
                    self._gripper_online = online
                    self._dexhand_dof = dof
                    if grip is not None:
                        self._gripper = grip
            except socket.timeout:
                continue
            except Exception:
                continue

    def connect(self, local_ip: str, arm_tcp_port: int, wait_s: float = 6.0):
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._udp.settimeout(1.0)
        self._udp.bind((local_ip, self.udp_port))
        self._running = True
        self._thread = threading.Thread(target=self._udp_loop, daemon=True)
        self._thread.start()
        self._tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._tcp.settimeout(3.0)
        self._tcp.connect((self.ip, arm_tcp_port))
        self._send({
            "command": "set_realtime_push",
            "cycle": 5,
            "enable": True,
            "port": self.udp_port,
            "force_coordinate": 2,
            "ip": local_ip,
            "custom": {
                "joint_speed": True,
                "arm_current_status": True,
                "hand": True,
                "rm_plus_base": True,
                "rm_plus_state": True,
            },
        }, timeout=2.0)
        # 设置灵巧手波特率
        self._send({"command": "set_rm_plus_mode", "mode": 115200}, timeout=2.0)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.joints() is not None:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(f"[{self.key}] 超时：未收到关节UDP数据")

    def disconnect(self):
        self._running = False
        for s in (self._udp, self._tcp):
            try:
                if s:
                    s.close()
            except Exception:
                pass

    def joints(self) -> Optional[List[float]]:
        with self._lock:
            return list(self._joints_deg) if self._joints_deg else None

    def gripper(self) -> Optional[float]:
        with self._lock:
            return float(self._gripper) if self._gripper is not None else None

    def gripper_online(self) -> bool:
        with self._lock:
            return self._gripper_online

    def dexhand_dof(self) -> int:
        with self._lock:
            return self._dexhand_dof

    def movej_nowait(self, joints_deg: List[float], repeats: int = 3) -> None:
        """发送 movej_canfd 命令，可选择重复发送多次以确保命令被接收"""
        cmd = json.dumps({
            "command": "movej_canfd",
            "joint": [int(round(x * 1000.0)) for x in joints_deg],
            "follow": True,
            "expand": 0,
            "trajectory_mode": 0,
            "radio": 0,
        }) + "\r\n"
        encoded_cmd = cmd.encode("utf-8")
        
        def _fire():
            try:
                # 连续发送多次，确保命令被机械臂接收
                for i in range(repeats):
                    self._tcp.send(encoded_cmd)
                    # 发送间隔极短（微秒级），连续下发
                    if i < repeats - 1:
                        time.sleep(0.002)  # 2ms 间隔
            except Exception:
                pass
        threading.Thread(target=_fire, daemon=True).start()

    def set_gripper_nowait(self, pos: int) -> None:
        dof = self.dexhand_dof()
        # 灵巧手控制逻辑：
        # ch0 (拇指) = 0 → 张开
        # ch1-4 (食指~小指) = pos → 闭合/张开
        # ch5 (拇指旋转) = 0 → 张开
        hand_pos = [0] * dof
        if dof >= 5:
            hand_pos[0] = 0  # 拇指始终张开
            for i in range(1, min(dof - 1, 5)):  # ch1-ch4
                hand_pos[i] = int(max(0, min(65535, pos)))
            hand_pos[min(dof - 1, 5)] = 0  # 拇指旋转始终张开
        cmd = json.dumps({
            "command": "hand_follow_pos",
            "hand_pos": hand_pos,
        }) + "\r\n"
        def _fire():
            try:
                self._tcp.send(cmd.encode("utf-8"))
            except Exception:
                pass
        threading.Thread(target=_fire, daemon=True).start()

# ============================================================
# 机械臂推理周期插补下发控制器
# ============================================================
class InterpolatedArmController:
    """
    机械臂推理周期插补下发控制器（睿尔曼 movej_canfd 专用）

    核心设计：
    1. 每次推理（约120ms）调用 set_target() 设置新目标
    2. 在两次推理之间（约120ms内），周期性下发若干插补点
    3. 每次推理都会获取当前实际位置，重新计算插补路径
    4. 插补完成后等待下一次推理

    逻辑：
        arm = ArmClient("right", "192.168.1.10", 12345)
        arm.connect(...)
        interp = InterpolatedArmController(arm,
                                           interpolation_period_ms=15.0,
                                           points_per_inference=8)
        interp.start()

        # 主循环（每120ms推理后调用一次）：
        interp.set_target(target_joints)
        time.sleep(0.12)  # 等待推理完成

        interp.stop()
    """

    def __init__(self,
                 arm_client: ArmClient,
                 interpolation_period_ms: float = 15.0,
                 points_per_inference: int = 8):
        """
        Args:
            arm_client: 机械臂客户端
            interpolation_period_ms: 插补点下发周期（毫秒）
            points_per_inference: 每个推理周期内的插补点数量
        """
        self._arm = arm_client
        self._interpolation_period_ms = interpolation_period_ms
        self._period_s = interpolation_period_ms / 1000.0
        self._points_per_inference = points_per_inference

        self._lock = threading.Lock()

        # 轨迹段信息（每次推理时更新）
        self._seg_start: Optional[List[float]] = None  # 当前段的起点角度
        self._seg_target: Optional[List[float]] = None  # 当前段的目标角度
        self._seg_point_idx: int = 0  # 当前段的插补点索引
        self._seg_total_points: int = 0  # 当前段的总插补点数

        # 下发状态
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._next_send_time: float = 0

        # 统计
        self._send_count = 0
        self._new_target_flag = False

    def start(self) -> None:
        """启动插补下发线程"""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._next_send_time = time.time()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("[InterpolatedArm] 插补控制器已启动 "
                        f"(周期={self._interpolation_period_ms}ms, "
                        f"每推理周期={self._points_per_inference}点)")

    def stop(self) -> None:
        """停止插补下发线程"""
        with self._lock:
            if not self._running:
                return
            self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info(f"[InterpolatedArm] 插补控制器已停止 (共下发{self._send_count}次)")

    def set_target(self, target_joints: List[float]) -> None:
        """
        设置新的目标角度（每次推理后调用）

        1. 获取当前实际关节角度作为插补起点
        2. 计算从当前到目标的线性插补路径
        3. 下发第一个插补点，之后由下发线程继续
        """
        with self._lock:
            # 获取当前实际关节角度作为插补起点
            current = self._arm.joints()
            if current is None:
                logger.warning("[InterpolatedArm] 关节数据不可用，直接下发目标")
                self._arm.movej_nowait(target_joints, repeats=1)
                return

            self._seg_start = list(current)
            self._seg_target = list(target_joints)
            self._seg_point_idx = 0
            self._seg_total_points = self._points_per_inference
            self._new_target_flag = True

            # 计算并下发第一个插补点（立即下发）
            interp_joints = self._compute_interp_point(0)
            self._seg_point_idx = 1

        # 在锁外发送
        self._arm.movej_nowait(interp_joints, repeats=1)
        self._send_count += 1

        # 设置下一次下发时间
        self._next_send_time = time.time() + self._period_s

        if self._send_count % 20 == 0:
            dist = max(abs(t - s) for s, t in zip[tuple[float, float]](self._seg_start, self._seg_target))
            logger.info(f"[InterpolatedArm] 已下发{self._send_count}次, 目标距离={dist:.2f}°")

    def _compute_interp_point(self, point_idx: int) -> List[float]:
        """计算第 point_idx 个插补点（线性插值）"""
        if self._seg_start is None or self._seg_target is None:
            return self._seg_target if self._seg_target else []

        # 线性插值：start + (target - start) * (idx / total)
        alpha = point_idx / self._seg_total_points if self._seg_total_points > 0 else 1.0
        alpha = min(max(alpha, 0.0), 1.0)

        return [
            s + (g - s) * alpha
            for s, g in zip(self._seg_start, self._seg_target)
        ]

    def _run_loop(self) -> None:
        """插补下发主循环 - 在推理周期内下发剩余插补点"""
        period_s = self._period_s

        while self._running:
            now = time.time()
            sleep_time = self._next_send_time - now
            if sleep_time > 0:
                time.sleep(sleep_time)

            # 下发下一个插补点
            self._send_next_interp_point()

            # 下一个下发时间
            self._next_send_time += period_s
            # 如果落后太多，跳到下一个周期
            if time.time() - self._next_send_time > period_s:
                self._next_send_time = time.time() + period_s

    def _send_next_interp_point(self) -> None:
        """下发下一个插补点"""
        with self._lock:
            # 如果没有新目标，不下发
            if not self._new_target_flag:
                return
            if self._seg_target is None:
                return

            # 如果本轮已发完，等待下一轮 set_target()
            if self._seg_point_idx >= self._seg_total_points:
                return

            # 计算并下发插补点
            interp_joints = self._compute_interp_point(self._seg_point_idx)
            self._seg_point_idx += 1
            self._send_count += 1

        # 在锁外发送
        self._arm.movej_nowait(interp_joints, repeats=1)

        if self._send_count % 50 == 0:
            dist = max(abs(t - s) for s, t in zip(self._seg_start, self._seg_target))
            logger.info(f"[InterpolatedArm] 已下发{self._send_count}次, "
                       f"本轮={self._seg_point_idx}/{self._seg_total_points}, 距离={dist:.2f}°")


# ============================================================
# 相机管理
# ============================================================
# 机器人相机配置（硬编码）
CAMERA_CONFIG = {
    # 相机SN映射
    "camera_sn": {
        "left_arm": "353322271325",    # 左臂 D405
        "right_arm": "353322271272",   # 右臂 D405
        "head": "346222070837",        # 头部 D435
    },
    # 默认相机参数
    "resolution": {"width": 640, "height": 480},
    "fps": 30,
}


class CameraRig:
    """四路RealSense相机"""
    def __init__(self, config_json: str = None):
        self.config_json = config_json
        self.cameras = []
        self._rs = None

    def start(self) -> int:
        import pyrealsense2 as rs
        self._rs = rs

        # 使用硬编码的相机配置
        role_sn = CAMERA_CONFIG["camera_sn"]
        width = CAMERA_CONFIG["resolution"]["width"]
        height = CAMERA_CONFIG["resolution"]["height"]
        fps = CAMERA_CONFIG["fps"]

        logger.info(f"[Camera] 分辨率: {width}x{height}, FPS: {fps}")
        logger.info(f"[Camera] 相机SN映射: {role_sn}")

        # 获取当前连接的设备
        ctx = rs.context()
        devices = []
        for d in ctx.query_devices():
            try:
                sn = d.get_info(rs.camera_info.serial_number)
                name = d.get_info(rs.camera_info.name)
                devices.append({"sn": sn, "name": name})
                logger.info(f"[Camera] 发现设备: SN={sn}, Name={name}")
            except Exception:
                continue

        # 启动配置的相机
        visible = {x["sn"] for x in devices}
        used = set()

        for role, sn in role_sn.items():
            if sn not in visible:
                logger.warning(f"[Camera] 角色 {role} 的相机 SN={sn} 未连接，跳过")
                continue

            if sn in used:
                continue

            try:
                pipeline = rs.pipeline()
                rs_cfg = rs.config()
                rs_cfg.enable_device(sn)
                rs_cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                pipeline.start(rs_cfg)
                self.cameras.append({"name": role, "sn": sn, "pipeline": pipeline})
                used.add(sn)
                logger.info(f"[Camera] 相机启动成功: {role} (SN={sn})")
            except Exception as e:
                logger.error(f"[Camera] 相机启动失败: {role} (SN={sn}): {e}")

        logger.info(f"[Camera] 共启动 {len(self.cameras)}/{len(role_sn)} 个相机")
        return fps

    def capture_once(self) -> Dict[str, Optional[np.ndarray]]:
        images = {}
        lock = threading.Lock()
        def _grab(cam):
            try:
                fs = cam["pipeline"].wait_for_frames(timeout_ms=1200)
                color = fs.get_color_frame()
                with lock:
                    images[cam["name"]] = np.asanyarray(color.get_data()) if color else None
            except Exception:
                with lock:
                    images[cam["name"]] = None
        threads = [threading.Thread(target=_grab, args=(c,), daemon=True) for c in self.cameras]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=1.5)
        for cam in self.cameras:
            if cam["name"] not in images:
                images[cam["name"]] = None
        return images

    def stop(self):
        for cam in self.cameras:
            try:
                cam["pipeline"].stop()
            except Exception:
                pass

# ============================================================
# 机器人执行器 - 三线程架构（实时观测版）
# ============================================================
class RobotExecutor:
    """机器人端执行器 - 三线程架构（实时观测版）

    线程分工：
    1. OBS_SENDER: 30Hz采集观测数据，发送到4090端
    2. ACTION_RECV: 接收4090端推理结果
    3. EXEC: 执行动作
    """
    def __init__(
        self,
        inference_host: str = "192.168.1.101",
        inference_port: int = 9000,
        local_ip: str = "192.168.1.100",
        arm_ips: Dict[str, str] = None,
        arm_tcp_port: int = 8080,
        arm_udp_ports: Dict[str, int] = None,
        control_hz: float = 30.0,
        max_step_deg: float = 10.0,
        jpeg_quality: int = 85,
        connect_delay: int = 60,
        robot_port: int = 9000,
        send_obs_hz: float = 30.0,
    ):
        self.inference_host = inference_host
        self.inference_port = inference_port
        self.local_ip = local_ip
        self.arm_ips = arm_ips or {"left": "192.168.1.18", "right": "192.168.1.19"}
        self.arm_tcp_port = arm_tcp_port
        self.arm_udp_ports = arm_udp_ports or {"left": 18088, "right": 18089}
        self.control_hz = control_hz
        self.max_step_deg = max_step_deg
        self.jpeg_quality = jpeg_quality
        self.connect_delay = connect_delay
        self.robot_port = robot_port
        self.send_obs_hz = send_obs_hz  # 观测发送频率

        # 硬件客户端
        self._clients: Dict[str, ArmClient] = {}
        self._rig: Optional[CameraRig] = None
        self._sock: Optional[socket.socket] = None  # TCP连接到4090端
        self._inference_sock: Optional[socket.socket] = None  # 备用

        # 插补控制器（周期性下发插补点）
        self._interp_controllers: Dict[str, InterpolatedArmController] = {}
        self._interp_period_ms = 10.0  # 插补下发周期（毫秒）
        self._interp_points = 10  # 每轮推理下发10个点 (10ms * 10 = 100ms)
        self._use_interpolation = True  # 是否启用插补下发

        # 序列号
        self._obs_seq = 0
        self._action_seq = 0
        self._exec_step = 0

        # 线程安全缓冲区
        self._obs_buffer = DataBuffer("obs")
        self._action_buffer = DataBuffer("action", latest_only=True)

        # 状态
        self._running = False
        self._threads = []

        # 统计
        self._send_count = 0
        self._recv_count = 0
        self._exec_count = 0
        self._obs_send_count = 0  # 观测发送计数

    def connect_hardware(self):
        """连接机械臂、相机和推理服务器"""
        logger.info("=" * 60)
        logger.info("连接硬件设备...")
        logger.info("=" * 60)

        for key in ["left", "right"]:
            c = ArmClient(key, self.arm_ips[key], self.arm_udp_ports[key])
            logger.info(f"[CONNECT] 连接 {key}臂 ({self.arm_ips[key]})...")
            c.connect(self.local_ip, self.arm_tcp_port)
            self._clients[key] = c
            logger.info(f"[CONNECT] {key}臂连接成功")

        # 初始化相机（使用硬编码配置）
        self._rig = CameraRig()
        fps = self._rig.start()
        logger.info(f"[CONNECT] 相机启动完成, FPS={fps}")
        self.control_hz = float(fps)

        logger.info("[CONNECT] 等待夹爪初始化...")
        time.sleep(2.0)

        for key, c in self._clients.items():
            joints = c.joints()
            gripper = c.gripper()
            online = c.gripper_online()
            logger.info(f"[CONNECT] [{key}] 关节={format_joints(joints)}, 夹爪online={online}, gripper={gripper}")

        # 启动插补控制器（周期性下发插补点）
        if self._use_interpolation:
            logger.info(f"[CONNECT] 启动插补控制器 (周期={self._interp_period_ms}ms, 每轮{self._interp_points}点)...")
            for key, c in self._clients.items():
                ctrl = InterpolatedArmController(
                    arm_client=c,
                    interpolation_period_ms=self._interp_period_ms,
                    points_per_inference=self._interp_points,
                )
                ctrl.start()
                self._interp_controllers[key] = ctrl
            logger.info("[CONNECT] 插补控制器已启动")

        # 连接推理服务器
        logger.info(f"[CONNECT] 连接推理服务器 {self.inference_host}:{self.inference_port}...")
        self._connect_to_inference_server()
        logger.info("[CONNECT] 推理服务器连接成功")

    def _connect_to_inference_server(self):
        """连接到4090推理服务器（带重试机制）"""
        retry_count = 0
        start_time = time.time()

        while True:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.settimeout(5.0)
                self._sock.connect((self.inference_host, self.inference_port))
                logger.info(f"[CONNECT] 推理服务器连接成功! (重试次数: {retry_count})")
                return
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                retry_count += 1
                elapsed = time.time() - start_time
                if elapsed >= self.connect_delay:
                    logger.error(f"[CONNECT] 推理服务器连接超时 ({self.connect_delay}秒): {e}")
                    raise TimeoutError(f"无法连接到推理服务器 {self.inference_host}:{self.inference_port}")
                logger.warning(f"[CONNECT] 推理服务器连接失败 (已等待 {elapsed:.0f}s/{self.connect_delay}s)，重试中... (#{retry_count})")
                time.sleep(1.0)

    def _start_interp_controllers(self):
        """启动所有插补控制器"""
        if self._use_interpolation:
            for key, ctrl in self._interp_controllers.items():
                ctrl.start()
            logger.info("[INTERP] 所有插补控制器已启动")

    def _stop_interp_controllers(self):
        """停止所有插补控制器"""
        for key, ctrl in self._interp_controllers.items():
            ctrl.stop()
        logger.info("[INTERP] 所有插补控制器已停止")

    def _capture_and_put_obs(self):
        """采集观测数据并存入缓冲区"""
        right = self._clients["right"]
        left = self._clients["left"]

        joints_r = right.joints()
        joints_l = left.joints()
        if joints_r is None or joints_l is None:
            return None

        images = {}
        if self._rig:
            images = self._rig.capture_once()
        else:
            images = {"head": None, "right_arm": None, "left_arm": None}

        rp = right.gripper()
        lp = left.gripper()
        dexhand_right = 1.0 if (rp is not None and rp >= 32767.5) else 0.0
        dexhand_left = 1.0 if (lp is not None and lp >= 32767.5) else 0.0

        self._obs_seq += 1
        obs = {
            "msg_type": "observation",
            "obs_seq": self._obs_seq,
            "ts_ns": int(time.time_ns()),
            "joints_right": joints_r,
            "joints_left": joints_l,
            "dexhand_right": float(dexhand_right),
            "dexhand_left": float(dexhand_left),
            "images": {
                "head": encode_image(images.get("head"), self.jpeg_quality),
                "right_arm": encode_image(images.get("right_arm"), self.jpeg_quality),
                "left_arm": encode_image(images.get("left_arm"), self.jpeg_quality),
            },
        }

        # 存入缓冲区
        self._obs_buffer.put(obs)
        return obs

    def _execute_action(self, action: dict) -> dict:
        """执行动作并返回执行结果（插补模式）"""
        right = self._clients["right"]
        left = self._clients["left"]
        exec_result = {
            "joints_r_sent": None,
            "joints_l_sent": None,
            "gripper_r_sent": None,
            "gripper_l_sent": None,
        }

        # 控制右臂（必须）
        current_r = right.joints()
        if current_r is not None:
            target_joints_r = action.get("joints_right")
            if target_joints_r is not None and len(target_joints_r) >= 7:
                safe_target_r = []
                for cur, tgt in zip(current_r, target_joints_r[:7]):
                    safe = max(cur - self.max_step_deg, min(cur + self.max_step_deg, tgt))
                    safe_target_r.append(safe)

                if self._use_interpolation and "right" in self._interp_controllers:
                    self._interp_controllers["right"].set_target(safe_target_r)
                    exec_result["joints_r_sent"] = safe_target_r
                else:
                    right.movej_nowait(safe_target_r)
                    exec_result["joints_r_sent"] = safe_target_r

        # 控制左臂（可选，如果消息中包含左臂数据才执行）
        target_joints_l = action.get("joints_left")
        if target_joints_l is not None and len(target_joints_l) >= 7:
            current_l = left.joints()
            if current_l is not None:
                safe_target_l = []
                for cur, tgt in zip(current_l, target_joints_l[:7]):
                    safe = max(cur - self.max_step_deg, min(cur + self.max_step_deg, tgt))
                    safe_target_l.append(safe)

                if self._use_interpolation and "left" in self._interp_controllers:
                    self._interp_controllers["left"].set_target(safe_target_l)
                    exec_result["joints_l_sent"] = safe_target_l
                else:
                    left.movej_nowait(safe_target_l)
                    exec_result["joints_l_sent"] = safe_target_l

        # 控制右灵巧手（必须）
        dexhand_r = action.get("dexhand_right")
        if dexhand_r is not None and right.gripper_online():
            grip_raw = 65535 if float(dexhand_r) >= 0.5 else 0
            right.set_gripper_nowait(grip_raw)
            exec_result["gripper_r_sent"] = grip_raw

        # 控制左灵巧手（可选，如果消息中包含左臂数据才执行）
        dexhand_l = action.get("dexhand_left")
        if dexhand_l is not None and left.gripper_online():
            grip_raw = 65535 if float(dexhand_l) >= 0.5 else 0
            left.set_gripper_nowait(grip_raw)
            exec_result["gripper_l_sent"] = grip_raw

        return exec_result

    def print_received_action(self, msg: dict):
        """详细打印接收到的动作"""
        obs_seq = msg.get('obs_seq', 0)
        infer_ms = msg.get('infer_ms', 0)
        joints_right = msg.get('joints_right', [])
        joints_left = msg.get('joints_left', [])  # 可能为空
        dexhand_right = msg.get('dexhand_right', 0)
        dexhand_left = msg.get('dexhand_left', None)  # 可能为空

        print(f"\n{'='*70}")
        print(f"【接收数据】(来自 4090 端)")
        print(f"{'='*70}")
        print(f"  帧号: {obs_seq}")
        print(f"  推理耗时: {infer_ms:.1f}ms")
        print()
        print(f"【右臂动作】")
        print(f"  关节(度): {[f'{x:.3f}' for x in joints_right] if joints_right else '无'}")
        print(f"  灵巧手: {dexhand_right:.3f} ({'闭合' if dexhand_right >= 0.5 else '张开'})")
        print()
        print(f"【左臂动作】")
        if joints_left:
            print(f"  关节(度): {[f'{x:.3f}' for x in joints_left]}")
            if dexhand_left is not None:
                print(f"  灵巧手: {dexhand_left:.3f} ({'闭合' if dexhand_left >= 0.5 else '张开'})")
        else:
            print(f"  关节(度): 无 (保持原位)")
            print(f"  灵巧手: 无 (保持原位)")
        print(f"{'='*70}")

    def print_execution_result(self, action: dict, exec_result: dict):
        """详细打印执行结果"""
        right = self._clients["right"]
        left = self._clients["left"]

        joints_r_current = right.joints()
        joints_l_current = left.joints()
        dexhand_r_current = right.gripper()
        dexhand_l_current = left.gripper()

        print(f"\n{'='*70}")
        print(f"【执行状态】")
        print(f"{'='*70}")

        if joints_r_current:
            print(f"  右臂当前关节(度): {[f'{x:.3f}' for x in joints_r_current]}")
        if dexhand_r_current is not None:
            print(f"  右灵巧手当前: {dexhand_r_current:.3f}")

        if joints_l_current:
            print(f"  左臂当前关节(度): {[f'{x:.3f}' for x in joints_l_current]}")
        if dexhand_l_current is not None:
            print(f"  左灵巧手当前: {dexhand_l_current:.3f}")

        print()
        print(f"【下发动作】")
        if exec_result.get('joints_r_sent'):
            print(f"  右臂已发送(度): {[f'{x:.3f}' for x in exec_result['joints_r_sent']]}")
        if exec_result.get('joints_l_sent'):
            print(f"  左臂已发送(度): {[f'{x:.3f}' for x in exec_result['joints_l_sent']]}")
        else:
            print(f"  左臂已发送(度): 无 (保持原位)")
        if exec_result.get('gripper_r_sent') is not None:
            print(f"  右灵巧手已发送: {exec_result['gripper_r_sent']} ({'闭合' if exec_result['gripper_r_sent'] else '张开'})")
        if exec_result.get('gripper_l_sent') is not None:
            print(f"  左灵巧手已发送: {exec_result['gripper_l_sent']} ({'闭合' if exec_result['gripper_l_sent'] else '张开'})")

        print()
        print(f"【统计】")
        print(f"  执行计数: {self._exec_count}")
        print(f"{'='*70}")

    # ============================================================
    # 线程1: 观测发送线程 - 30Hz采集并发送观测数据到4090端
    # ============================================================
    def _obs_sender_thread_worker(self):
        """观测发送线程: 30Hz采集摄像头+关节数据，发送到4090端"""
        logger.info(f"[OBS_SENDER] 观测发送线程启动 (目标频率: {self.send_obs_hz}Hz)")
        period_s = 1.0 / self.send_obs_hz

        while self._running:
            try:
                # 采集关节数据
                right = self._clients["right"]
                left = self._clients["left"]
                joints_r = right.joints()
                joints_l = left.joints()

                if joints_r is None or joints_l is None:
                    logger.warning("[OBS_SENDER] 关节数据不可用，等待...")
                    time.sleep(period_s)
                    continue

                # 采集相机图像
                images = {}
                if self._rig:
                    images = self._rig.capture_once()
                else:
                    images = {"head": None, "right_arm": None, "left_arm": None}

                # 获取灵巧手状态
                rp = right.gripper()
                lp = left.gripper()
                dexhand_right = 1.0 if (rp is not None and rp >= 32767.5) else 0.0
                dexhand_left = 1.0 if (lp is not None and lp >= 32767.5) else 0.0

                self._obs_seq += 1
                obs_msg = {
                    "msg_type": "observation",
                    "obs_seq": self._obs_seq,
                    "ts_ns": int(time.time_ns()),
                    "joints_right": joints_r,
                    "joints_left": joints_l,
                    "dexhand_right": float(dexhand_right),
                    "dexhand_left": float(dexhand_left),
                    "images": {
                        "head": encode_image(images.get("head"), self.jpeg_quality),
                        "right_arm": encode_image(images.get("right_arm"), self.jpeg_quality),
                        "left_arm": encode_image(images.get("left_arm"), self.jpeg_quality),
                    },
                }

                # 发送到4090端
                send_msg(self._sock, obs_msg)
                self._obs_send_count += 1

                if self._obs_send_count % 100 == 0:
                    logger.info(f"[OBS_SENDER] 已发送 {self._obs_send_count} 帧观测数据")

                time.sleep(period_s)

            except Exception as e:
                logger.error(f"[OBS_SENDER] 采集/发送观测失败: {e}")
                time.sleep(0.1)

    # ============================================================
    # 线程2: 动作接收线程 - 接收4090端推理结果
    # ============================================================
    def _recv_thread_worker(self):
        """动作接收线程: 通过已建立的TCP连接接收4090端推理结果"""
        logger.info("[ACTION_RECV] 动作接收线程启动")
        
        # 等待连接建立
        while self._running and self._sock is None:
            time.sleep(0.1)

        # 循环接收动作
        while self._running:
            if self._sock is None:
                time.sleep(0.1)
                continue

            try:
                msg = recv_msg(self._sock, timeout=2.0)
                if msg is None:
                    continue

                if msg.get("msg_type") == "action":
                    self._action_buffer.put(msg)
                    self._recv_count += 1
                    logger.info(f"[ACTION_RECV] 接收动作 #{self._recv_count}, obs_seq={msg.get('obs_seq', 0)}")

            except Exception as e:
                logger.error(f"[ACTION_RECV] 接收动作异常: {e}")
                time.sleep(0.05)

    # ============================================================
    # 线程3: 执行线程 - 从action缓冲区读取并执行
    # ============================================================
    def _exec_thread_worker(self):
        """执行线程: 从action缓冲区读取，执行动作"""
        logger.info("[EXEC] 执行线程启动")
        while self._running:
            # 从缓冲区获取最新动作
            action, ts_ns = self._action_buffer.get()
            if action is None:
                time.sleep(0.005)
                continue

            # 执行动作
            exec_result = self._execute_action(action)
            self._exec_count += 1
            
            # 打印执行结果
            if self._exec_count % 10 == 0:
                logger.info(f"[EXEC] 执行 #{self._exec_count}: joints_r_sent={exec_result.get('joints_r_sent') is not None}")

    # ============================================================
    # 主循环 - 三线程架构
    # ============================================================
    def run_loop(self):
        """主循环 - 启动三线程架构"""
        logger.info("=" * 60)
        logger.info("【机器人执行器 - 实时观测版】")
        logger.info(f"推理服务器: {self.inference_host}:{self.inference_port}")
        logger.info(f"本地IP: {self.local_ip}")
        logger.info(f"左臂: {self.arm_ips['left']}, 右臂: {self.arm_ips['right']}")
        logger.info(f"观测发送频率: {self.send_obs_hz}Hz")
        logger.info(f"单步限幅: ±{self.max_step_deg}°")
        logger.info("=" * 60)

        self._running = True

        # 启动三线程（观测发送/动作接收/执行）
        threads_info = [
            (self._obs_sender_thread_worker, "OBS_SENDER"),
            (self._recv_thread_worker, "ACTION_RECV"),
            (self._exec_thread_worker, "EXEC"),
        ]

        for worker, name in threads_info:
            t = threading.Thread(target=worker, name=name, daemon=True)
            t.start()
            self._threads.append(t)
            logger.info(f"[START] {name} 线程已启动")

        # 启动插补控制器
        if self._use_interpolation:
            self._start_interp_controllers()

        # 无限期等待
        logger.info("[MAIN] 所有线程已启动，等待运行...")
        try:
            for t in self._threads:
                t.join()
        except KeyboardInterrupt:
            self._running = False
        finally:
            self._running = False
            self._cleanup()
            logger.info(f"[MAIN] 执行器已停止: 观测发送={self._obs_send_count}, 动作接收={self._recv_count}, 执行={self._exec_count}")

    def _cleanup(self):
        """清理资源"""
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._inference_sock:
            try:
                self._inference_sock.close()
            except Exception:
                pass
            self._inference_sock = None

    def stop(self):
        """停止执行器"""
        self._running = False
        # 停止插补控制器
        self._stop_interp_controllers()
        for c in self._clients.values():
            c.disconnect()
        if self._rig:
            self._rig.stop()
        self._cleanup()
        logger.info("[STOP] 机器人执行器已停止")

# ============================================================
# 主入口
# ============================================================
def main():
    print("=" * 60)
    print(f"机器人执行器 - 实时观测版")
    print(f"日志文件: {LOG_FILE}")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="机器人执行器 (实时观测版)")
    parser.add_argument("--inference-host", default="192.168.1.101", help="推理服务器IP")
    parser.add_argument("--inference-port", type=int, default=9000, help="推理服务器端口")
    parser.add_argument("--local-ip", default="192.168.1.100", help="本地IP")
    parser.add_argument("--arm-left-ip", default="192.168.1.18", help="左臂IP")
    parser.add_argument("--arm-right-ip", default="192.168.1.19", help="右臂IP")
    parser.add_argument("--arm-tcp-port", type=int, default=8080, help="机械臂TCP端口")
    parser.add_argument("--arm-left-udp-port", type=int, default=18088, help="左臂UDP端口")
    parser.add_argument("--arm-right-udp-port", type=int, default=18089, help="右臂UDP端口")
    parser.add_argument("--control-hz", type=float, default=30.0, help="控制频率")
    parser.add_argument("--max-step", type=float, default=10.0, help="单步限幅(度)")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG质量")
    parser.add_argument("--connect-delay", type=int, default=60, help="连接推理服务器超时(秒)")
    parser.add_argument("--robot-port", type=int, default=9000, help="机器人监听端口(保留参数)")
    parser.add_argument("--send-obs-hz", type=float, default=30.0, help="观测发送频率(Hz)")
    args = parser.parse_args()

    executor = RobotExecutor(
        inference_host=args.inference_host,
        inference_port=args.inference_port,
        local_ip=args.local_ip,
        arm_ips={"left": args.arm_left_ip, "right": args.arm_right_ip},
        arm_tcp_port=args.arm_tcp_port,
        arm_udp_ports={"left": args.arm_left_udp_port, "right": args.arm_right_udp_port},
        control_hz=args.control_hz,
        max_step_deg=args.max_step,
        jpeg_quality=args.jpeg_quality,
        connect_delay=args.connect_delay,
        robot_port=args.robot_port,
        send_obs_hz=args.send_obs_hz,
    )

    try:
        executor.connect_hardware()
        executor.run_loop()
    finally:
        executor.stop()

if __name__ == "__main__":
    main()
