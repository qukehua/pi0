#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robot_policy_bridge.py — 机器人策略执行桥接服务端

拓扑：
  PC(推理端) ←─TCP:9000─→ 本脚本(机器人端, 192.168.1.100)
                              ├─TCP:8080+UDP:8088─→ 左臂 (192.168.1.18)
                              ├─TCP:8080+UDP:8089─→ 右臂 (192.168.1.19)
                              └─USB─→ 4路RealSense

消息协议（TCP，长度前缀）：
  [4字节 uint32 big-endian 消息长度][JSON字节串]

观测包（机器人 → PC），字段：
  joints_right[7]  关节角度（度）
  joints_left[7]
  gripper_right    夹爪开口量 0~1000
  gripper_left
  images           {left_arm, right_arm, head, chest} → base64 JPEG字符串
  ts_ns            采样时间戳（纳秒）

动作包（PC → 机器人），字段：
  joints_right[7] | null    目标关节角度（度），null=跳过
  joints_left[7]  | null
  gripper_right   | null    目标夹爪位置 0~1000，null=跳过
  gripper_left    | null

用法：
  python3 robot_policy_bridge.py [--port 9000] [--hz 10]
                                  [--max-step 10.0] [--speed 5]
                                  [--jpeg-quality 85]
"""

import argparse
import base64
import json
import logging
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 网络 / 硬件配置 ────────────────────────────────────────
LOCAL_IP     = "192.168.1.100"
ARM_TCP_PORT = 8080
TCP_TIMEOUT  = 3.0
UDP_TIMEOUT  = 1.0
PUSH_CYCLE   = 5   # UDP推送周期 ms

ARMS = {
    "left":  {"ip": "192.168.1.18", "udp_port": 8088, "label": "左臂"},
    "right": {"ip": "192.168.1.19", "udp_port": 8089, "label": "右臂"},
}

CAMERA_SN_JSON = (
    "/home/data/robot/hmi-robot/Robot-terminal-system"
    "/Bin/camera_server/config/Camera_SN.json"
)


# ══════════════════════════════════════════════════════════════
# ArmClient — 双信道读写（UDP状态 + TCP控制）
# ══════════════════════════════════════════════════════════════
class ArmClient:
    def __init__(self, key: str, ip: str, udp_port: int):
        self.key      = key
        self.ip       = ip
        self.udp_port = udp_port

        self._lock    = threading.Lock()
        self._tcp     = None
        self._udp     = None
        self._running = False
        self._thread  = None
        self._buf     = ""

        self._joints_deg:     Optional[List[float]] = None
        self._gripper:        Optional[float]       = None
        self._gripper_dof:    int                   = 1
        self._gripper_online: bool                  = False
        self._frame_cnt:      int                   = 0

    # ── TCP 收发 ──────────────────────────────────────────
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
            raw = self._buf.strip()
            if raw:
                try:
                    obj = json.loads(raw)
                    self._buf = ""
                    return obj
                except Exception:
                    pass
        return None

    def _send(self, cmd: dict, timeout: float = TCP_TIMEOUT) -> Optional[dict]:
        self._tcp.send((json.dumps(cmd) + "\r\n").encode("utf-8"))
        return self._recv_json(timeout)

    # ── UDP 监听线程 ──────────────────────────────────────
    def _udp_loop(self):
        while self._running:
            try:
                data, _ = self._udp.recvfrom(65535)
                msg = json.loads(data.decode("utf-8", errors="ignore"))
                joints_raw = msg["joint_status"]["joint_position"]
                joints_deg = [x / 1000.0 for x in joints_raw]
                grip, online, dof = None, False, 1
                rp = msg.get("rm_plus_state")
                if isinstance(rp, dict):
                    online = True
                    pos = rp.get("pos", [])
                    if isinstance(pos, list):
                        dof = max(1, len(pos))
                        if len(pos) > 0:
                            grip = float(pos[0])
                with self._lock:
                    self._joints_deg     = joints_deg
                    self._frame_cnt     += 1
                    self._gripper_online = online
                    self._gripper_dof    = dof
                    if grip is not None:
                        self._gripper = grip
            except socket.timeout:
                continue
            except Exception:
                continue

    # ── 连接 / 断开 ──────────────────────────────────────
    def connect(self, wait_s: float = 6.0):
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._udp.settimeout(UDP_TIMEOUT)
        self._udp.bind((LOCAL_IP, self.udp_port))

        self._running = True
        self._thread  = threading.Thread(target=self._udp_loop, daemon=True)
        self._thread.start()

        self._tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._tcp.settimeout(TCP_TIMEOUT)
        self._tcp.connect((self.ip, ARM_TCP_PORT))

        self._send({
            "command": "set_realtime_push",
            "cycle": PUSH_CYCLE, "enable": True,
            "port": self.udp_port,
            "force_coordinate": 2, "ip": LOCAL_IP,
            "custom": {"joint_speed": True, "arm_current_status": True},
        }, timeout=2.0)
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

    # ── 状态读取 ──────────────────────────────────────────
    def joints(self) -> Optional[List[float]]:
        with self._lock:
            return list(self._joints_deg) if self._joints_deg else None

    def gripper(self) -> Optional[float]:
        with self._lock:
            return float(self._gripper) if self._gripper is not None else None

    def gripper_norm(self) -> Optional[float]:
        raw = self.gripper()
        if raw is None:
            return None
        return max(0.0, min(1.0, raw / 65535.0))

    def gripper_online(self) -> bool:
        with self._lock:
            return self._gripper_online

    def gripper_dof(self) -> int:
        with self._lock:
            return self._gripper_dof

    # ── 控制 ──────────────────────────────────────────────
    def movej(self, joints_deg: List[float], v: int = 5) -> Optional[dict]:
        """阻塞式 movej，适合预归位；实时策略控制不要用它。"""
        mdeg = [int(round(x * 1000.0)) for x in joints_deg]
        return self._send({
            "command": "movej",
            "joint":   mdeg,
            "v":       max(1, min(10, v)),
            "r":       0,
            "trajectory_connect": 0,
        }, timeout=6.0)

    def movej_nowait(self, joints_deg: List[float]) -> None:
        """非阻塞 movej_canfd；停止下发后机械臂更容易就地停住。"""
        cmd = json.dumps({
            "command": "movej_canfd",
            "joint": [int(round(x * 1000.0)) for x in joints_deg],
            "follow": True,
            "expand": 0,
            "trajectory_mode": 0,
            "radio": 0,
        }) + "\r\n"
        try:
            self._tcp.send(cmd.encode("utf-8"))
        except Exception:
            pass

    def set_gripper(self, pos: int) -> Optional[dict]:
        pos      = int(max(0, min(65535, pos)))
        dof      = self.gripper_dof()
        hand_pos = [pos] * dof
        return self._send({
            "command":  "hand_follow_pos",
            "hand_pos": hand_pos,
        }, timeout=2.0)

    def set_gripper_nowait(self, pos: int) -> None:
        pos      = int(max(0, min(65535, pos)))
        dof      = self.gripper_dof()
        cmd = json.dumps({
            "command":  "hand_follow_pos",
            "hand_pos": [pos] * dof,
        }) + "\r\n"
        try:
            self._tcp.send(cmd.encode("utf-8"))
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
# FourCameraRig — 四路 RealSense 相机
# ══════════════════════════════════════════════════════════════
class FourCameraRig:
    WANTED = ["left_arm", "right_arm", "head"]
    ROLE_NAME_ALIASES = {
        "left_arm": ["left_arm", "left_image"],
        "right_arm": ["right_arm", "right_image"],
        "head": ["head", "head_image"],
        "chest": ["chest", "chest_image"],
    }

    def __init__(self, config_json: str):
        self.config_json = config_json
        self.cameras: list = []

    def start(self) -> int:
        with open(self.config_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        settings = cfg.get("settings", {})
        width  = int(settings.get("resolution", {}).get("width",  640))
        height = int(settings.get("resolution", {}).get("height", 480))
        fps    = int(settings.get("fps", 30))

        ctx     = rs.context()
        devices = []
        for d in ctx.query_devices():
            try:
                devices.append({
                    "sn":   d.get_info(rs.camera_info.serial_number),
                    "name": d.get_info(rs.camera_info.name),
                    "port": d.get_info(rs.camera_info.physical_port),
                })
            except Exception:
                continue
        logger.info("可见RealSense设备: %s", [x["sn"] for x in devices])

        d405 = sorted(
            [x for x in devices if "D405" in x["name"]],
            key=lambda x: x["port"],
        )
        role_sn = {}
        if len(d405) >= 2:
            role_sn["left_arm"]  = d405[1]["sn"]
            role_sn["right_arm"] = d405[0]["sn"]
        role_sn["head"]  = "346222070837"
        role_sn["chest"] = "348522075096"

        visible = {x["sn"] for x in devices}
        used    = set()

        cam_list = [c for c in cfg.get("cameras", []) if c.get("enabled", True)]
        wanted_camera_names = {
            alias
            for role in self.WANTED
            for alias in self.ROLE_NAME_ALIASES.get(role, [role])
        }
        cam_list = [c for c in cam_list if c.get("name") in wanted_camera_names]

        for cam in cam_list:
            raw_role = cam["name"]
            role = next(
                (
                    canonical
                    for canonical, aliases in self.ROLE_NAME_ALIASES.items()
                    if raw_role in aliases
                ),
                raw_role,
            )
            target_sn = role_sn.get(role) or cam.get("sn")
            if target_sn not in visible:
                fallback = next((x["sn"] for x in devices if x["sn"] not in used), None)
                if fallback is None:
                    logger.warning("相机[%s] 无可用设备，跳过", role)
                    continue
                logger.warning("相机[%s] SN=%s不可见，回退SN=%s", role, target_sn, fallback)
                target_sn = fallback
            pipeline = rs.pipeline()
            rs_cfg   = rs.config()
            rs_cfg.enable_device(target_sn)
            rs_cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            pipeline.start(rs_cfg)
            self.cameras.append({"name": role, "sn": target_sn, "pipeline": pipeline})
            used.add(target_sn)
            logger.info("相机启动: %s (SN=%s)", role, target_sn)

        miss = [r for r in self.WANTED if r not in {c["name"] for c in self.cameras}]
        if miss:
            raise RuntimeError(f"相机未齐全，缺少: {miss}")
        return fps

    def capture_once(self) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, int]]:
        images: Dict[str, Optional[np.ndarray]] = {}
        ts_ns:  Dict[str, int] = {}
        lock = threading.Lock()

        def _grab(cam):
            name = cam["name"]
            try:
                fs    = cam["pipeline"].wait_for_frames(timeout_ms=1200)
                color = fs.get_color_frame()
                if not color:
                    with lock:
                        images[name] = None
                        ts_ns[name]  = 0
                    return
                with lock:
                    images[name] = np.asanyarray(color.get_data())
                    ts_ns[name]  = time.time_ns()
            except Exception:
                with lock:
                    images[name] = None
                    ts_ns[name]  = 0

        threads = [threading.Thread(target=_grab, args=(c,), daemon=True) for c in self.cameras]
        for t in threads: t.start()
        for t in threads: t.join(timeout=1.5)
        for cam in self.cameras:
            if cam["name"] not in images:
                images[cam["name"]] = None
                ts_ns[cam["name"]]  = 0
        return images, ts_ns

    def stop(self):
        for cam in self.cameras:
            try:
                cam["pipeline"].stop()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════
# 工具函数：TCP 消息收发（长度前缀 + JSON）
# ══════════════════════════════════════════════════════════════
def send_msg(sock: socket.socket, obj: dict) -> None:
    """发送一条 JSON 消息：[4字节长度(big-endian)][JSON字节串]"""
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    header  = struct.pack(">I", len(payload))
    sock.sendall(header + payload)


def recv_msg(sock: socket.socket) -> Optional[dict]:
    """接收一条 JSON 消息，超时/断开返回 None"""
    try:
        header = _recv_exact(sock, 4)
        if header is None:
            return None
        length  = struct.unpack(">I", header)[0]
        payload = _recv_exact(sock, length)
        if payload is None:
            return None
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return None


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except Exception:
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


# ══════════════════════════════════════════════════════════════
# PolicyBridgeServer — 策略执行桥接主体
# ══════════════════════════════════════════════════════════════
class PolicyBridgeServer:
    def __init__(
        self,
        clients: Dict[str, ArmClient],
        rig: FourCameraRig,
        host: str = "0.0.0.0",
        port: int = 9000,
        hz: float = 10.0,
        max_step_deg: float = 10.0,
        speed: int = 5,
        jpeg_quality: int = 85,
        control_mode: str = "canfd",
        active_joint: Optional[int] = None,
    ):
        self._clients      = clients
        self._rig          = rig
        self._host         = host
        self._port         = port
        self._period       = 1.0 / hz
        self._max_step     = max_step_deg
        self._speed        = speed
        self._jpeg_quality = jpeg_quality
        self._control_mode = control_mode
        self._active_joint = active_joint

        self._conn_lock    = threading.Lock()
        self._current_conn: Optional[socket.socket] = None
        self._step_counter = 0
        self._last_joint0: Dict[str, Optional[float]] = {"left": None, "right": None}
        self._stuck_count: Dict[str, int] = {"left": 0, "right": 0}

    # ── 图片编码 ──────────────────────────────────────────
    def _encode_image(self, img: Optional[np.ndarray]) -> Optional[str]:
        if img is None:
            return None
        ok, buf = cv2.imencode(
            ".jpg", img,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
        )
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")

    # ── 构建观测包 ────────────────────────────────────────
    def _build_obs(self, ts_ns: int) -> Optional[dict]:
        right = self._clients["right"]
        left  = self._clients["left"]

        joints_r = right.joints()
        joints_l = left.joints()
        if joints_r is None or joints_l is None:
            return None

        images, _cam_ts = self._rig.capture_once()

        # 灵巧手状态：真实 HDF5 语义已确认是 0 或 1000，归一化后就是 0 或 1。
        # 因此推理闭环里也保持二值语义：0=全打开，1=全闭合。
        _rp = right.gripper()
        _lp = left.gripper()
        dexhand_right = 1.0 if (_rp is not None and _rp >= 32767.5) else 0.0
        dexhand_left  = 1.0 if (_lp is not None and _lp >= 32767.5) else 0.0

        obs = {
            "ts_ns":          ts_ns,
            "joints_right":   joints_r,       # list[7], 单位:度
            "joints_left":    joints_l,
            "dexhand_right":  dexhand_right,  # binary float: 0=张开, 1=闭合
            "dexhand_left":   dexhand_left,
            "images": {
                role: self._encode_image(img)
                for role, img in images.items()
            },
        }
        return obs

    # ── 安全执行动作 ──────────────────────────────────────
    def _execute_action(self, action: dict):
        """
        接收动作字典，安全执行：
          - 每个关节目标不得超过当前位置 ±max_step_deg
          - null 表示该臂/夹爪本次不执行
        """
        self._step_counter += 1
        # ★ 切换为夹爪采集：把 dexhand_right/dexhand_left 改回
        #   gripper_right/gripper_left，末端块改用 set_gripper(int(val))
        for arm_key, joints_key, dexhand_key in [
            ("right", "joints_right", "dexhand_right"),
            ("left",  "joints_left",  "dexhand_left"),
        ]:
            c = self._clients[arm_key]

            # ── 关节 ──
            target_joints = action.get(joints_key)
            if target_joints is not None and len(target_joints) == 7:
                current = c.joints()
                if current is None:
                    logger.warning("[%s] 关节数据不可用，跳过本帧", arm_key)
                else:
                    # 逐关节限幅：目标裁切到 [current-max_step, current+max_step]
                    safe_target = [float(cur) for cur in current]
                    for idx, (cur, tgt) in enumerate(zip(current, target_joints)):
                        if self._active_joint is not None and idx != self._active_joint:
                            continue
                        safe_target[idx] = float(
                            max(cur - self._max_step, min(cur + self._max_step, tgt))
                        )
                    use_movej = (self._control_mode == "movej")

                    if use_movej:
                        resp = c.movej(safe_target, v=self._speed)
                        send_mode = "movej"
                    else:
                        c.movej_nowait(safe_target)
                        resp = {"ok": True, "mode": "canfd"}
                        send_mode = "canfd"

                    cur_after = c.joints()
                    cur0_after = float(cur_after[0]) if cur_after and len(cur_after) == 7 else None
                    prev0 = self._last_joint0[arm_key]
                    if prev0 is not None and cur0_after is not None:
                        if abs(cur0_after - prev0) < 0.05:
                            self._stuck_count[arm_key] += 1
                        else:
                            self._stuck_count[arm_key] = 0
                    self._last_joint0[arm_key] = cur0_after

                    if self._step_counter % 10 == 0:
                        delta = [round(st - cur, 2) for st, cur in zip(safe_target, current)]
                        logger.info(
                            "[%s] step=%d mode=%s send_mode=%s cur=%s target=%s safe=%s delta=%s resp=%s stuck_count=%d cur0_after=%s",
                            arm_key,
                            self._step_counter,
                            self._control_mode,
                            send_mode,
                            [round(x, 2) for x in current],
                            [round(float(x), 2) for x in target_joints],
                            [round(x, 2) for x in safe_target],
                            delta,
                            resp,
                            self._stuck_count[arm_key],
                            None if cur0_after is None else round(cur0_after, 3),
                        )

            # ── 灵巧手：模型输出先判成 0/1，再下发真实硬件值 0 或 65535 ──
            dexhand_target = action.get(dexhand_key)
            if dexhand_target is not None and c.gripper_online():
                grip_raw = 65535 if float(dexhand_target) >= 0.5 else 0
                if self._control_mode == "canfd":
                    c.set_gripper_nowait(grip_raw)
                    resp = {"ok": True, "mode": "canfd_nowait"}
                else:
                    resp = c.set_gripper(grip_raw)
                if self._step_counter % 10 == 0:
                    logger.info(
                        "[%s] step=%d dexhand_target=%s grip_raw=%d resp=%s",
                        arm_key,
                        self._step_counter,
                        dexhand_target,
                        grip_raw,
                        resp,
                    )

    # ── 处理单个 PC 客户端连接 ────────────────────────────
    def _handle_client(self, conn: socket.socket, addr):
        logger.info("PC客户端已连接: %s", addr)
        conn.settimeout(5.0)

        next_tick = time.monotonic()
        try:
            while True:
                # 1. 构建观测包
                ts_ns = time.time_ns()
                obs   = self._build_obs(ts_ns)
                if obs is None:
                    time.sleep(0.05)
                    continue

                # 2. 发送观测给 PC
                try:
                    send_msg(conn, obs)
                except Exception as e:
                    logger.warning("发送观测失败: %s", e)
                    break

                # 3. 接收 PC 推理结果（动作）
                action = recv_msg(conn)
                if action is None:
                    logger.warning("接收动作失败，PC端可能已断开")
                    break

                # 4. 安全执行动作
                self._execute_action(action)

                # 5. 固定频率节拍
                next_tick += self._period
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()

        except Exception as e:
            logger.error("客户端处理异常: %s", e)
        finally:
            conn.close()
            logger.info("PC客户端断开: %s", addr)

    # ── 启动服务器 ────────────────────────────────────────
    def serve_forever(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self._host, self._port))
        srv.listen(1)
        logger.info(
            "PolicyBridgeServer 已启动: %s:%d  hz=%.1f  max_step=±%.1f° mode=%s active_joint=%s",
            self._host, self._port,
            1.0 / self._period, self._max_step, self._control_mode,
            "all" if self._active_joint is None else self._active_joint,
        )

        while True:
            conn, addr = srv.accept()
            # 每次只服务一个 PC 客户端
            t = threading.Thread(
                target=self._handle_client,
                args=(conn, addr),
                daemon=True,
            )
            t.start()


def _startup_control_self_check(right_client: ArmClient):
    """开机自检：右臂做小幅往返，验证确实可控。"""
    cur = right_client.joints()
    if not cur or len(cur) != 7:
        raise RuntimeError("启动自检失败：无法读取右臂关节")

    base = [float(x) for x in cur]
    target = list(base)
    target[0] += 3.0

    logger.info("启动自检：右臂J1往返3°，验证可控性")
    right_client.movej(target, v=3)

    mid = None
    deadline = time.time() + 3.5
    while time.time() < deadline:
        mid = right_client.joints()
        if mid and abs(float(mid[0]) - base[0]) >= 0.8:
            break
        time.sleep(0.05)
    if not mid or abs(float(mid[0]) - base[0]) < 0.8:
        raise RuntimeError(
            f"启动自检失败：右臂角度无明显变化（before={base[0]:.2f}, after={float(mid[0]) if mid else float('nan'):.2f}）"
        )

    right_client.movej(base, v=3)
    end = None
    deadline = time.time() + 3.5
    while time.time() < deadline:
        end = right_client.joints()
        if end and abs(float(end[0]) - float(mid[0])) >= 0.8:
            break
        time.sleep(0.05)
    if not end or abs(float(end[0]) - float(mid[0])) < 0.8:
        raise RuntimeError(
            f"启动自检失败：右臂未能回程（mid={float(mid[0]):.2f}, end={float(end[0]) if end else float('nan'):.2f}）"
        )

    logger.info("启动自检通过：右臂可控")


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="机器人策略桥接服务端")
    parser.add_argument("--host",         default="0.0.0.0",   help="监听地址（默认0.0.0.0）")
    parser.add_argument("--port",         type=int,   default=9000,  help="监听端口（默认9000）")
    parser.add_argument("--hz",           type=float, default=10.0,  help="控制频率Hz（默认10）")
    parser.add_argument("--max-step",     type=float, default=10.0,  help="关节单步限幅°（默认10）")
    parser.add_argument("--speed",        type=int,   default=5,     help="movej速度1~10（默认5）")
    parser.add_argument("--jpeg-quality", type=int,   default=85,    help="图像JPEG质量1~100（默认85）")
    parser.add_argument("--control-mode", choices=["canfd", "movej"], default="canfd", help="关节控制模式")
    parser.add_argument("--active-joint", type=int, default=-1, help="仅控制单个关节(0~6)，-1表示控制全部")
    parser.add_argument("--startup-self-check", action="store_true", help="启动时做右臂往返自检，失败则退出")
    args = parser.parse_args()
    active_joint = None if args.active_joint < 0 else args.active_joint
    if active_joint is not None and not (0 <= active_joint <= 6):
        raise ValueError("--active-joint 必须是 -1 或 0~6")

    clients: Dict[str, ArmClient] = {}
    rig = None

    try:
        # ── 1. 连接双臂 ──
        for key in ("left", "right"):
            cfg = ARMS[key]
            c   = ArmClient(key, cfg["ip"], cfg["udp_port"])
            logger.info("连接 %s (%s) ...", cfg["label"], cfg["ip"])
            c.connect(wait_s=6.0)
            clients[key] = c
            logger.info("连接 %s 成功", cfg["label"])

        # ── 2. 等待夹爪 rm_plus 初始化（约2秒）──
        logger.info("等待夹爪rm_plus就绪...")
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if all(clients[k].gripper_online() for k in clients):
                break
            time.sleep(0.1)
        for key, c in clients.items():
            logger.info(
                "[%s] 关节=%s  夹爪online=%s pos=%.0f",
                key,
                [f"{v:.1f}" for v in (c.joints() or [])],
                c.gripper_online(),
                c.gripper() or 0,
            )

        if args.startup_self_check:
            _startup_control_self_check(clients["right"])

        # ── 3. 启动四路相机 ──
        rig = FourCameraRig(CAMERA_SN_JSON)
        cam_fps = rig.start()
        # 如果没有指定 hz，跟随相机帧率
        hz = args.hz if args.hz > 0 else float(cam_fps)

        # ── 4. 启动桥接服务 ──
        server = PolicyBridgeServer(
            clients      = clients,
            rig          = rig,
            host         = args.host,
            port         = args.port,
            hz           = hz,
            max_step_deg = args.max_step,
            speed        = args.speed,
            jpeg_quality = args.jpeg_quality,
            control_mode = args.control_mode,
            active_joint = active_joint,
        )
        server.serve_forever()

    except KeyboardInterrupt:
        logger.info("手动停止")
    except Exception as e:
        logger.error("致命错误: %s", e, exc_info=True)
    finally:
        for c in clients.values():
            c.disconnect()
        if rig:
            rig.stop()
        logger.info("已断开所有连接")


if __name__ == "__main__":
    main()
