#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import json
import math
import socket
import struct
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from openpi_client import websocket_client_policy
except ModuleNotFoundError:
    _pkg_src = Path(__file__).resolve().parents[1] / "model3_openpi0.5" / "openpi-main" / "packages" / "openpi-client" / "src"
    import sys
    sys.path.insert(0, str(_pkg_src))
    from openpi_client import websocket_client_policy

RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0


def send_msg(sock: socket.socket, obj: dict) -> None:
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_msg(sock: socket.socket) -> Optional[dict]:
    header = recv_exact(sock, 4)
    if header is None:
        return None
    length = struct.unpack(">I", header)[0]
    payload = recv_exact(sock, length)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def decode_image(b64: Optional[str]) -> Optional[np.ndarray]:
    if not b64:
        return None
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def build_observation(obs: dict, prompt: str) -> tuple[dict, dict]:
    joints_right = obs["joints_right"]
    joints_left = obs["joints_left"]
    dexhand_right = float(obs.get("dexhand_right", 1))
    dexhand_left = float(obs.get("dexhand_left", 1))

    state = np.asarray([x * DEG2RAD for x in joints_right] + [dexhand_right], dtype=np.float32)
    head = decode_image(obs["images"].get("head"))
    right_wrist = decode_image(obs["images"].get("right_arm"))
    left_wrist = decode_image(obs["images"].get("left_arm"))
    if head is None or right_wrist is None or left_wrist is None:
        raise RuntimeError("head / right_arm / left_arm 图像不能为空")

    debug = {
        "obs_joints_right_deg": [float(x) for x in joints_right],
        "obs_joints_left_deg": [float(x) for x in joints_left],
        "obs_dexhand_right": dexhand_right,
        "obs_dexhand_left": dexhand_left,
        "head_shape": tuple(int(x) for x in head.shape),
        "right_wrist_shape": tuple(int(x) for x in right_wrist.shape),
        "left_wrist_shape": tuple(int(x) for x in left_wrist.shape),
        "head_mean": float(head.mean()),
        "right_wrist_mean": float(right_wrist.mean()),
        "left_wrist_mean": float(left_wrist.mean()),
        "head_std": float(head.std()),
        "right_wrist_std": float(right_wrist.std()),
        "left_wrist_std": float(left_wrist.std()),
    }

    return {
        "observation/state": state,
        "observation/image": head,
        "observation/wrist_image": right_wrist,
        "observation/left_wrist_image": left_wrist,
        "prompt": prompt,
    }, debug


def build_action(action_chunk: np.ndarray, action_index: int = 0) -> tuple[dict, dict]:
    actions = np.asarray(action_chunk)
    if actions.ndim == 1:
        actions = actions[None, :]
    action = actions[action_index]
    joints_deg = [float(x * RAD2DEG) for x in action[:7]]
    gripper_score = float(action[7])
    debug = {
        "raw_action_first": [float(x) for x in action.tolist()],
        "raw_action_min": float(actions.min()),
        "raw_action_max": float(actions.max()),
        "raw_action_mean": float(actions.mean()),
        "gripper_score": gripper_score,
    }
    return {
        "joints_right": joints_deg,
        "joints_left": None,
        "dexhand_right": None,
        "dexhand_left": None,
    }, debug


def main():
    parser = argparse.ArgumentParser(description="Right-arm OpenPI client for robot bridge")
    parser.add_argument("--robot-host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=9000)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=34000)
    parser.add_argument("--prompt", default="pick up the box with the right arm")
    parser.add_argument("--recv-timeout", type=float, default=10.0)
    parser.add_argument("--n-action-steps", type=int, default=10, help="每次推理后缓存并连续执行前 N 个动作")
    parser.add_argument("--chunk-refill-ratio", type=float, default=0.5, help="动作队列低于 n_action_steps*ratio 时触发下一次推理")
    parser.add_argument("--gripper-close-thresh", type=float, default=-0.15, help="夹爪闭合阈值（带迟滞）")
    parser.add_argument("--gripper-open-thresh", type=float, default=0.15, help="夹爪张开阈值（带迟滞）")
    parser.add_argument("--gripper-min-hold-steps", type=int, default=12, help="夹爪状态最小保持步数")
    parser.add_argument("--one-shot", action="store_true", help="只执行一次：观测->推理->下发一次动作后退出")
    parser.add_argument("--test-joint0-offset-deg", type=float, default=0.0, help="one-shot测试时额外叠加到J1的角度偏移")
    args = parser.parse_args()

    policy = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(args.recv_timeout)
    sock.connect((args.robot_host, args.robot_port))

    step = 0
    infer_count = 0
    last_infer_ms = 0.0
    pending_actions: deque[np.ndarray] = deque()
    refill_threshold = max(1, int(args.n_action_steps * args.chunk_refill_ratio))

    gripper_cmd = 0.0
    gripper_hold = 0
    try:
        while True:
            obs = recv_msg(sock)
            if obs is None:
                raise RuntimeError("机器人桥接断开")

            policy_obs, obs_debug = build_observation(obs, args.prompt)

            need_infer = (len(pending_actions) <= refill_threshold) or (step == 0)
            if args.one_shot:
                need_infer = (step == 0)

            if need_infer:
                t0 = time.perf_counter()
                result = policy.infer(policy_obs)
                last_infer_ms = (time.perf_counter() - t0) * 1000.0
                infer_count += 1

                action_chunk = np.asarray(result["actions"])
                if action_chunk.ndim == 1:
                    action_chunk = action_chunk[None, :]
                if action_chunk.shape[0] == 0:
                    raise RuntimeError("策略返回空动作序列")

                take_n = min(max(1, args.n_action_steps), int(action_chunk.shape[0]))
                for i in range(take_n):
                    pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

            if not pending_actions:
                raise RuntimeError("动作队列为空，无法下发控制")

            action_vec = pending_actions.popleft()
            action, action_debug = build_action(action_vec, 0)
            if args.one_shot and abs(args.test_joint0_offset_deg) > 1e-6:
                action["joints_right"][0] = float(action["joints_right"][0] + args.test_joint0_offset_deg)

            gripper_score = float(action_debug["gripper_score"])
            if gripper_hold <= 0:
                if gripper_score <= args.gripper_close_thresh:
                    gripper_cmd = 1.0
                    gripper_hold = args.gripper_min_hold_steps
                elif gripper_score >= args.gripper_open_thresh:
                    gripper_cmd = 0.0
                    gripper_hold = args.gripper_min_hold_steps
            else:
                gripper_hold -= 1
            action["dexhand_right"] = gripper_cmd

            send_msg(sock, action)
            step += 1

            if args.one_shot:
                print(
                    "one_shot_done step=1 infer_ms={infer_ms:.1f} gripper_score={gripper_score:.4f} cmd_joints_deg={cmd_joints}".format(
                        infer_ms=last_infer_ms,
                        gripper_score=action_debug["gripper_score"],
                        cmd_joints=[round(x, 2) for x in action["joints_right"]],
                    ),
                    flush=True,
                )
                break

            if step % 10 == 0:
                print(
                    "step={step} infer_cnt={infer_cnt} infer_ms={infer_ms:.1f} queue_size={queue_size} refill_threshold={refill_threshold} "
                    "obs_gripper={obs_gripper:.4f} act_gripper_binary={act_gripper:.0f} gripper_score={gripper_score:.4f} gripper_hold={gripper_hold} "
                    "obs_joints_deg={obs_joints} cmd_joints_deg={cmd_joints} "
                    "head_shape={head_shape} rwrist_shape={right_wrist_shape} lwrist_shape={left_wrist_shape} "
                    "head_mean={head_mean:.1f} rwrist_mean={right_wrist_mean:.1f} lwrist_mean={left_wrist_mean:.1f} "
                    "head_std={head_std:.1f} rwrist_std={right_wrist_std:.1f} lwrist_std={left_wrist_std:.1f} "
                    "raw_a0_7={raw_action}".format(
                        step=step,
                        infer_cnt=infer_count,
                        infer_ms=last_infer_ms,
                        queue_size=len(pending_actions),
                        refill_threshold=refill_threshold,
                        obs_gripper=obs_debug["obs_dexhand_right"],
                        act_gripper=action["dexhand_right"],
                        gripper_score=action_debug["gripper_score"],
                        gripper_hold=gripper_hold,
                        obs_joints=[round(x, 2) for x in obs_debug["obs_joints_right_deg"]],
                        cmd_joints=[round(x, 2) for x in action["joints_right"]],
                        head_shape=obs_debug["head_shape"],
                        right_wrist_shape=obs_debug["right_wrist_shape"],
                        left_wrist_shape=obs_debug["left_wrist_shape"],
                        head_mean=obs_debug["head_mean"],
                        right_wrist_mean=obs_debug["right_wrist_mean"],
                        left_wrist_mean=obs_debug["left_wrist_mean"],
                        head_std=obs_debug["head_std"],
                        right_wrist_std=obs_debug["right_wrist_std"],
                        left_wrist_std=obs_debug["left_wrist_std"],
                        raw_action=[round(x, 4) for x in action_debug["raw_action_first"]],
                    ),
                    flush=True,
                )
    finally:
        sock.close()


if __name__ == "__main__":
    main()
