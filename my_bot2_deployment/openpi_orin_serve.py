#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import dataclasses
import pathlib

import numpy as np
import torch

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


@dataclasses.dataclass(frozen=True)
class RightArmInputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        right_wrist_image = _parse_image(data["observation/wrist_image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        state = np.asarray(data["observation/state"], dtype=np.float32)
        return {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                # 训练时相机映射是:
                # cam_right_wrist -> left_wrist_0_rgb
                # cam_left_wrist  -> right_wrist_0_rgb
                "left_wrist_0_rgb": right_wrist_image,
                "right_wrist_0_rgb": left_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "prompt": data["prompt"],
        }


@dataclasses.dataclass(frozen=True)
class PadStateTo32(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] < 32:
            state = np.pad(state, (0, 32 - state.shape[-1]))
        data = dict(data)
        data["state"] = state
        return data


@dataclasses.dataclass(frozen=True)
class RightArmOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8], dtype=np.float32)}


@dataclasses.dataclass(frozen=True)
class RightArmDataConfig(_config.DataConfigFactory):
    @staticmethod
    def _load_norm_stats_from_checkpoint(checkpoint_dir: pathlib.Path) -> dict[str, _normalize.NormStats]:
        return _normalize.load(checkpoint_dir / "assets" / "local" / "right_arm_box_pick")

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        del assets_dirs
        del model_config
        raise RuntimeError("This custom config is only used for inference server construction.")


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def build_policy(
    checkpoint_dir: pathlib.Path,
    prompt: str | None,
    device: str,
    fixed_noise_seed: int | None = None,
) -> _policy.Policy:
    train_config = _config.TrainConfig(
        name="pi0_right_arm_orin_infer",
        model=pi0_config.Pi0Config(action_dim=32, pytorch_compile_mode=None),
        data=_config.FakeDataConfig(),
        policy_metadata={"checkpoint": str(checkpoint_dir)},
        assets_base_dir=str(checkpoint_dir.parent),
        checkpoint_base_dir=str(checkpoint_dir.parent),
        wandb_enabled=False,
    )

    weight_path = checkpoint_dir / "model.safetensors"
    model = train_config.model.load_pytorch(train_config, str(weight_path))

    norm_stats = _normalize.load(checkpoint_dir / "assets" / "local" / "right_arm_box_pick")

    transforms = [
        RightArmInputs(),
        _transforms.Normalize(norm_stats, use_quantiles=True),
        PadStateTo32(),
        _transforms.ResizeImages(224, 224),
        _transforms.TokenizePrompt(
            tokenizer=_tokenizer.PaligemmaTokenizer(train_config.model.max_token_len)
        ),
    ]

    delta_mask = _transforms.make_bool_mask(7, -1)

    output_transforms = [
        _transforms.Unnormalize(norm_stats, use_quantiles=True),
        _transforms.AbsoluteActions(delta_mask),
        RightArmOutputs(),
    ]

    policy_sample_kwargs = {}
    if fixed_noise_seed is not None:
        rng = np.random.default_rng(fixed_noise_seed)
        fixed_noise_np = rng.standard_normal((1, 50, 32), dtype=np.float32)
        if device.startswith("cuda") and torch.cuda.is_available():
            policy_sample_kwargs["noise"] = torch.from_numpy(fixed_noise_np).to(device)
        else:
            policy_sample_kwargs["noise"] = torch.from_numpy(fixed_noise_np)

    return _policy.Policy(
        model,
        transforms=transforms,
        output_transforms=output_transforms,
        sample_kwargs=policy_sample_kwargs,
        metadata=train_config.policy_metadata,
        pytorch_device=device,
        is_pytorch=True,
    )


def main():
    parser = argparse.ArgumentParser(description="OpenPI right-arm server for Orin")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--port", type=int, default=34000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--default-prompt", default="pick up the box with the right arm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fixed-noise-seed",
        type=int,
        default=None,
        help="固定推理噪声seed；用于稳定联机输出调试，不设则保持默认随机采样",
    )
    args = parser.parse_args()

    checkpoint_dir = pathlib.Path(args.checkpoint_dir).resolve()
    policy = build_policy(
        checkpoint_dir,
        args.default_prompt,
        args.device,
        fixed_noise_seed=args.fixed_noise_seed,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={"checkpoint": str(checkpoint_dir), "action_dim": 8},
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
