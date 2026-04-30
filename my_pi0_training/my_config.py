"""Pi0 微调配置 —— 双臂 16 维任务

数据集信息:
  - 总条数: 1 条，366 帧
  - 相机: cam_high (顶部), cam_left_wrist (左腕), cam_right_wrist (右腕)
  - state/action: 16 维双臂（右臂7关节+1灵巧手 + 左臂7关节+1灵巧手）
  - 权重: pi0_base (PyTorch safetensors 格式)

训练参数:
  - num_train_steps=30000: 30000 步充分收敛
  - batch_size=1: 单条数据
  - save_interval=500: 每 500 步保存 checkpoint
"""

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import openpi.models.pi0_config as pi0_config
import openpi.transforms as _transforms

# ============================================================================
# 路径配置
# ============================================================================

OPENPI_DIR = Path("/share/0xyj/model3_openpi0.5/openpi-main")
DATASET_PATH = "/share/0xyj/model3_openpi0.5/lerobot_dataset"
WEIGHT_PATH = "/share/0xyj/model3_openpi0.5/pi0_base"

# repo_id 用于 norm_stats 路径
REPO_ID = "local/dual_arm_16d"

# OpenPI 模型期望的相机槽位顺序
OPENPI_CAMERA_SLOTS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

# 数据集中的相机 key（对应 observation.images.xxx）
# 映射关系:
#   cam_high        → base_0_rgb       (第1槽位：顶部全局视角)
#   cam_left_wrist  → left_wrist_0_rgb (第2槽位：左腕相机)
#   cam_right_wrist → right_wrist_0_rgb(第3槽位：右腕相机)
IMAGE_KEYS = (
    "observation.images.cam_high",        # → base_0_rgb
    "observation.images.cam_left_wrist",  # → left_wrist_0_rgb
    "observation.images.cam_right_wrist", # → right_wrist_0_rgb
)


# ============================================================================
# 数据转换器
# ============================================================================

@dataclasses.dataclass(frozen=True)
class MyRobotInputs(_transforms.DataTransformFn):
    """将 LeRobot 数据集格式转换为 OpenPI 模型期望的输入格式。

    相机映射:
        cam_high        → base_0_rgb       (第1槽位)
        cam_left_wrist  → left_wrist_0_rgb (第2槽位)
        cam_right_wrist → right_wrist_0_rgb(第3槽位)

    State: 16 维 (双臂 7+1 关节)
    Action: 16 维 (双臂动作)
    """
    image_keys: Sequence[str] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        import einops

        # ---- state: 取全部 16 维 (双臂) ----
        state = np.asarray(data["state"])
        if state.ndim > 1:
            state = state[..., :16]
        else:
            state = state[:16]

        # ---- 图像处理 ----
        def convert_image(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).clip(0, 255).astype(np.uint8)
            # 确保 HWC 格式
            if img.ndim == 3 and img.shape[0] == 3:
                img = einops.rearrange(img, "c h w -> h w c")
            return img

        images_dict = data.get("images", data.get("image", {}))

        extracted = []
        for key in self.image_keys:
            short_key = key.split(".")[-1] if "." in key else key
            if short_key in images_dict:
                extracted.append(convert_image(images_dict[short_key]))
            else:
                raise ValueError(
                    f"图像 key '{short_key}' 未找到。可用 key: {list(images_dict.keys())}"
                )

        placeholder = np.zeros_like(extracted[0])
        openpi_images = {}
        openpi_masks = {}
        for i, slot in enumerate(OPENPI_CAMERA_SLOTS):
            if i < len(extracted):
                openpi_images[slot] = extracted[i]
                openpi_masks[slot] = np.True_
            else:
                openpi_images[slot] = placeholder
                openpi_masks[slot] = np.False_

        result = {
            "state": state,
            "image": openpi_images,
            "image_mask": openpi_masks,
        }

        # ---- actions: 取全部 16 维 ----
        if "actions" in data:
            actions = np.asarray(data["actions"])
            result["actions"] = actions[..., :16] if actions.shape[-1] > 16 else actions

        if "prompt" in data:
            result["prompt"] = data["prompt"]

        return result


@dataclasses.dataclass(frozen=True)
class MyRobotOutputs(_transforms.DataTransformFn):
    """将模型输出的 actions 保持为 16 维。"""
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :16])}


# ============================================================================
# 辅助函数
# ============================================================================

def _make_repack_transforms(image_keys: Sequence[str]) -> _transforms.Group:
    """根据 image_keys 创建 repack_transforms。"""
    images_mapping = {}
    for key in image_keys:
        short_key = key.split(".")[-1] if "." in key else key
        full_key = key if key.startswith("observation.images.") else f"observation.images.{short_key}"
        images_mapping[short_key] = full_key

    return _transforms.Group(inputs=[
        _transforms.RepackTransform({
            "images": images_mapping,
            "state": "observation.state",
            "actions": "action",
        })
    ])


# ============================================================================
# 配置工厂
# ============================================================================

def get_my_configs():
    """返回自定义训练配置列表，注册到 OpenPI 配置系统。"""
    from openpi.training import config as _config
    from openpi.training import optimizer as _optimizer

    @dataclasses.dataclass(frozen=True)
    class MyDataConfig(_config.DataConfigFactory):
        """自定义数据配置工厂。"""
        use_delta_joint_actions: bool = True
        default_prompt: str | None = None
        prompt_from_task: bool = False
        local_root: str | None = None
        image_keys: Sequence[str] = IMAGE_KEYS
        repack_transforms: _transforms.Group = dataclasses.field(default=None)
        action_sequence_keys: Sequence[str] = ("action",)

        def __post_init__(self):
            if self.repack_transforms is None:
                object.__setattr__(
                    self,
                    'repack_transforms',
                    _make_repack_transforms(self.image_keys)
                )

        def create(self, assets_dirs, model_config) -> _config.DataConfig:
            short_keys = tuple(
                k.split(".")[-1] if "." in k else k for k in self.image_keys
            )
            data_transforms = _transforms.Group(
                inputs=[MyRobotInputs(image_keys=short_keys)],
                outputs=[MyRobotOutputs()],
            )

            if self.use_delta_joint_actions:
                # 前 14 维关节用 delta，后 2 维灵巧手用 absolute
                delta_mask = _transforms.make_bool_mask(14, -1)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_mask)],
                )

            model_transforms = _config.ModelTransformFactory(
                default_prompt=self.default_prompt
            )(model_config)

            repack = self.repack_transforms or _make_repack_transforms(self.image_keys)

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=self.action_sequence_keys,
                prompt_from_task=self.prompt_from_task,
            )

    return [
        # =====================================================================
        # 主训练配置: pi0 全量微调，使用本地 PyTorch pi0_base 权重
        # 双臂 16 维动作
        # =====================================================================
        _config.TrainConfig(
            name="pi0_dual_arm_16d",
            checkpoint_base_dir=str(OPENPI_DIR / "checkpoints"),
            assets_base_dir=str(OPENPI_DIR / "assets"),
            model=pi0_config.Pi0Config(
                action_dim=32,       # 模型内部维度固定32，实际用16维
                action_horizon=50,    # 每步预测50帧动作序列
                dtype="bfloat16",    # bf16 混合精度
            ),
            data=MyDataConfig(
                repo_id=REPO_ID,
                local_root=DATASET_PATH,
                use_delta_joint_actions=True,
                default_prompt="dual arm pick and place task",
                image_keys=IMAGE_KEYS,
            ),
            # 使用本地 pi0_base PyTorch 权重
            pytorch_weight_path=WEIGHT_PATH,
            num_train_steps=30000,
            batch_size=1,       # 单条数据
            log_interval=100,
            save_interval=500,   # 每 500 步保存 checkpoint
            num_workers=4,       # 数据加载 workers
            keep_period=0,        # 0=永久保留所有 checkpoint
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500,
                peak_lr=5e-5,
                decay_steps=30000,
                decay_lr=5e-6,
            ),
            ema_decay=0.999,
            wandb_enabled=True,
            overwrite=True,
        ),
    ]
