"""Realman RM75-6FB training configs for OpenPI.

================================================================================
相机命名约定（Realman 数据集标准）
================================================================================

在 Realman 数据集中，相机的命名遵循以下约定：

    cam0_rgb: 腕部相机 (wrist camera)
        - 安装位置：机械臂末端/夹爪附近
        - 视角特点：近距离观察操作对象
        - 对应 OpenPI 槽位：left_wrist_0_rgb
    
    cam1_rgb: 基座相机 (base camera)
        - 安装位置：工作台上的固定位置
        - 视角特点：全局视角，观察整个工作空间
        - 对应 OpenPI 槽位：base_0_rgb
    
    cam2_rgb+: 其他视角的基座相机（如有）
        - 额外的固定视角相机
        - 对应 OpenPI 槽位：right_wrist_0_rgb 或其他

================================================================================
图像输入配置说明
================================================================================

- 使用 image_keys 参数指定需要使用的图像观测
- 格式：["observation.images.cam1_rgb", "observation.images.cam0_rgb"]
- 图像按照 image_keys 中的顺序映射到 OpenPI 模型输入槽位：
  - 第 1 个图像 → base_0_rgb (基座相机视角)
  - 第 2 个图像 → left_wrist_0_rgb (腕部相机视角)
  - 第 3 个图像 → right_wrist_0_rgb (额外视角，如有)
- 未使用的相机通道会被填充为零图像，并通过 image_mask 标记为无效

示例：
- 单相机（仅基座）：image_keys=["observation.images.cam1_rgb"]
  → cam1_rgb (基座) → base_0_rgb
  
- 双相机（推荐）：image_keys=["observation.images.cam1_rgb", "observation.images.cam0_rgb"]
  → cam1_rgb (基座) → base_0_rgb
  → cam0_rgb (腕部) → left_wrist_0_rgb

================================================================================
Delta Action 配置
================================================================================

- 使用 use_delta_joint_actions=True 启用 delta 预测模式
- Delta mask: make_bool_mask(7, -1) 生成 [True]*7 + [False]
  - 前 7 维 (joints): 使用 delta 预测 (action = state_t+1 - state_t)
  - 第 8 维 (gripper): 使用 absolute 预测 (action = target_value)
"""

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Literal
import numpy as np
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.transforms as _transforms


# OpenPI 模型期望的相机输入顺序
OPENPI_CAMERA_SLOTS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]


@dataclasses.dataclass(frozen=True)
class RealmanInputs(_transforms.DataTransformFn):
    """Realman 输入转换器。
    
    将 LeRobot 数据集格式转换为 OpenPI 模型期望的输入格式。
    
    图像映射规则：
    - image_keys 中的图像按顺序映射到 OpenPI 的相机槽位
    - 第 1 个 → base_0_rgb (基座相机)
    - 第 2 个 → left_wrist_0_rgb (腕部相机)
    - 第 3 个 → right_wrist_0_rgb (额外视角)
    
    相机命名约定（Realman 数据集）：
    - cam0_rgb = 腕部相机 (wrist camera)，安装在机械臂末端
    - cam1_rgb = 基座相机 (base camera)，固定在工作台上
    - cam2_rgb+ = 其他视角的基座相机（如有）
    
    Attributes:
        image_keys: 图像观测 key 列表，格式如 ["cam0_rgb", "cam1_rgb"]
                   这里只需要短 key，因为 repack_transforms 已经处理了前缀。
                   默认顺序：["cam1_rgb", "cam0_rgb"] 即 [基座, 腕部]
    """
    # 默认：第 1 个是 cam1_rgb (基座) → base_0_rgb
    #       第 2 个是 cam0_rgb (腕部) → left_wrist_0_rgb
    image_keys: Sequence[str] = ("cam1_rgb", "cam0_rgb")
    
    def __call__(self, data: dict) -> dict:
        import einops
        
        # 处理 state：只取前 8 维 (7 joints + 1 gripper)
        state = np.asarray(data["state"])
        if state.shape[-1] == 14:
            state = state[..., :8]
        
        def convert_image(img):
            """将图像转换为 uint8 格式，并确保是 HWC 格式"""
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            if img.shape[0] == 3:
                img = einops.rearrange(img, "c h w -> h w c")
            return img
        
        images = data.get("images", data.get("image", {}))
        
        # 提取 image_keys 中指定的图像
        extracted_images = []
        for key in self.image_keys:
            # 支持完整 key 或短 key
            short_key = key.split(".")[-1] if "." in key else key
            
            if short_key in images:
                extracted_images.append(convert_image(images[short_key]))
            else:
                raise ValueError(
                    f"图像 key '{short_key}' 在数据集中未找到。"
                    f"可用的 key: {list(images.keys())}"
                )
        
        # 创建占位图像（用于未使用的相机通道）
        if extracted_images:
            placeholder = np.zeros_like(extracted_images[0])
        else:
            raise ValueError("image_keys 不能为空")
        
        # 按顺序映射到 OpenPI 相机槽位
        openpi_images = {}
        openpi_masks = {}
        
        for i, slot_name in enumerate(OPENPI_CAMERA_SLOTS):
            if i < len(extracted_images):
                openpi_images[slot_name] = extracted_images[i]
                openpi_masks[slot_name] = np.True_
            else:
                openpi_images[slot_name] = placeholder
                openpi_masks[slot_name] = np.False_
        
        # 构建 OpenPI 期望的输入格式
        inputs = {
            "state": state,
            "image": openpi_images,
            "image_mask": openpi_masks,
        }
        
        # 处理 actions
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = actions[..., :8] if actions.shape[-1] > 8 else actions
        
        # 处理 prompt
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        return inputs


@dataclasses.dataclass(frozen=True)
class RealmanOutputs(_transforms.DataTransformFn):
    """Realman 输出转换器。
    
    将模型输出的 actions 截取为实际需要的 8 维 (7 joints + 1 gripper)。
    """
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}


def _make_repack_transforms(image_keys: Sequence[str]) -> _transforms.Group:
    """根据 image_keys 创建 repack_transforms。
    
    只提取指定的图像，不包含深度图或其他未指定的图像。
    
    Args:
        image_keys: 图像观测 key 列表
            格式：["observation.images.cam0_rgb", "observation.images.cam1_rgb"]
            或短格式：["cam0_rgb", "cam1_rgb"]
    
    Returns:
        配置好的 repack_transforms
    """
    images_mapping = {}
    
    for key in image_keys:
        # 处理完整 key 和短 key
        if key.startswith("observation.images."):
            full_key = key
            short_key = key.split(".")[-1]
        else:
            short_key = key
            full_key = f"observation.images.{key}"
        
        images_mapping[short_key] = full_key
    
    return _transforms.Group(inputs=[
        _transforms.RepackTransform({
            "images": images_mapping,
            "state": "observation.state",
            "actions": "action",
        })
    ])


def _normalize_image_keys(image_keys: Sequence[str]) -> tuple[str, ...]:
    """标准化 image_keys，提取短 key 用于 RealmanInputs。
    
    Args:
        image_keys: 可能是完整 key 或短 key 的列表
    
    Returns:
        短 key 的元组
    """
    normalized = []
    for key in image_keys:
        short_key = key.split(".")[-1] if "." in key else key
        normalized.append(short_key)
    return tuple(normalized)



def get_realman_configs():
    """获取 Realman 训练配置列表。
    
    配置说明：
    - repo_id: 数据集标识符，格式为 "local/<dataset_name>" 或 HuggingFace repo
    - local_root: 本地数据集的绝对路径，设置后将直接从此路径加载
    - image_keys: 图像观测 key 列表，按顺序映射到 OpenPI 相机槽位
        - 格式：["observation.images.cam1_rgb", "observation.images.cam0_rgb"]
        - 第 1 个 → base_0_rgb, 第 2 个 → left_wrist_0_rgb, 第 3 个 → right_wrist_0_rgb
    
    使用时请修改 local_root 为你的实际数据集路径！
    """
    from openpi.training import config as _config
    from openpi.training import optimizer as _optimizer
    from openpi.training import weight_loaders
    
    # 默认图像 keys（基座 + 腕部）
    DEFAULT_IMAGE_KEYS = (
        "observation.images.cam1_rgb",  # 基座相机 → base_0_rgb
        "observation.images.cam0_rgb",  # 腕部相机 → left_wrist_0_rgb
    )
    
    @dataclasses.dataclass(frozen=True)
    class LeRobotRealmanDataConfig(_config.DataConfigFactory):
        """Realman 数据集配置工厂。
        
        Attributes:
            use_delta_joint_actions: 是否使用 delta joints 模式
            default_prompt: 默认任务指令
            prompt_from_task: 是否从数据集的 task 字段读取 prompt
            local_root: 本地数据集的绝对路径
            image_keys: 图像观测 key 列表，按顺序映射到 OpenPI 相机槽位
                - 格式：["observation.images.cam1_rgb", "observation.images.cam0_rgb"]
                - 第 1 个 → base_0_rgb
                - 第 2 个 → left_wrist_0_rgb
                - 第 3 个 → right_wrist_0_rgb
        """
        use_delta_joint_actions: bool = True
        default_prompt: str | None = None
        prompt_from_task: bool = False
        local_root: str | None = None
        image_keys: Sequence[str] = DEFAULT_IMAGE_KEYS
        repack_transforms: _transforms.Group = dataclasses.field(default=None)
        action_sequence_keys: Sequence[str] = ("action",)
        
        def __post_init__(self):
            # 如果 repack_transforms 未设置，根据 image_keys 自动生成
            if self.repack_transforms is None:
                object.__setattr__(
                    self, 
                    'repack_transforms', 
                    _make_repack_transforms(self.image_keys)
                )
        
        def create(self, assets_dirs, model_config) -> _config.DataConfig:
            # 标准化 image_keys 为短 key
            short_image_keys = _normalize_image_keys(self.image_keys)
            
            # 创建数据转换器，传入标准化后的 image_keys
            data_transforms = _transforms.Group(
                inputs=[RealmanInputs(image_keys=short_image_keys)], 
                outputs=[RealmanOutputs()]
            )
            
            if self.use_delta_joint_actions:
                delta_mask = _transforms.make_bool_mask(7, -1)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_mask)],
                )
            
            model_transforms = _config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
            
            # 确保 repack_transforms 已初始化
            repack = self.repack_transforms
            if repack is None:
                repack = _make_repack_transforms(self.image_keys)
            
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=self.action_sequence_keys,
                prompt_from_task=self.prompt_from_task,
            )
    
    # =========================================================================
    # 默认数据集路径（请根据实际情况修改）
    # =========================================================================
    DEFAULT_DATASET_PATH = "/home/ubuntu/Desktop/Workspace/lerobot_policy_deploy/data/local/realman_teleop_0111_right_plate_50_openpi1"
    
    return [
        # =====================================================================
        # π₀ 配置
        # =====================================================================
        _config.TrainConfig(
            name="pi0_realman",
            model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
            data=LeRobotRealmanDataConfig(
                repo_id="local/realman_teleop_0111_right_plate_50",
                local_root=DEFAULT_DATASET_PATH,
                use_delta_joint_actions=True,
                default_prompt="the gripper pick up the yellow banana and put it in the white plate",
                image_keys=DEFAULT_IMAGE_KEYS,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30000, batch_size=32, log_interval=100, save_interval=1000, keep_period=5000,
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=1e-4, decay_steps=30000, decay_lr=1e-5),
        ),
        
        # π₀ LoRA 微调（低显存）
        _config.TrainConfig(
            name="pi0_realman_lora",
            model=pi0_config.Pi0Config(action_dim=32, action_horizon=50, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=LeRobotRealmanDataConfig(
                repo_id="local/realman_teleop_0111_right_plate_50",
                local_root=DEFAULT_DATASET_PATH,
                use_delta_joint_actions=True,
                default_prompt="the gripper pick up the yellow banana and put it in the white plate",
                image_keys=DEFAULT_IMAGE_KEYS,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30000, batch_size=32,
            freeze_filter=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora").get_freeze_filter(),
            ema_decay=None,
        ),
        
        # =====================================================================
        # π₀.5 配置
        # =====================================================================
        _config.TrainConfig(
            name="pi05_realman",
            model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
            data=LeRobotRealmanDataConfig(
                repo_id="local/realman_teleop_0111_right_plate_50",
                local_root=DEFAULT_DATASET_PATH,
                use_delta_joint_actions=True,
                default_prompt="the gripper pick up the yellow banana and put it in the white plate",
                image_keys=DEFAULT_IMAGE_KEYS,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=30000, batch_size=32,
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=5e-5, decay_steps=30000, decay_lr=5e-6),
            ema_decay=0.999,
        ),
        
        # π₀.5 使用本地 PyTorch 权重（离线训练）
        _config.TrainConfig(
            name="pi05_realman_pytorch",
            model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
            data=LeRobotRealmanDataConfig(
                repo_id="local/realman_teleop_0111_right_plate_50",
                local_root=DEFAULT_DATASET_PATH,
                use_delta_joint_actions=True,
                default_prompt="the gripper pick up the yellow banana and put it in the white plate",
                image_keys=DEFAULT_IMAGE_KEYS,
            ),
            pytorch_weight_path="data/openpi-assets/checkpoints/pi05_droid_pytorch",
            num_train_steps=30000, batch_size=8,  # 减小 batch_size 以适应 24GB 显存
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=5e-5, decay_steps=30000, decay_lr=5e-6),
            ema_decay=0.999,
        ),
        
        # π₀ 使用本地 PyTorch 权重（离线训练）
        _config.TrainConfig(
            name="pi0_realman_pytorch",
            model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
            data=LeRobotRealmanDataConfig(
                repo_id="local/realman_teleop_0111_right_plate_50",
                local_root=DEFAULT_DATASET_PATH,
                use_delta_joint_actions=True,
                default_prompt="the gripper pick up the yellow banana and put it in the white plate",
                image_keys=DEFAULT_IMAGE_KEYS,
            ),
            pytorch_weight_path="data/openpi-assets/checkpoints/pi0_droid_pytorch",
            num_train_steps=30000, batch_size=8,  # 减小 batch_size 以适应 24GB 显存
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=5e-5, decay_steps=30000, decay_lr=5e-6),
            ema_decay=0.999,
        ),
        
        # =====================================================================
        # π₀-FAST 配置
        # =====================================================================
        _config.TrainConfig(
            name="pi0_fast_realman",
            model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=50, max_token_len=180),
            data=LeRobotRealmanDataConfig(
                repo_id="local/realman_teleop_0111_right_plate_50",
                local_root=DEFAULT_DATASET_PATH,
                use_delta_joint_actions=True,
                default_prompt="the gripper pick up the yellow banana and put it in the white plate",
                image_keys=DEFAULT_IMAGE_KEYS,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
            num_train_steps=30000, batch_size=32,
        ),
        
        # =====================================================================
        # 推理配置（不需要数据集路径）
        # =====================================================================
        _config.TrainConfig(
            name="pi0_realman_inference",
            model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
            data=LeRobotRealmanDataConfig(
                use_delta_joint_actions=True,
                image_keys=DEFAULT_IMAGE_KEYS,
            ),
        ),
        _config.TrainConfig(
            name="pi05_realman_inference",
            model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
            data=LeRobotRealmanDataConfig(
                use_delta_joint_actions=True,
                image_keys=DEFAULT_IMAGE_KEYS,
            ),
        ),
        
        # =====================================================================
        # 单相机配置示例
        # =====================================================================
        # 只使用基座相机
        _config.TrainConfig(
            name="pi0_realman_base_only_pytorch",
            model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
            data=LeRobotRealmanDataConfig(
                repo_id="local/realman_teleop_0111_right_plate_50",
                local_root=DEFAULT_DATASET_PATH,
                use_delta_joint_actions=True,
                default_prompt="the gripper pick up the yellow banana and put it in the white plate",
                image_keys=("observation.images.cam1_rgb",),  # 只使用基座相机
            ),
            pytorch_weight_path="data/openpi-assets/checkpoints/pi0_droid_pytorch",
            num_train_steps=30000, batch_size=32,
        ),
        
        # 只使用腕部相机
        _config.TrainConfig(
            name="pi0_realman_wrist_only_pytorch",
            model=pi0_config.Pi0Config(action_dim=32, action_horizon=50),
            data=LeRobotRealmanDataConfig(
                repo_id="local/realman_teleop_0111_right_plate_50",
                local_root=DEFAULT_DATASET_PATH,
                use_delta_joint_actions=True,
                default_prompt="the gripper pick up the yellow banana and put it in the white plate",
                image_keys=("observation.images.cam0_rgb",),  # 只使用腕部相机
            ),
            pytorch_weight_path="data/openpi-assets/checkpoints/pi0_droid_pytorch",
            num_train_steps=30000, batch_size=32,
        ),
        
        # =====================================================================
        # 多任务配置
        # =====================================================================
        _config.TrainConfig(
            name="pi05_realman_multitask",
            model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
            data=LeRobotRealmanDataConfig(
                repo_id="local/realman_multi_task_dataset",
                local_root="/path/to/your/multi_task_dataset",  # 请修改为实际路径
                use_delta_joint_actions=True,
                prompt_from_task=True,  # 从数据集 task 字段读取 prompt
                default_prompt=None,
                image_keys=DEFAULT_IMAGE_KEYS,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
            num_train_steps=30000, batch_size=32,
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=5e-5, decay_steps=30000, decay_lr=5e-6),
            ema_decay=0.999,
        ),
    ]
