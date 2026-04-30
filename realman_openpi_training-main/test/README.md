# OpenPI 数据流测试模块

验证从 LeRobot 数据集到 OpenPI 模型输入的完整数据流正确性。

## 测试内容

1. **图像格式验证** - 确保传入 OpenPI 的图像是 uint8 格式
2. **相机映射验证** - 验证 cam0 → left_wrist_0_rgb, cam1 → base_0_rgb
3. **状态 Padding 验证** - 验证状态正确截取和 padding 到 32 维
4. **Delta 处理验证** - 验证 delta action 转换的正确性
5. **单位转换验证** - 验证 rad↔degree 转换的正确性
6. **端到端集成测试** - 验证完整数据流

## 运行方法

```bash
# 方式 1: 直接运行脚本
python realman_openpi_training/test/test_data_pipeline.py

# 方式 2: 作为模块运行
python -m realman_openpi_training.test.test_data_pipeline

# 方式 3: 指定数据集路径
python realman_openpi_training/test/test_data_pipeline.py --dataset-path /path/to/dataset
```

## 依赖要求

### 必需依赖
- `numpy` - 数值计算

### 可选依赖（部分测试需要）
- `torch` - 图像格式、相机映射、状态 padding 测试
- `opencv-python` - 图像处理
- `lerobot` / `robocoin` - 数据集加载测试
- `openpi` - Delta mask 和 Delta action 转换测试

如果可选依赖缺失，相关测试将显示 `[SKIP]` 状态。

## 测试结果状态

- `[PASS]` - 测试通过
- `[FAIL]` - 测试失败
- `[SKIP]` - 测试跳过（依赖缺失）

## 输出示例

```
测试汇总
========
[PASS] 代码静态分析
[PASS] 图像格式
[PASS] 相机映射
[SKIP] 状态 Padding
[PASS] 单位转换
[SKIP] 端到端集成

统计: 4 PASS, 0 FAIL, 2 SKIP

✅ 所有测试通过！
```
