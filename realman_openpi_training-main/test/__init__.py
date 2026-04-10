"""OpenPI 数据流测试模块。

提供端到端测试，验证从 LeRobot 数据集到 OpenPI 模型输入的完整数据流。

使用方法：
    # 运行所有测试
    python -m realman_openpi_training.test.test_data_pipeline
    
    # 或直接运行
    python realman_openpi_training/test/test_data_pipeline.py
    
    # 从代码中调用
    from realman_openpi_training.test import run_all_tests
    success = run_all_tests()
"""

from .test_data_pipeline import run_all_tests

__all__ = ["run_all_tests"]
