"""
因子研究配置文件 - 最简版本
"""

# 导入必要的模块
import sys

sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils import *

# 基础算子定义
HIGH = Factor("high")
LOW = Factor("low")
OPEN = Factor("open")
CLOSE = Factor("close")
VOLUME = Factor("volume")
TURNOVER = Factor("total_turnover")

# 因子定义字典
FACTOR_DEFINITIONS = {
    "high_low_std_504": STD(HIGH / LOW, 504),
    "market_cap": Factor("market_cap"),
    "dp_ttm": Factor("dividend_yield_ttm"),
    "roe_ttm": Factor("roe_ttm"),
}

# 因子配置信息（支持多种配置）
FACTOR_CONFIGS = {
    "high_low_std_504": [
        {
            "direction": -1,
            "neutralize": True,
        },
        {
            "direction": -1,
            "neutralize": False,
        },
    ],
    "market_cap": [
        {
            "direction": -1,
            "neutralize": True,
        },
        {
            "direction": -1,
            "neutralize": False,
        },
    ],
    "dp_ttm": [
        {
            "direction": 1,
            "neutralize": True,
        },
        {
            "direction": 1,
            "neutralize": False,
        },
    ],
    "roe_ttm": [
        {
            "direction": 1,
            "neutralize": True,
        },
        {
            "direction": 1,
            "neutralize": False,
        },
    ],
}


def get_factor_for_test(factor_name, direction=None, neutralize=None):
    """
    获取用于测试的因子配置
    参数:
        factor_name: 因子名称
        direction: 指定方向 (1 或 -1)，如果为None则返回第一个配置
        neutralize: 指定是否中性化 (True 或 False)，如果为None则返回第一个配置
    """
    if factor_name not in FACTOR_CONFIGS:
        raise ValueError(f"未找到因子配置: {factor_name}")

    configs = FACTOR_CONFIGS[factor_name]

    # 如果没有指定参数，返回第一个配置
    if direction is None and neutralize is None:
        config = configs[0].copy()
    else:
        # 查找匹配的配置
        matching_config = None
        for config_item in configs:
            if (direction is None or config_item["direction"] == direction) and (
                neutralize is None or config_item["neutralize"] == neutralize
            ):
                matching_config = config_item
                break

        if matching_config is None:
            raise ValueError(
                f"未找到匹配的配置: {factor_name}, direction={direction}, neutralize={neutralize}"
            )

        config = matching_config.copy()

    # 添加因子定义
    config["definition"] = FACTOR_DEFINITIONS[factor_name]

    return config
