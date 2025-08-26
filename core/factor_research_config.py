"""
因子研究配置文件 - 最简版本
"""

# 导入必要的模块
import sys
sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils import Factor, STD

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
}

# 因子配置信息（只保留必要字段）
FACTOR_CONFIGS = {
    "high_low_std_504": {
        "direction": -1,
        "neutralize": True,
        "description": "高低价比的504日标准差"
    },
    "market_cap": {
        "direction": -1,
        "neutralize": False,
        "description": "市值因子，小市值效应"
    },
}


def get_factor_for_test(factor_name):
    """
    获取用于测试的因子配置
    """
    if factor_name not in FACTOR_CONFIGS:
        raise ValueError(f"未找到因子配置: {factor_name}")
    
    config = FACTOR_CONFIGS[factor_name].copy()
    # 直接从 FACTOR_DEFINITIONS 获取因子定义
    config["definition"] = FACTOR_DEFINITIONS[factor_name]
    
    return config
