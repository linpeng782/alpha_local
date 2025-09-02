"""
因子配置管理模块
"""
from .price_volume import PRICE_VOLUME_FACTORS, PRICE_VOLUME_CONFIGS
from .value import VALUE_FACTORS, VALUE_CONFIGS
from .quality import QUALITY_FACTORS, QUALITY_CONFIGS
from .growth import GROWTH_FACTORS, GROWTH_CONFIGS

# 合并所有因子定义
FACTOR_DEFINITIONS = {
    **PRICE_VOLUME_FACTORS,
    **VALUE_FACTORS,
    **QUALITY_FACTORS,
    **GROWTH_FACTORS,
}

# 合并所有因子配置
FACTOR_CONFIGS = {
    **PRICE_VOLUME_CONFIGS,
    **VALUE_CONFIGS,
    **QUALITY_CONFIGS,
    **GROWTH_CONFIGS,
}

def get_factor_config(factor_name, neutralize=False):
    """
    获取因子配置信息
    
    :param factor_name: 因子名称
    :param neutralize: 是否中性化
    :return: 因子配置字典
    """
    if factor_name not in FACTOR_CONFIGS:
        raise ValueError(f"未找到因子 {factor_name} 的配置信息")
    
    if factor_name not in FACTOR_DEFINITIONS:
        raise ValueError(f"未找到因子 {factor_name} 的定义")
    
    config = FACTOR_CONFIGS[factor_name].copy()
    config["definition"] = FACTOR_DEFINITIONS[factor_name]
    config["neutralize"] = neutralize
    
    # 检查中性化选项是否支持
    if neutralize not in config["neutralize_options"]:
        raise ValueError(f"因子 {factor_name} 不支持中性化参数 {neutralize}")
    
    return config

def list_factors_by_category():
    """列出所有因子按类别分组"""
    return {
        "price_volume": list(PRICE_VOLUME_FACTORS.keys()),
        "value": list(VALUE_FACTORS.keys()),
        "quality": list(QUALITY_FACTORS.keys()),
        "growth": list(GROWTH_FACTORS.keys()),
    }

def get_all_factor_names():
    """获取所有因子名称列表"""
    return list(FACTOR_DEFINITIONS.keys())
