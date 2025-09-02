"""
量价因子定义
"""
from .base import *

# 波动率因子
high_low_std_60 = STD(HIGH / LOW, 60)
high_low_std_504 = STD(HIGH / LOW, 504)

# 换手率因子
turnover_avg_20 = MA(DAILY_TURNOVER_RATE, 20)
turnover_std_20 = STD(DAILY_TURNOVER_RATE, 20)
turnover_avg_60 = MA(DAILY_TURNOVER_RATE, 60)
turnover_std_60 = STD(DAILY_TURNOVER_RATE, 60)
turnover_std_252 = STD(DAILY_TURNOVER_RATE, 252)

# 成交额因子
turnover_zamount_20 = MA(TURNOVER, 20) / STD(TURNOVER, 20)

# 量价相关性
corr_price_turnover_20 = CORR(CLOSE, DAILY_TURNOVER_RATE, 20)

# 因子定义字典
PRICE_VOLUME_FACTORS = {
    "high_low_std_60": high_low_std_60,
    "high_low_std_504": high_low_std_504,
    "turnover_avg_20": turnover_avg_20,
    "turnover_std_20": turnover_std_20,
    "turnover_avg_60": turnover_avg_60,
    "turnover_std_60": turnover_std_60,
    "turnover_std_252": turnover_std_252,
    "turnover_zamount_20": turnover_zamount_20,
    "corr_price_turnover_20": corr_price_turnover_20,
}

# 因子配置信息
PRICE_VOLUME_CONFIGS = {
    "high_low_std_60": {
        "direction": -1,  # 波动率因子：低波动率股票表现更好
        "neutralize_options": [True, False],
    },
    "high_low_std_504": {
        "direction": -1,  # 波动率因子：低波动率股票表现更好
        "neutralize_options": [True, False],
    },
    "turnover_avg_20": {
        "direction": -1,  # 换手率因子：低换手率股票表现更好
        "neutralize_options": [True, False],
    },
    "turnover_std_20": {
        "direction": -1,  # 换手率因子：低换手率股票表现更好
        "neutralize_options": [True, False],
    },
    "turnover_avg_60": {
        "direction": -1,  # 换手率因子：低换手率股票表现更好
        "neutralize_options": [True, False],
    },
    "turnover_std_60": {
        "direction": -1,  # 换手率因子：低换手率股票表现更好
        "neutralize_options": [True, False],
    },
    "turnover_std_252": {
        "direction": -1,  # 换手率因子：低换手率股票表现更好
        "neutralize_options": [True, False],
    },
    "turnover_zamount_20": {
        "direction": -1,  # 成交额因子：低成交额波动更好
        "neutralize_options": [True, False],
    },
    "corr_price_turnover_20": {
        "direction": -1,  # 量价相关性因子：低相关性更好
        "neutralize_options": [True, False],
    },
}
