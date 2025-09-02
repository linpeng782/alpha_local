"""
价值因子定义
"""
from .base import *

# 价值因子
market_cap_3 = Factor("market_cap_3")
pe_ttm = Factor("market_cap_3") / Factor("net_profit_parent_company_ttm_0")
pb_lyr = Factor("market_cap_3") / Factor("equity_parent_company_lyr_0")
bp_lyr = Factor("equity_parent_company_lyr_0") / Factor("market_cap_3")
dp_ttm = Factor("dividend_yield_ttm")

# 因子定义字典
VALUE_FACTORS = {
    "market_cap_3": market_cap_3,
    "pe_ttm": pe_ttm,
    "pb_lyr": pb_lyr,
    "bp_lyr": bp_lyr,
    "dp_ttm": dp_ttm,
}

# 因子配置信息
VALUE_CONFIGS = {
    "market_cap_3": {
        "direction": -1,  # 市值因子：小市值效应
        "neutralize_options": [True, False],
    },
    "pe_ttm": {
        "direction": -1,  # PE因子：低PE更好
        "neutralize_options": [True, False],
    },
    "pb_lyr": {
        "direction": -1,  # PB因子：低PB更好
        "neutralize_options": [True, False],
    },
    "bp_lyr": {
        "direction": 1,  # BP因子：高BP更好
        "neutralize_options": [True, False],
    },
    "dp_ttm": {
        "direction": 1,  # 股息率因子：高股息率更好
        "neutralize_options": [True, False],
    },
}
