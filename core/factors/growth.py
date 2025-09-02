"""
成长因子定义
"""
from .base import *

# 增长因子
inc_revenue_ttm = Factor("revenue_ttm_0") / Factor("revenue_ttm_4") - 1
net_profit_parent_company_growth_ratio_ttm = (
    Factor("net_profit_parent_company_ttm_0")
    / Factor("net_profit_parent_company_ttm_4")
    - 1
)

# 营业利润同比增长率 = （当期营业利润 - 去年同期营业利润）/绝对值（去年同期营业利润）
op_q_yoy_mrq = (
    Factor("profit_from_operation_mrq_0") - Factor("profit_from_operation_mrq_4")
) / ABS(Factor("profit_from_operation_mrq_4"))

# 因子定义字典
GROWTH_FACTORS = {
    "inc_revenue_ttm": inc_revenue_ttm,
    "net_profit_parent_company_growth_ratio_ttm": net_profit_parent_company_growth_ratio_ttm,
    "op_q_yoy_mrq": op_q_yoy_mrq,
}

# 因子配置信息
GROWTH_CONFIGS = {
    "inc_revenue_ttm": {
        "direction": 1,  # 营收增长率：高增长更好
        "neutralize_options": [True, False],
    },
    "net_profit_parent_company_growth_ratio_ttm": {
        "direction": 1,  # 净利润增长率：高增长更好
        "neutralize_options": [True, False],
    },
    "op_q_yoy_mrq": {
        "direction": 1,  # 营业利润同比增长率：高增长更好
        "neutralize_options": [True, False],
    },
}
