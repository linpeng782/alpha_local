"""
质量因子定义（盈利质量相关）
"""
from .base import *

# 盈利因子
eps = Factor("basic_earnings_per_share")  # 基本每股收益
earned_reserve = Factor("earned_reserve_per_share_lyr")  # 盈余公积
# 全部资产现金回收率：经营活动现金流量/全部资产
cfoa = Factor("cash_flow_from_operating_activities_mrq_0") / Factor("total_assets_mrq_0")

# ROE相关因子
roe_ttm = (
    2 * Factor("net_profit_parent_company_ttm_0") / 
    (Factor("equity_parent_company_lyr_0") + Factor("equity_parent_company_lyr_1"))
)
roe_lyr = Factor("net_profit_parent_company_lyr_0") / Factor("equity_parent_company_lyr_0")

# 季度ROE因子
roe_mrq_0 = Factor("net_profit_parent_company_mrq_0") / Factor("equity_parent_company_mrq_0")
roe_mrq_1 = Factor("net_profit_parent_company_mrq_1") / Factor("equity_parent_company_mrq_1")
roe_mrq_2 = Factor("net_profit_parent_company_mrq_2") / Factor("equity_parent_company_mrq_2")
roe_mrq_3 = Factor("net_profit_parent_company_mrq_3") / Factor("equity_parent_company_mrq_3")
roe_mrq_4 = Factor("net_profit_parent_company_mrq_4") / Factor("equity_parent_company_mrq_4")

# ROE变化因子
roe_qoq = roe_mrq_0 - roe_mrq_1  # 环比差分
roe_yoy = roe_mrq_0 - roe_mrq_4  # 同比差分
roe_delta = (
    roe_mrq_0 - 0.2 * (roe_mrq_1 + roe_mrq_2 + roe_mrq_3) - 0.4 * roe_mrq_4
)  # 平滑后的年度环比

# 因子定义字典
QUALITY_FACTORS = {
    "eps": eps,
    "cfoa": cfoa,
    "roe_ttm": roe_ttm,
    "roe_lyr": roe_lyr,
    "roe_mrq_0": roe_mrq_0,
    "roe_qoq": roe_qoq,
    "roe_yoy": roe_yoy,
    "roe_delta": roe_delta,
    "earned_reserve": earned_reserve,
}

# 因子配置信息
QUALITY_CONFIGS = {
    "cfoa": {
        "direction": 1,  # 全部资产现金回收率因子：高现金回收率更好
        "neutralize_options": [True, False],
    },
    "roe_ttm": {
        "direction": 1,  # ROE因子：高ROE更好
        "neutralize_options": [True, False],
    },
    "roe_lyr": {
        "direction": 1,  # ROE因子：高ROE更好
        "neutralize_options": [True, False],
    },
    "roe_mrq_0": {
        "direction": 1,  # ROE因子：高ROE更好
        "neutralize_options": [True, False],
    },
    "roe_qoq": {
        "direction": 1,  # ROE环比增长：高增长更好
        "neutralize_options": [True, False],
    },
    "roe_yoy": {
        "direction": 1,  # ROE同比增长：高增长更好
        "neutralize_options": [True, False],
    },
    "roe_delta": {
        "direction": 1,  # ROE平滑增长：高增长更好
        "neutralize_options": [True, False],
    },
    "earned_reserve": {
        "direction": 1,  # 盈余公积：高盈余公积更好
        "neutralize_options": [True, False],
    },
    "eps": {
        "direction": 1,  # EPS因子：高EPS更好
        "neutralize_options": [True, False],
    },
}
