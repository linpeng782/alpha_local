"""
因子研究配置文件 - 最简版本
"""

# 导入必要的模块
import sys

sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils import *

# 量价基础算子定义
HIGH = Factor("high")
LOW = Factor("low")
OPEN = Factor("open")
CLOSE = Factor("close")
VOLUME = Factor("volume")
TURNOVER = Factor("total_turnover")


def turnover_rate(order_book_ids, start_date, end_date):
    return (
        get_turnover_rate(order_book_ids, start_date, end_date, fields="today")
        .today.unstack("order_book_id")
        .reindex(
            columns=order_book_ids,
            index=pd.to_datetime(get_trading_dates(start_date, end_date)),
        )
    )


DAILY_TURNOVER_RATE = UserDefinedLeafFactor("DAILY_TURNOVER_RATE", turnover_rate)

# 定义量价因子
high_low_std_504 = STD(HIGH / LOW, 504)
turnover_std_20 = STD(DAILY_TURNOVER_RATE, 20)
turnover_std_60 = STD(DAILY_TURNOVER_RATE, 60)
turnover_std_252 = STD(DAILY_TURNOVER_RATE, 252)
corr_price_turnover_20 = CORR(CLOSE, DAILY_TURNOVER_RATE, 20)


# 定义基本面因子
# 价值因子
market_cap_3 = Factor("market_cap_3")
pe_ttm = Factor("market_cap_3") / Factor("net_profit_parent_company_ttm_0")
pb_lyr = Factor("market_cap_3") / Factor("equity_parent_company_lyr_0")
bp_lyr = Factor("equity_parent_company_lyr_0") / Factor("market_cap_3")
dp_ttm = Factor("dividend_yield_ttm")


# 盈利因子
eps = Factor("basic_earnings_per_share")
earned_reserve = Factor("earned_reserve_per_share_lyr")  # 盈余公积
roe_ttm = (
    2
    * Factor("net_profit_parent_company_ttm_0")
    / (Factor("equity_parent_company_ttm_0") + Factor("equity_parent_company_ttm_1"))
)
roe_lyr = (
    2
    * Factor("net_profit_parent_company_lyr_0")
    / (Factor("equity_parent_company_lyr_0") + Factor("equity_parent_company_lyr_1"))
)
roe_mrq_0 = (
    4
    * Factor("net_profit_parent_company_mrq_0")
    / (
        (Factor("equity_parent_company_mrq_0") + Factor("equity_parent_company_mrq_1"))
        / 2
    )
)
roe_mrq_1 = (
    4
    * Factor("net_profit_parent_company_mrq_1")
    / (
        (Factor("equity_parent_company_mrq_1") + Factor("equity_parent_company_mrq_2"))
        / 2
    )
)
roe_mrq_2 = (
    4
    * Factor("net_profit_parent_company_mrq_2")
    / (
        (Factor("equity_parent_company_mrq_2") + Factor("equity_parent_company_mrq_3"))
        / 2
    )
)
roe_mrq_3 = (
    4
    * Factor("net_profit_parent_company_mrq_3")
    / (
        (Factor("equity_parent_company_mrq_3") + Factor("equity_parent_company_mrq_4"))
        / 2
    )
)
roe_mrq_4 = (
    4
    * Factor("net_profit_parent_company_mrq_4")
    / (
        (Factor("equity_parent_company_mrq_4") + Factor("equity_parent_company_mrq_5"))
        / 2
    )
)

roe_qoq = roe_mrq_0 - roe_mrq_1
roe_yoy = roe_mrq_0 - roe_mrq_4
roe_delta = roe_mrq_0 - 0.2 * (roe_mrq_1 + roe_mrq_2 + roe_mrq_3) - 0.4 * roe_mrq_4

# 增长因子
inc_revenue_ttm = Factor("revenue_ttm_0") / Factor("revenue_ttm_4") - 1
net_profit_parent_company_growth_ratio_ttm = (
    Factor("net_profit_parent_company_ttm_0")
    / Factor("net_profit_parent_company_ttm_4")
    - 1
)


# 定义因子定义字典
FACTOR_DEFINITIONS = {
    "high_low_std_504": high_low_std_504,
    "turnover_std_20": turnover_std_20,
    "turnover_std_252": turnover_std_252,
    "turnover_std_60": turnover_std_60,
    "corr_price_turnover_20": corr_price_turnover_20,
    "market_cap_3": market_cap_3,
    "pe_ttm": pe_ttm,
    "eps": eps,
    "dp_ttm": dp_ttm,
    "roe_ttm": roe_ttm,
    "roe_lyr": roe_lyr,
    "roe_mrq_0": roe_mrq_0,
    "roe_mrq_1": roe_mrq_1,
    "roe_qoq": roe_qoq,
    "roe_yoy": roe_yoy,
    "roe_delta": roe_delta,
    "pb_lyr": pb_lyr,
    "bp_lyr": bp_lyr,
    "inc_revenue_ttm": inc_revenue_ttm,
    "net_profit_parent_company_growth_ratio_ttm": net_profit_parent_company_growth_ratio_ttm,
    "earned_reserve": earned_reserve,
}

# 定义因子配置信息（每个因子固定direction，只配置neutralize）
FACTOR_CONFIGS = {
    "high_low_std_504": {
        "direction": -1,  # 波动率因子：低波动率股票表现更好
        "neutralize_options": [True, False],
    },
    "turnover_std_20": {
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
    "corr_price_turnover_20": {
        "direction": -1,  # 换手率因子：低换手率股票表现更好
        "neutralize_options": [True, False],
    },
    "market_cap_3": {
        "direction": -1,  # 市值因子：小市值效应
        "neutralize_options": [True, False],
    },
    "dp_ttm": {
        "direction": 1,  # 股息率因子：高股息率更好
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
    "roe_mrq_1": {
        "direction": 1,  # ROE因子：高ROE更好
        "neutralize_options": [True, False],
    },
    "roe_qoq": {
        "direction": 1,  # ROE因子：高ROE更好
        "neutralize_options": [True, False],
    },
    "roe_yoy": {
        "direction": 1,  # ROE因子：高ROE更好
        "neutralize_options": [True, False],
    },
    "roe_delta": {
        "direction": 1,  # ROE因子：高ROE更好
        "neutralize_options": [True, False],
    },
    "pe_ttm": {
        "direction": -1,  # PE因子：低估值效应
        "neutralize_options": [True, False],
    },
    "pb_lyr": {
        "direction": -1,  # PB因子：低估值效应
        "neutralize_options": [True, False],
    },
    "bp_lyr": {
        "direction": 1,  # BP因子：价值效应（PB的倒数）
        "neutralize_options": [True, False],
    },
    "inc_revenue_ttm": {
        "direction": 1,  # 营收增长率：高增长更好
        "neutralize_options": [True, False],
    },
    "net_profit_parent_company_growth_ratio_ttm": {
        "direction": 1,  # 净利润增长率：高增长更好
        "neutralize_options": [True, False],
    },
    "earned_reserve": {
        "direction": 1,  # 盈余公积：高公积更好
        "neutralize_options": [True, False],
    },
    "eps": {
        "direction": 1,  # EPS因子：高EPS更好
        "neutralize_options": [True, False],
    },
}


def get_factor_config(factor_name, neutralize=False):
    """
    获取用于测试的因子配置（简化版）

    参数:
        factor_name: 因子名称
        neutralize: 是否中性化 (True 或 False)，默认False

    返回:
        包含 direction, neutralize, definition 的配置字典
    """
    if factor_name not in FACTOR_CONFIGS:
        raise ValueError(f"未找到因子配置: {factor_name}")

    factor_config = FACTOR_CONFIGS[factor_name]

    # 检查neutralize参数是否在支持的选项中
    if neutralize not in factor_config["neutralize_options"]:
        raise ValueError(
            f"因子 {factor_name} 不支持 neutralize={neutralize}。"
            f"支持的选项: {factor_config['neutralize_options']}"
        )

    # 返回配置
    config = {
        "direction": factor_config["direction"],
        "neutralize": neutralize,
        "definition": FACTOR_DEFINITIONS[factor_name],
    }

    return config
