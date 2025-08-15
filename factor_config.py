"""
因子配置文件 
"""

import pandas as pd
from factor_processing_utils import *

# 自定义因子函数
def turnover_rate(order_book_ids, start_date, end_date):
    """
    获取日度换手率数据
    """
    return (
        get_turnover_rate(order_book_ids, start_date, end_date, fields="today")
        .today.unstack("order_book_id")
        .reindex(
            columns=order_book_ids,
            index=pd.to_datetime(get_trading_dates(start_date, end_date)),
        )
    )

# 用户定义因子
DAILY_TURNOVER_RATE = UserDefinedLeafFactor("DAILY_TURNOVER_RATE", turnover_rate)

# 因子定义
FACTOR_DICT = {
    # 基本面因子
    "cfoa_mrq": Factor("cash_flow_from_operating_activities_mrq_0")
    / Factor("total_assets_mrq_0"),
    "atdy_mrq": Factor("operating_revenue_mrq_0") / Factor("total_assets_mrq_0")
    - Factor("operating_revenue_mrq_4") / Factor("total_assets_mrq_4"),
    "ccr_mrq": Factor("cash_flow_from_operating_activities_mrq_0")
    / Factor("current_liabilities_mrq_0"),
    
    # 流动性因子
    "liq_turn_avg": MA(DAILY_TURNOVER_RATE, 20),  # 过20日换手率的均值
}

# 基础配置
DEFAULT_CONFIG = {
    "start_date": "2015-01-01",
    "end_date": "2025-07-01",
    "index_item": "000852.XSHG",
    "join_type": "inner",
    "force_rebuild": False,
    "cache_dir": "/Users/didi/KDCJ/factor_lib",  # 使用绝对路径，统一到根目录
}
