"""
基础算子和工具函数定义
"""
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
