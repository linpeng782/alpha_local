import sys
import os
import pickle

sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils import *
from factor_utils.path_manager import get_data_path, load_processed_factors
from alpha_local.core.factor_config import get_factor_config
import pandas as pd
from alpha_local.core.feval_single_factor_analysis import (
    get_stock_universe,
    get_factor_backtest,
)


def print_cap_stats(market_cap_data, title="市值统计"):
    """打印市值统计信息（易读格式）"""

    def format_value(value):
        if pd.isna(value):
            return "NaN"

        # 转换为正数（因为market_cap_3是负值）
        abs_value = abs(value)

        if abs_value >= 1e12:  # 万亿
            return f"{abs_value/1e12:.2f}万亿"
        elif abs_value >= 1e8:  # 亿
            return f"{abs_value/1e8:.1f}亿"
        elif abs_value >= 1e4:  # 万
            return f"{abs_value/1e4:.1f}万"
        else:
            return f"{abs_value:.0f}"

    # 打印股票数量统计
    stock_counts = market_cap_data.notna().sum(axis=1)
    print(f"截面非空股票数量: {stock_counts.describe()}")

    # 计算市值统计信息
    stats = market_cap_data.stack().describe()

    print(f"{title}:")
    for key, value in stats.items():
        if key == "count":
            formatted_value = f"{value:.0f}"
        else:
            formatted_value = format_value(value)
        print(f"  {key}: {formatted_value}")
    print()


if __name__ == "__main__":
    # 用户指定的期望日期范围
    start_date = "2015-01-01"
    end_date = "2025-07-01"
    index_item = "000985.XSHG"
    rebalance_days = 5

    stock_universe = get_stock_universe(start_date, end_date, index_item)
    universe_start = stock_universe.index[0].strftime("%F")
    universe_end = stock_universe.index[-1].strftime("%F")

    neutralize = False

    # 批量加载多个因子
    factor_names = [
        "bp_lyr",
        "eps",
        "roe_yoy",
        "turnover_std_20",
        "market_cap_3",
    ]
    factors_dict = load_processed_factors(
        factor_names=factor_names,
        neutralize=neutralize,
        index_item=index_item,
        start_date=universe_start,
        end_date=universe_end,
    )

    bp_lyr = factors_dict["bp_lyr"]
    eps = factors_dict["eps"]
    roe_yoy = factors_dict["roe_yoy"]

    positive_bp_mask = bp_lyr > 0
    positvie_eps_mask = eps > 0
    positive_roe_mask = roe_yoy > 0
    positive_bp_eps = positive_bp_mask & positvie_eps_mask
    positive_bp_eps_roe = positive_bp_eps & positive_roe_mask

    # 使用where保留三个因子都为正的股票的market_cap_3值
    market_cap_positive_filtered = factors_dict["market_cap_3"].where(
        positive_bp_eps_roe
    )
    print_cap_stats(market_cap_positive_filtered, "三因子都为正股票的平均市值")

    # 在三因子都为正的股票中，选择市值最小的前cap_rank只股票
    cap_rank = 1000
    market_cap_mask = (
        market_cap_positive_filtered.rank(axis=1, ascending=False) <= cap_rank
    )
    market_cap_filtered = market_cap_positive_filtered.where(market_cap_mask)

    # 计算前cap_rank只小市值股票的平均市值
    avg_market_cap_rank = market_cap_filtered.mean(axis=1, skipna=True)
    formatted_stats_rank = format_market_cap_stats(avg_market_cap_rank.describe())
    print(f"前{cap_rank}只市值股票的平均市值:")
    for key, value in formatted_stats_rank.items():
        print(f"  {key}: {value}")

    # 在前cap_rank只小市值股票中，选择换手率最小的前turnover_rank只股票
    turnover_rank = 1000
    turnover_mask = (
        factors_dict["turnover_std_20"].rank(axis=1, ascending=False) <= turnover_rank
    )
    market_cap_turnover_filtered = market_cap_filtered.where(turnover_mask)
    print(
        "经过turnover过滤后的股票数量: ",
        market_cap_turnover_filtered.notna().sum(axis=1).describe(),
    )

    # 计算经过turnover过滤后股票的平均市值
    avg_market_cap_turnover = market_cap_turnover_filtered.mean(axis=1, skipna=True)
    formatted_stats_turnover = format_market_cap_stats(
        avg_market_cap_turnover.describe()
    )
    print(f"经过turnover过滤后的股票的平均市值:")
    for key, value in formatted_stats_turnover.items():
        print(f"  {key}: {value}")

    # 1月份空仓
    # january_mask = market_cap_turnover_filtered.index.month == 1
    # january_data = market_cap_turnover_filtered.loc[january_mask]
    # market_cap_turnover_filtered.loc[january_mask] = january_data.where(
    #     january_data.isna(), 0
    # )

    factor_name = "combo3_turnover_rebalance_5_january_out"
    direction = "long"
    neutralize = False
    rebalance_days = 5
    # 因子回测
    get_factor_backtest(
        processed_factor=market_cap_turnover_filtered,
        factor_name=factor_name,
        index_item=index_item,
        direction=direction,
        neutralize=neutralize,
        start_date=start_date,
        end_date=end_date,
        rebalance_days=rebalance_days,
        rank_n=50,
    )
