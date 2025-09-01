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


# 分域研究：提取roe_yoy的特定分组
def extract_factor_groups(factor_data, group_numbers, total_groups=10):
    """
    从因子中提取特定分组的股票

    :param factor_data: 因子数据DataFrame
    :param group_numbers: 要提取的分组编号列表，如[7,8,9]
    :param total_groups: 总分组数，默认10
    :return: 提取的股票mask
    """
    # 计算因子排名（升序，NaN值排在最后）
    factor_rank = factor_data.rank(axis=1, method="first", na_option="bottom")

    # 计算每日有效股票数量
    valid_count = factor_data.notna().sum(axis=1)

    # 为每个日期创建分组mask
    group_mask = pd.DataFrame(
        False, index=factor_data.index, columns=factor_data.columns
    )

    for date in factor_data.index:
        daily_valid_count = valid_count[date]
        if daily_valid_count == 0:
            continue

        # 计算分组边界
        group_size = daily_valid_count / total_groups

        for group_num in group_numbers:
            # 计算该组的排名范围
            start_rank = (group_num - 1) * group_size + 1
            end_rank = group_num * group_size

            # 找到该组的股票
            daily_rank = factor_rank.loc[date]
            group_stocks = (daily_rank >= start_rank) & (daily_rank <= end_rank)
            group_mask.loc[date] = group_mask.loc[date] | group_stocks

    return group_mask


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
        "high_low_std_504",
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
    roe_yoy = factors_dict["roe_yoy"]
    high_low_std_504 = factors_dict["high_low_std_504"]
    turnover_std_20 = factors_dict["turnover_std_20"]
    market_cap_3 = factors_dict["market_cap_3"]

    bp_groups = extract_factor_groups(bp_lyr, [7, 8, 9, 10])
    roe_groups = extract_factor_groups(roe_yoy, [6, 7, 8, 9])
    bp_roe_groups = bp_groups & roe_groups
    market_cap_filtered = market_cap_3.where(bp_roe_groups)
    print_cap_stats(market_cap_filtered, "筛选后股票市值分布")

    market_cap_mask = market_cap_3.rank(axis=1, ascending=False) <= 1000
    market_cap_filtered = market_cap_filtered.where(market_cap_mask)
    print_cap_stats(market_cap_filtered, "市值最小的前1000只股票市值分布")

    turnover_filtered = turnover_std_20.where(market_cap_filtered.notna())
    turnover_mask = turnover_std_20.rank(axis=1, ascending=False) <= 1000
    turnover_filtered = turnover_filtered.where(turnover_mask)
    print_cap_stats(turnover_filtered, "最终选股市值分布")

    factor_name = "combo3_group_division"
    direction = "long"
    neutralize = False
    rebalance_days = 5
    buy_rank = 50
    # 因子回测
    get_factor_backtest(
        processed_factor=market_cap_filtered,
        factor_name=factor_name,
        index_item=index_item,
        direction=direction,
        neutralize=neutralize,
        start_date=start_date,
        end_date=end_date,
        rebalance_days=rebalance_days,
        rank_n=buy_rank,
    )
