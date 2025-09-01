import sys
import os
import pickle

sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils import *
from factor_utils.path_manager import get_data_path
from alpha_local.core.factor_config import get_factor_config
import pandas as pd
from alpha_local.core.feval_single_factor_analysis import (
    get_stock_universe,
    get_factor_backtest,
)


def format_market_cap_stats(stats_series):
    """将市值统计数据从科学计数法转换为易读格式"""

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

    formatted_stats = {}
    for key, value in stats_series.items():
        if key == "count":
            formatted_stats[key] = f"{value:.0f}"
        else:
            formatted_stats[key] = format_value(value)

    return formatted_stats


def load_processed_factors(factor_names, neutralize, index_item, start_date, end_date):
    """
    从 processed 文件夹加载处理后的因子，支持单个或多个因子

    :param factor_names: 因子名称或因子名称列表
    :param neutralize: 是否中性化
    :param index_item: 指数代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 单个因子返回DataFrame，多个因子返回字典
    """
    # 统一处理为列表格式
    if isinstance(factor_names, str):
        factor_names = [factor_names]
        return_single = True
    else:
        return_single = False

    factors_dict = {}

    for factor_name in factor_names:
        try:
            # 获取因子配置信息
            factor_info = get_factor_config(factor_name, neutralize=neutralize)
            direction = factor_info["direction"]

            # 构建文件名
            filename = f"{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}.pkl"

            # 使用统一路径管理生成文件路径
            file_path = get_data_path(
                "factor_processed",
                factor_name=factor_name,
                index_item=index_item,
                direction=direction,
                neutralize=neutralize,
                start_date=start_date,
                end_date=end_date,
                filename=filename,
            )

            # 加载因子数据
            factor_df = pd.read_pickle(file_path)
            factors_dict[factor_name] = factor_df
            print(f"✅加载因子: {factor_name} (中性化: {neutralize})")

        except FileNotFoundError:
            print(f"❌未找到因子文件: {factor_name}")
        except Exception as e:
            print(f"❌加载因子 {factor_name} 失败: {e}")

    # 根据输入类型返回结果
    if return_single:
        if len(factors_dict) == 1:
            return list(factors_dict.values())[0]
        else:
            return None
    else:
        print(f"\n📊成功加载 {len(factors_dict)} 个因子")
        return factors_dict


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

    # print(f"bp_lyr正因子数量: {positive_bp_mask.sum(axis=1)}")
    # print(f"eps正因子数量: {positvie_eps_mask.sum(axis=1)}")
    # print(f"bp_lyr和eps正因子数量: {positive_bp_eps.sum(axis=1)}")
    # print(f"roe_yoy正因子数量: {positive_roe_mask.sum(axis=1)}")

    positive_bp_eps_roe = positive_bp_eps & positive_roe_mask
    # print(f"bp_lyr和eps和roe_yoy正因子数量: {positive_bp_eps_roe.sum(axis=1)}")

    # 使用where保留三个因子都为正的股票的market_cap_3值
    market_cap_positive_filtered = factors_dict["market_cap_3"].where(
        positive_bp_eps_roe
    )

    # 计算过滤后每个截面的平均市值（三因子都为正的股票）
    avg_market_cap_positive = market_cap_positive_filtered.mean(axis=1, skipna=True)
    formatted_stats_positive = format_market_cap_stats(
        avg_market_cap_positive.describe()
    )
    print("三因子都为正股票的平均市值:")
    for key, value in formatted_stats_positive.items():
        print(f"  {key}: {value}")

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
