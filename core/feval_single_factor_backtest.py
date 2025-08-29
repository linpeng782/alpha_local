"""
因子回测评估模块
将因子数据转换为投资组合权重并进行回测分析
"""

import sys
import os

sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils import *
from factor_utils.path_manager import get_data_path
from alpha_local.core.factor_config import get_factor_config
from alpha_local.core.analyze_single_factor import get_stock_universe
import pandas as pd


def load_processed_factor(factor_name, index_item, neutralize, start_date, end_date):
    """
    从 processed 文件夹加载单个处理后的因子

    :param factor_name: 因子名称
    :param index_item: 指数代码
    :param neutralize: 是否中性化
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 因子DataFrame
    """

    # 从配置文件获取因子信息并直接加载因子
    print(f"✅从配置文件获取因子信息...")
    config = get_factor_config(factor_name, neutralize=neutralize)
    direction = config["direction"]

    # 使用get_data_path构建文件路径
    factor_file = get_data_path(
        "factor_processed",
        filename=f"{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}.pkl",
        index_item=index_item,
        neutralize=neutralize,
    )

    try:
        factor_df = pd.read_pickle(factor_file)
        print(f"✅加载因子 {factor_name} 成功")
        return direction, factor_df
    except FileNotFoundError:
        print(f"❌未找到因子文件: {factor_file}")
        raise
    except Exception as e:
        print(f"❌加载因子 {factor_name} 失败: {e}")
        raise


if __name__ == "__main__":

    start_date = "2015-01-01"
    end_date = "2025-07-01"
    index_item = "000985.XSHG"
    rebalance_days = 20

    stock_universe = get_stock_universe(start_date, end_date, index_item)
    universe_start = stock_universe.index[0].strftime("%F")
    universe_end = stock_universe.index[-1].strftime("%F")

    factor_name = "high_low_std_504"
    neutralize = True
    backtest_start_date = universe_start
    print(
        f"✅加载处理后的因子{factor_name}_{index_item}_{neutralize}_{universe_start}_{universe_end}..."
    )

    direction, processed_factor = load_processed_factor(
        factor_name=factor_name,
        index_item=index_item,
        neutralize=neutralize,
        start_date=universe_start,
        end_date=universe_end,
    )
    print("因子 shape:", processed_factor.shape)

    print(f"✅进行策略回测...")
    buy_list = get_buy_list(processed_factor, rank_n=50)
    df_weight = buy_list.div(buy_list.sum(axis=1), axis=0)
    df_weight = df_weight.shift(1).dropna(how="all")

    account_result = backtest(
        df_weight,
        rebalance_frequency=rebalance_days,
        backtest_start_date=backtest_start_date,
    )

    # 使用get_data_path生成路径，按指数、日期和中性化状态分类
    account_result_file = get_data_path(
        "account_result",
        start_date=backtest_start_date,
        end_date=end_date,
        factor_name=factor_name,
        index_item=index_item,
        direction=direction,
        neutralize=neutralize,
    )
    account_result.to_pickle(account_result_file)
    print(f"✅单因子策略结果已保存到: {account_result_file}")

    # 绩效分析并保存策略报告
    performance_cumnet, result = get_performance_analysis(
        account_result,
        direction,
        neutralize,
        benchmark_index=index_item,
        factor_name=factor_name,
        start_date=start_date,
        end_date=end_date,
    )
