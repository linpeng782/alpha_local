"""
因子回测评估模块
将因子数据转换为投资组合权重并进行回测分析
"""

import sys
import os

sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils import *
import pandas as pd


def load_processed_factors(factor_configs, index_item, start_date, end_date):
    """
    从 processed 文件夹加载多个处理后的因子

    :param factor_configs: 因子配置列表，每个元素为 (factor_name, direction, neutralize)
    :param index_item: 指数代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 因子DataFrame字典
    """
    factors_dict = {}
    base_path = "/Users/didi/KDCJ/alpha_local/data/factor_lib/processed"

    for factor_name, direction, neutralize in factor_configs:
        # 构建文件名
        filename = f"{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}.pkl"
        file_path = os.path.join(base_path, filename)

        try:
            # 加载因子数据
            factor_df = pd.read_pickle(file_path)
            factors_dict[factor_name] = factor_df
            print(f"✅加载因子 {filename}")
        except FileNotFoundError:
            print(f"❌未找到因子文件: {filename}")
        except Exception as e:
            print(f"❌加载因子 {filename} 失败: {e}")

    return factors_dict


if __name__ == "__main__":

    start_date = "2015-01-01"
    end_date = "2025-07-01"
    index_item = "000985.XSHG"
    rebalance_days = 20
    backtest_start_date = start_date

    stock_universe = INDEX_FIX(start_date, end_date, index_item)

    factor_name = "market_cap"
    direction = -1
    neutralize = False

    factor_configs = [(factor_name, direction, neutralize)]

    print(f"✅加载因子...")
    factors_dict = load_processed_factors(
        factor_configs, index_item, start_date, end_date
    )
    processed_factor = factors_dict[factor_name]

    print(f"✅进行策略回测...")
    buy_list = get_buy_list(processed_factor, rank_n=50)
    df_weight = buy_list.div(buy_list.sum(axis=1), axis=0)
    df_weight = df_weight.shift(1).dropna(how="all")
    account_result = backtest(
        df_weight,
        rebalance_frequency=rebalance_days,
        backtest_start_date=backtest_start_date,
    )
    
    
    result_dir = "/Users/didi/KDCJ/alpha_local/data/account_result"
    result_file = os.path.join(result_dir, "single_factor_backtest_result.pkl")
    account_result.to_pickle(result_file)
    print(f"✅单因子策略结果已保存到: {result_file}")
    
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
