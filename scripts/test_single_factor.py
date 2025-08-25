import sys
import os

sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils import *
import pandas as pd


if __name__ == "__main__":
    # 用户指定的期望日期范围
    start_date = "2015-01-01"
    end_date = "2025-07-01"
    index_item = "000985.XSHG"
    neutralize = True
    rebalance_days = [20]
    group_num = 10
    direction = -1

    stock_universe = INDEX_FIX(start_date, end_date, index_item)
    stock_list = stock_universe.columns.tolist()

    factor_name = "high_low_std"

    HIGH = Factor("high")
    LOW = Factor("low")
    factor_definition = STD(HIGH / LOW, 20)

    print(f"计算因子{factor_name}_{index_item}_{start_date}_{end_date}...")
    raw_factor = direction * execute_factor(
        factor_definition, stock_list, start_date, end_date
    )
    raw_factor.index.names = ["datetime"]

    if neutralize:
        # 因子清洗带中性化
        print(f"因子清洗{factor_name}_{index_item}_{start_date}_{end_date}带中性化...")
        processed_factor = preprocess_factor(raw_factor, stock_universe, index_item)
    else:
        # 因子清洗不带中性化
        print(
            f"因子清洗{factor_name}_{index_item}_{start_date}_{end_date}不带中性化..."
        )
        processed_factor = preprocess_factor_without_neutralization(
            raw_factor, stock_universe, index_item
        )

    # 计算IC
    ic_values, ic_report = calc_ic(
        processed_factor, rebalance_days, index_item, factor_name
    )
    # 分层回测
    return_group_hold, _ = factor_layered_backtest(
        processed_factor,
        group_num,
        index_item,
        name=factor_name,
    )
    # 绘制分层回测结果
    plot_factor_layered_analysis(
        return_group_hold,
        index_item,
        factor_name,
        stock_universe,
    )

    # 生成买卖信号并完成回测
    buy_list = get_buy_list(processed_factor, rank_n=100)
    df_weight = buy_list.div(buy_list.sum(axis=1), axis=0)
    df_weight = df_weight.shift(1).dropna(how="all")
    account_result = backtest(df_weight)
    # 绩效分析并保存策略报告
    performance_cumnet, result = get_performance_analysis(
        account_result,
        benchmark_index=index_item,
        factor_name=factor_name,
        stock_universe=stock_universe,
    )

    # 使用统一的路径管理函数
    raw_path = get_data_path(
        "factor_raw",
        filename=f"{factor_name}_{index_item}_{start_date}_{end_date}.pkl",
    )
    processed_path = get_data_path(
        "factor_processed",
        filename=f"{factor_name}_{index_item}_{start_date}_{end_date}.pkl",
    )
    raw_factor.to_pickle(raw_path)
    processed_factor.to_pickle(processed_path)
    print(f"raw_factor已保存到: {raw_path}")
    print(f"processed_factor已保存到: {processed_path}")
