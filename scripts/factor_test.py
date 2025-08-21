import sys
import os

sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils import *
import pandas as pd


if __name__ == "__main__":
    start_date = "2015-01-01"
    end_date = "2025-07-01"
    index_item = "000985.XSHG"

    change_days = 20
    group_num = 10

    stock_universe = INDEX_FIX(start_date, end_date, index_item)
    stock_list = stock_universe.columns.tolist()
    date_list = stock_universe.index.tolist()
    factor_name = "market_cap"

    factor_definition = Factor(factor_name)
    print("计算因子...")
    raw_factor = (
        execute_factor(factor_definition, stock_list, start_date, end_date) * -1
    )
    raw_factor.index.names = ["datetime"]

    # 因子清洗带中性化
    # processed_factor = preprocess_factor(
    #     raw_factor, stock_universe, index_item
    # )

    # 因子清洗不带中性化
    processed_factor = preprocess_factor_without_neutralization(
        raw_factor, stock_universe, index_item
    )

    ic, ic_report = calc_ic(processed_factor, change_days, index_item, factor_name)
    

    return_group_hold, _ = factor_layered_backtest(
        processed_factor, change_days, group_num, index_item, name=factor_name
    )

    buy_list = get_buy_list(processed_factor, rank_n=100)
    df_weight = buy_list.div(buy_list.sum(axis=1), axis=0)
    df_weight = df_weight.shift(1).dropna(how="all")
    account_result = backtest(df_weight)
    performance_cumnet, result = get_performance_analysis(
        account_result, benchmark_index=index_item
    )

    # 使用统一的路径管理函数
    raw_path = get_data_path(
        "factor_raw", 
        filename=f"{factor_name}_{index_item}_{start_date}_{end_date}.pkl"
    )
    processed_path = get_data_path(
        "factor_processed", 
        filename=f"{factor_name}_{index_item}_{start_date}_{end_date}.pkl"
    )
    raw_factor.to_pickle(raw_path)
    processed_factor.to_pickle(processed_path)
