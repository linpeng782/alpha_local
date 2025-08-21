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

    # 因子清洗不带中性化
    processed_factor = preprocess_factor_without_neutralization(
        raw_factor, stock_universe, index_item
    )

    print("开始分层回测...")
    # 先运行原有的分层回测函数
    return_group_hold, turnover_ratio = factor_layered_backtest(
        processed_factor, change_days, group_num, index_item, name=factor_name
    )

    print("开始测试新的三图合一可视化函数...")
    # 使用新的简洁绘图函数
    plot_factor_layered_analysis(
        return_group_hold, index_item, factor_name, stock_universe
    )

    print("三图合一测试完成！")
