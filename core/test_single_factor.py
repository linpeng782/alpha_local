import sys
import os

sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils import *
import pandas as pd
from factor_research_config import get_factor_for_test


def factor_factory(
    start_date,
    end_date,
    index_item,
    factor_name,
    factor_definition,
    rebalance_days,
    direction,
    neutralize,
    run_backtest=False,
    save_factor=True,
):
    """
    因子工厂函数：从因子定义到完整的因子测试流程

    :param factor_name: 因子名称
    :param factor_definition: 因子定义表达式
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param index_item: 指数代码
    :param direction: 因子方向（1或-1）
    :param neutralize: 是否中性化
    :param run_backtest: 是否运行回测
    :param rebalance_days: 换手周期列表
    :param group_num: 分组数量
    :param rank_n: 选股数量
    :param save_factor: 是否保存因子
    :return: 处理后的因子、IC报告、分层回测结果
    """
    print(f"\n=== 开始因子测试: {factor_name} ===")

    # 1. 获取股票池
    stock_universe = INDEX_FIX(start_date, end_date, index_item)
    stock_list = stock_universe.columns.tolist()

    # 2. 计算原始因子
    print(
        f"计算因子{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}..."
    )
    raw_factor = direction * execute_factor(
        factor_definition, stock_list, start_date, end_date
    )
    raw_factor.index.names = ["datetime"]
    raw_factor.columns.names = ["order_book_id"]

    # 3. 因子清洗
    if neutralize:
        print(
            f"清洗因子{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}..."
        )
        processed_factor = preprocess_factor(raw_factor, stock_universe, index_item)
    else:
        print(
            f"清洗因子{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}..."
        )
        processed_factor = preprocess_factor_without_neutralization(
            raw_factor, stock_universe, index_item
        )

    # 4. 计算IC
    print(f"因子IC分析...")
    ic_values, ic_report = calc_ic(
        processed_factor,
        index_item,
        direction,
        neutralize,
        rebalance_days,
        factor_name,
    )

    # 5. 分层回测
    print(f"因子分层回测...")
    return_group_hold, turnover_ratio = factor_layered_backtest(
        processed_factor, index_item
    )

    # 6. 绘制分层回测结果
    print(f"生成分层分析图表...")
    plot_factor_layered_analysis(
        return_group_hold,
        index_item,
        direction,
        neutralize,
        rebalance_days=rebalance_days,
        stock_universe=stock_universe,
        factor_name=factor_name,
    )

    # 7. 保存因子数据
    if save_factor:
        raw_path = get_data_path(
            "factor_raw",
            filename=f"{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}.pkl",
        )
        processed_path = get_data_path(
            "factor_processed",
            filename=f"{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}.pkl",
        )
        raw_factor.to_pickle(raw_path)
        processed_factor.to_pickle(processed_path)
        print(f"raw_factor已保存到: {raw_path}")
        print(f"processed_factor已保存到: {processed_path}")

    # 8. 策略回测（可选）
    if run_backtest:
        print(f"进行策略回测...")
        buy_list = get_buy_list(processed_factor, rank_n=100)
        df_weight = buy_list.div(buy_list.sum(axis=1), axis=0)
        df_weight = df_weight.shift(1).dropna(how="all")
        account_result = backtest(df_weight, rebalance_frequency=rebalance_days)

        # 绩效分析并保存策略报告
        performance_cumnet, result = get_performance_analysis(
            account_result,
            direction,
            neutralize,
            benchmark_index=index_item,
            factor_name=factor_name,
            stock_universe=stock_universe,
        )

        print(result)

    print(f"=== 因子测试完成: {factor_name} ===\n")


if __name__ == "__main__":

    start_date = "2015-01-01"
    end_date = "2025-07-01"
    index_item = "000985.XSHG"
    rebalance_days = 20
    run_backtest = False

   

    # 当前要测试的因子（修改这里来切换因子）
    factor_name = "high_low_std_504"  # 可选: high_low_std_504, market_cap

    # 从配置文件获取因子信息
    config = get_factor_for_test(factor_name)
    factor_definition = config["definition"]
    direction = config["direction"]
    neutralize = config["neutralize"]

    # 打印因子信息
    print(f"\n正在测试因子: {factor_name}")

    factor_factory(
        start_date=start_date,
        end_date=end_date,
        index_item=index_item,
        factor_name=factor_name,
        factor_definition=factor_definition,
        rebalance_days=rebalance_days,
        direction=direction,
        neutralize=neutralize,
        run_backtest=run_backtest,
    )
