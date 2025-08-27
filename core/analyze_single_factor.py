import sys
import os

sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils import *
import pandas as pd
from factor_config import get_factor_for_test


def factor_factory(
    start_date,
    end_date,
    index_item,
    factor_name,
    factor_definition,
    rebalance_days,
    direction,
    neutralize,
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
    :param rebalance_days: 换手周期列表
    :param group_num: 分组数量
    :param rank_n: 选股数量
    :param save_factor: 是否保存因子
    :return: 处理后的因子、IC报告、分层回测结果
    """
    print(f"\n=== 开始因子测试: {factor_name} ===")

    # 1. 获取股票池 -> unstack
    stock_universe = INDEX_FIX(start_date, end_date, index_item)
    stock_list = stock_universe.columns.tolist()

    # 2. 计算原始因子
    try:
        raw_path = get_data_path(
            "factor_raw",
            filename=f"{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}.pkl",
        )
        raw_factor = pd.read_pickle(raw_path)
        print(f"✅从因子库加载因子 {raw_path}")
    except:
        print(
            f"✅从因子库加载因子失败，开始计算因子{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}..."
        )
        raw_factor = (
            execute_factor(factor_definition, stock_list, start_date, end_date)
            * direction
        )
        raw_factor.index.names = ["datetime"]
        raw_factor.columns.names = ["order_book_id"]

    # 3. 因子清洗
    if neutralize:
        print(
            f"✅清洗因子{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}..."
        )
        processed_factor = preprocess_factor(raw_factor, stock_universe, index_item)
    else:
        print(
            f"✅清洗因子{factor_name}_{index_item}_{direction}_{neutralize}_{start_date}_{end_date}..."
        )
        processed_factor = preprocess_factor_without_neutralization(
            raw_factor, stock_universe, index_item
        )

    # 4. 计算IC
    print(f"✅因子IC分析...")
    ic_values, ic_report = calc_ic(
        processed_factor,
        index_item,
        direction,
        neutralize,
        rebalance_days,
        factor_name,
    )

    # 5. 分层回测
    print(f"✅因子分层回测...")
    return_group_hold, turnover_ratio = factor_layered_backtest(
        processed_factor, index_item
    )

    # 6. 绘制分层回测结果
    print(f"✅生成分层分析图表...")
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


if __name__ == "__main__":

    start_date = "2015-01-01"
    end_date = "2025-07-01"
    index_item = "000985.XSHG"
    rebalance_days = 20
    backtest_start_date = start_date

    # 当前要测试的因子
    factor_name = "market_cap"
    direction = -1
    neutralize = False

    # 从配置文件获取匹配的因子信息
    config = get_factor_for_test(
        factor_name, direction=direction, neutralize=neutralize
    )
    factor_definition = config["definition"]

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
    )
