import sys
import os

sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils import *
import pandas as pd
from factor_config import get_factor_for_test


def get_raw_factor(
    factor_name,
    factor_definition,
    index_item,
    direction,
    stock_universe,
):
    """
    获取原始因子数据：优先从因子库加载，不存在则计算并保存

    :param factor_name: 因子名称
    :param factor_definition: 因子定义表达式
    :param index_item: 指数代码
    :param direction: 因子方向（1或-1）
    :param stock_universe: 股票池
    :return: 原始因子DataFrame
    """

    universe_start = stock_universe.index[0].strftime("%F")
    universe_end = stock_universe.index[-1].strftime("%F")
    stock_list = stock_universe.columns.tolist()
    try:
        raw_path = get_data_path(
            "factor_raw",
            filename=f"{factor_name}_{index_item}_{direction}_{universe_start}_{universe_end}.pkl",
        )
        raw_factor = pd.read_pickle(raw_path)
        print(f"✅从因子库加载原始因子 {raw_path}")
    except:
        print(
            f"✅从因子库加载原始因子失败，开始计算因子{factor_name}_{index_item}_{direction}_{universe_start}_{universe_end}..."
        )
        raw_factor = (
            execute_factor(factor_definition, stock_list, universe_start, universe_end)
            * direction
        )
        raw_factor.index.names = ["datetime"]
        raw_factor.columns.names = ["order_book_id"]

        # 保存因子
        raw_path = get_data_path(
            "factor_raw",
            filename=f"{factor_name}_{index_item}_{direction}_{universe_start}_{universe_end}.pkl",
        )
        raw_factor.to_pickle(raw_path)
        print(f"✅raw_factor已保存到: {raw_path}")

    return raw_factor


def factor_factory(
    start_date,
    end_date,
    index_item,
    factor_name,
    factor_definition,
    direction,
    neutralize,
    rebalance_days,
    save_factor=True,
):
    """
    因子工厂函数：从因子定义到完整的因子测试流程

    :param start_date: 开始日期
    :param end_date: 结束日期
    :param index_item: 指数代码
    :param factor_name: 因子名称
    :param factor_definition: 因子定义表达式
    :param direction: 因子方向（1或-1）
    :param neutralize: 是否中性化,True/False
    :param rebalance_days: 换手周期列表
    :param save_factor: 是否保存因子
    :return: 处理后的因子、IC报告、分层回测结果
    """
    print(f"\n=== 开始因子测试: {factor_name} ===")

    # 1. 获取股票池 -> unstack
    print(f"✅获取股票池{index_item}_{start_date}_{end_date}...")
    stock_universe = INDEX_FIX(start_date, end_date, index_item)
    universe_start = stock_universe.index[0].strftime("%F")
    universe_end = stock_universe.index[-1].strftime("%F")

    # 2. 获取原始因子
    raw_factor = get_raw_factor(
        factor_name,
        factor_definition,
        index_item,
        direction,
        stock_universe,
    )

    # 3. 因子清洗
    print(
        f"✅因子清洗{factor_name}_{index_item}_{direction}_{neutralize}_{universe_start}_{universe_end}..."
    )
    processed_factor = preprocess_factor(
        raw_factor, stock_universe, index_item, neutralize
    )

    # 4. 保存因子数据
    if save_factor:
        processed_path = get_data_path(
            "factor_processed",
            filename=f"{factor_name}_{index_item}_{direction}_{neutralize}_{universe_start}_{universe_end}.pkl",
        )
        processed_factor.to_pickle(processed_path)
        print(f"✅processed_factor已保存到: {processed_path}")

    # 5. 计算IC
    print(f"✅因子IC分析...")
    ic_values, ic_report = calc_ic(
        processed_factor,
        index_item,
        direction,
        neutralize,
        rebalance_days,
        factor_name,
    )

    # 6. 分层回测
    print(f"✅因子分层回测...")
    return_group_hold, turnover_ratio = factor_layered_backtest(
        processed_factor,
        index_item,
        direction,
        neutralize,
        factor_name=factor_name,
        rebalance_days=rebalance_days,
    )


if __name__ == "__main__":

    start_date = "2015-01-01"
    end_date = "2025-07-01"
    index_item = "000985.XSHG"
    rebalance_days = 20
    backtest_start_date = start_date

    # 当前要测试的因子
    factor_name = "roe_ttm"
    direction = 1
    neutralize = True

    # 从配置文件获取匹配的因子信息
    config = get_factor_for_test(
        factor_name, direction=direction, neutralize=neutralize
    )
    factor_definition = config["definition"]

    # 因子工厂函数
    factor_factory(
        start_date=start_date,
        end_date=end_date,
        index_item=index_item,
        factor_name=factor_name,
        factor_definition=factor_definition,
        direction=direction,
        neutralize=neutralize,
        rebalance_days=rebalance_days,
    )
