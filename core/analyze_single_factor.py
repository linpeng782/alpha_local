import sys
import os

sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils import *
from factor_utils.path_manager import get_data_path
import pandas as pd
from alpha_local.core.factor_config import get_factor_config


def get_stock_universe(start_date, end_date, index_item):
    """
    :param start_date: 开始日 -> str
    :param end_date: 结束日 -> str
    :param index_item: 指数代码 -> str
    :return stock_universe: 股票池 -> unstack
    """

    universe_name = f"{index_item}_{start_date}_{end_date}"
    try:
        print(f"✅从因子库加载stock_universe: {universe_name}")
        stock_universe = pd.read_pickle(
            get_data_path("stock_universe", filename=universe_name)
        )
    except:
        print(f"✅因子库加载失败,开始计算新的stock_universe: {universe_name}")
        stock_universe = INDEX_FIX(start_date, end_date, index_item)

        universe_path = get_data_path("stock_universe", filename=universe_name)
        # 保存stock_universe
        stock_universe.to_pickle(universe_path)
        print(f"✅stock_universe已保存到: {universe_path}")

    return stock_universe


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

    # 生成因子文件路径（只需要调用一次）
    raw_path = get_data_path(
        "factor_raw",
        filename=f"{factor_name}_{index_item}_{direction}_{universe_start}_{universe_end}.pkl",
        index_item=index_item,
    )

    try:
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

        # 保存因子（使用已生成的路径）
        raw_factor.to_pickle(raw_path)
        print(f"✅raw_factor已保存到: {raw_path}")

    print("原始因子 shape:", raw_factor.shape)

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
    layer_test=False,
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
    print(f"\n✅开始因子测试: {factor_name}_{index_item}_{direction}_{neutralize}✅")

    # 1. 获取股票池 -> unstack
    print(f"✅获取股票池{index_item}_{start_date}_{end_date}...")
    stock_universe = get_stock_universe(start_date, end_date, index_item)
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
    processed_factor = preprocess_raw_factor(
        factor_name,
        raw_factor,
        index_item,
        direction,
        neutralize,
        stock_universe,
    )

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

    if layer_test:
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
    layer_test = True

    # 当前要测试的因子
    factor_name = "high_low_std_504"
    neutralize = False

    # 从配置文件获取因子信息（简化版）
    config = get_factor_config(factor_name, neutralize=neutralize)
    factor_definition = config["definition"]
    direction = config["direction"]

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
        layer_test=layer_test,
    )
