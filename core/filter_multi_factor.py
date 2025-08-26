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
    # 用户指定的期望日期范围
    start_date = "2015-01-01"
    end_date = "2025-07-01"
    index_item = "000985.XSHG"
    rebalance_days = 20

    stock_universe = INDEX_FIX(start_date, end_date, index_item)

    # 因子配置：(factor_name, direction, neutralize)
    factor_configs = [
        ("market_cap", -1, False),
        ("high_low_std_504", -1, True),
    ]

    # 加载所有因子
    factors_dict = load_processed_factors(
        factor_configs, index_item, start_date, end_date
    )

    market_cap = factors_dict["market_cap"]
    high_low_std = factors_dict["high_low_std_504"]

    # 第一层筛选：获取最小市值的100只股票
    market_cap_mask = market_cap.rank(axis=1, ascending=False) <= 100

    # 第二层筛选：从小市值股票中筛选high_low_std最小的50只
    # 首先将high_low_std中非小市值股票的值设为NaN
    high_low_std_filtered = high_low_std.mask(~market_cap_mask)

    # 使用ascending=False降序排名，选择排名前50的股票
    high_low_std_mask = high_low_std_filtered.rank(axis=1, ascending=False) <= 50

    # 最终筛选结果：既是小市值又是low volatility的股票
    combo_factor = high_low_std_filtered.mask(~high_low_std_mask)

    # 将非NaN的因子值转换为1，形成等权重买入列表
    buy_list = combo_factor.notna().astype(int)
    buy_list = buy_list.replace(0, np.nan)  # 保持稀疏矩阵结构，提高回测系统兼容性
    # 计算每日的等权重权重矩阵
    df_weight = buy_list.div(buy_list.sum(axis=1), axis=0)
    df_weight = df_weight.shift(1).dropna(how="all")
    # 执行回测
    backtest_start_date = start_date
    account_result = backtest(
        df_weight, rebalance_frequency=rebalance_days, start_date=backtest_start_date
    )

    direction = -1
    neutralize = False
    performance_cumnet, result = get_performance_analysis(
        account_result,
        direction,
        neutralize,
        benchmark_index=index_item,
        factor_name="combo",
        stock_universe=stock_universe,
    )
    print(result)
