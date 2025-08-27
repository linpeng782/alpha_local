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
        ("dp_ttm", 1, True),
        ("high_low_std_504", -1, False),
    ]

    # 加载所有因子
    factors_dict = load_processed_factors(
        factor_configs, index_item, start_date, end_date
    )

    # 两层筛选逻辑：小市值 → 低波动率
    market_cap = factors_dict["market_cap"]
    high_low_std_504 = factors_dict["high_low_std_504"]
    
    # 调试信息：查看每个因子的数据情况
    print(f"market_cap有效数据数量: {market_cap.notna().sum(axis=1).iloc[0]}只")
    print(f"high_low_std_504有效数据数量: {high_low_std_504.notna().sum(axis=1).iloc[0]}只")
    
    # 第一层筛选：从所有market_cap数据中选择最小的100只
    small_cap_mask = market_cap.rank(axis=1, ascending=False) <= 100
    print(f"第一层筛选后股票数量: {small_cap_mask.sum(axis=1).iloc[0]}只")
    
    
    # 第二层筛选：从小市值股票中选择波动率最小的50只
    high_low_std_filtered = high_low_std_504.mask(~small_cap_mask)
    print(f"筛选后有效数据数量: {high_low_std_filtered.notna().sum(axis=1).iloc[-1]}只")
    low_volatility_mask = high_low_std_filtered.rank(axis=1, ascending=False) <= 50
    print(f"第二层筛选后股票数量: {low_volatility_mask.sum(axis=1).iloc[-1]}只")

    # 最终筛选结果：小市值 + 低波动率的股票
    combo_factor = high_low_std_filtered.mask(~low_volatility_mask)

    # 将非NaN的因子值转换为1，形成等权重买入列表
    buy_list = combo_factor.notna().astype(int)
    buy_list = buy_list.replace(0, np.nan)  # 保持稀疏矩阵结构，提高回测系统兼容性

    # 计算每日的等权重权重矩阵
    df_weight = buy_list.div(buy_list.sum(axis=1), axis=0)
    df_weight = df_weight.shift(1).dropna(how="all")

    # 添加1月份和4月份空仓逻辑：小市值策略在这两个月份容易受财报披露和机构调仓影响
    # 1月份空仓：年报预告和机构调仓期
    january_mask = df_weight.index.month == 1
    january_data = df_weight.loc[january_mask]
    df_weight.loc[january_mask] = january_data.where(january_data.isna(), 0)
    
    # 4月份空仓：年报披露集中期，小市值股票风险较大
    april_mask = df_weight.index.month == 4
    april_data = df_weight.loc[april_mask]
    df_weight.loc[april_mask] = april_data.where(april_data.isna(), 0)

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
    
