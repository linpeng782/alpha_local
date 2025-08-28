import sys
import os
import pickle

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

    # =============================================================================
    # 三因子策略配置区域 - 方便调整因子
    # =============================================================================

    # 第一个因子（固定）：小市值筛选
    FIRST_FACTOR = {
        'name': 'market_cap',
        'neutralize': False,
        'select_count': 200,  # 选择前200只
        'direction': -1,
    }

    SECOND_FACTOR = {
        "name": "dp_ttm",
        "neutralize": True,
        "select_count": 100,  # 从第一层筛选结果中选择前50只
        "direction": 1,
    }
    THIRD_FACTOR = {
        'name': 'high_low_std_504',
        'neutralize': False,
        'select_count': 50,   # 从第二层筛选结果中选择前50只
        'direction': -1,
    }

    # 因子配置列表（所有因子都使用direction=1，因为已经调整为值越大越好）
    factor_configs = [
        (FIRST_FACTOR['name'], FIRST_FACTOR['direction'], FIRST_FACTOR['neutralize']),
        (SECOND_FACTOR['name'], SECOND_FACTOR['direction'], SECOND_FACTOR['neutralize']),
        (THIRD_FACTOR['name'], THIRD_FACTOR['direction'], THIRD_FACTOR['neutralize']),
    ]

    print(f"三因子策略配置:")
    print(f"  第一层: {FIRST_FACTOR['name']} (选择{FIRST_FACTOR['select_count']}只)")
    print(f"  第二层: {SECOND_FACTOR['name']} (选择{SECOND_FACTOR['select_count']}只)")
    print(f"  第三层: {THIRD_FACTOR['name']} (选择{THIRD_FACTOR['select_count']}只)")
    print()

    # 加载所有因子
    factors_dict = load_processed_factors(
        factor_configs, index_item, start_date, end_date
    )

    # 获取因子数据
    first_factor_data = factors_dict[FIRST_FACTOR['name']]
    second_factor_data = factors_dict[SECOND_FACTOR['name']]
    third_factor_data = factors_dict[THIRD_FACTOR['name']]

    # 调试信息：查看每个因子的数据情况
    print(f"{FIRST_FACTOR['name']}有效数据数量: {first_factor_data.notna().sum(axis=1).iloc[0]}只")
    print(f"{SECOND_FACTOR['name']}有效数据数量: {second_factor_data.notna().sum(axis=1).iloc[0]}只")
    print(f"{THIRD_FACTOR['name']}有效数据数量: {third_factor_data.notna().sum(axis=1).iloc[0]}只")

    # 第一层筛选：按第一个因子筛选（统一使用ascending=False，从大到小）
    first_mask = first_factor_data.rank(axis=1, ascending=False) <= FIRST_FACTOR['select_count']
    print(f"第一层筛选后第一行股票数量: {first_mask.sum(axis=1).iloc[0]}只")
    print(f"第一层筛选后最后一行股票数量: {first_mask.sum(axis=1).iloc[-1]}只")
    
    # 检查两个因子的数据重叠情况
    overlap_first = (first_factor_data.notna() & second_factor_data.notna()).sum(axis=1)
    print(f"两个因子数据重叠数量 - 第一行: {overlap_first.iloc[0]}只")
    print(f"两个因子数据重叠数量 - 最后一行: {overlap_first.iloc[-1]}只")

    # 第二层筛选：从第一层结果中按第二个因子筛选（统一使用ascending=False，从大到小）
    second_factor_filtered = second_factor_data.where(first_mask)
    print(f"第二层筛选前第一行股票数量: {second_factor_filtered.notna().sum(axis=1).iloc[0]}只")
    print(f"第二层筛选前最后一行股票数量: {second_factor_filtered.notna().sum(axis=1).iloc[-1]}只")
    second_mask = second_factor_filtered.rank(axis=1, ascending=False) <= SECOND_FACTOR['select_count']
    print(f"第二层筛选后第一行股票数量: {second_mask.sum(axis=1).iloc[0]}只")
    print(f"第二层筛选后最后一行股票数量: {second_mask.sum(axis=1).iloc[-1]}只")

    # 第三层筛选：从前两层筛选结果中按第三个因子筛选（统一使用ascending=False，从大到小）
    # 需要同时满足前两个因子条件
    combined_mask = first_mask & second_mask
    print(f"组合mask后股票数量: {combined_mask.sum(axis=1).iloc[0]}只")

    third_factor_filtered = third_factor_data.where(combined_mask)
    print(f"第三层筛选前有效数据数量: {third_factor_filtered.notna().sum(axis=1).iloc[0]}只")

    third_mask = third_factor_filtered.rank(axis=1, ascending=False) <= THIRD_FACTOR['select_count']
    print(f"第三层筛选后股票数量: {third_mask.sum(axis=1).iloc[0]}只")
    print(f"第三层筛选后最后一行股票数量: {third_mask.sum(axis=1).iloc[-1]}只")

    # 最终筛选结果：同时满足三个因子条件的股票
    combo_factor = third_factor_filtered.where(third_mask)

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

    # 2月份空仓：年报披露集中期，小市值股票风险较大
    # february_mask = df_weight.index.month == 2
    # february_data = df_weight.loc[february_mask]
    # df_weight.loc[february_mask] = february_data.where(february_data.isna(), 0)

    # 执行回测
    backtest_start_date = start_date
    account_result = backtest(
        df_weight,
        rebalance_frequency=rebalance_days,
        backtest_start_date=backtest_start_date,
    )

    direction = -1
    neutralize = False
    factor_name = "combo_3"
    result_dir = "/Users/didi/KDCJ/alpha_local/data/account_result"
    file_name = f"{backtest_start_date}_{end_date}_{factor_name}_account_result.pkl"
    result_file = os.path.join(result_dir, file_name)
    account_result.to_pickle(result_file)
    print(f"✅小市值三因子策略结果已保存到: {result_file}")

    performance_cumnet, result = get_performance_analysis(
        account_result,
        direction,
        neutralize,
        benchmark_index=index_item,
        factor_name=factor_name,
        start_date=start_date,
        end_date=end_date,
    )
