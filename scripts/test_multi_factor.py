import sys
import os

sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils import *
import pandas as pd


def load_processed_factors(factor_names, index_item, start_date, end_date):
    """
    从 processed 文件夹加载多个处理后的因子

    :param factor_names: 因子名称列表
    :param index_item: 指数代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 合并后的因子DataFrame字典
    """
    factors_dict = {}
    base_path = "/Users/didi/KDCJ/alpha_local/data/factor_lib/processed"

    for factor_name in factor_names:
        # 构建文件名
        filename = f"{factor_name}_{index_item}_{start_date}_{end_date}.pkl"
        file_path = os.path.join(base_path, filename)

        try:
            # 加载因子数据
            factor_df = pd.read_pickle(file_path)
            factors_dict[factor_name] = factor_df
            print(f"✅加载因子 {factor_name}: {factor_df.shape}")
        except FileNotFoundError:
            print(f"❌未找到因子文件: {filename}")
        except Exception as e:
            print(f"❌加载因子 {factor_name} 失败: {e}")

    return factors_dict


if __name__ == "__main__":
    # 用户指定的期望日期范围
    input_start_date = "2015-01-01"
    input_end_date = "2025-07-01"
    index_item = "000852.XSHG"

    # 需要的因子列表
    factor_names = ["cfoa_mrq_0", "corr_price_turn", "pe_ttm"]

    # 加载所有因子
    factors_dict = load_processed_factors(
        factor_names, index_item, input_start_date, input_end_date
    )

    
