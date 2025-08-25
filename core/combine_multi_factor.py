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
            print(
                f"✅加载因子 {factor_name} (direction={direction}, neutralize={neutralize}): {factor_df.shape}"
            )
        except FileNotFoundError:
            print(f"❌未找到因子文件: {filename}")
        except Exception as e:
            print(f"❌加载因子 {factor_name} 失败: {e}")

    return factors_dict


if __name__ == "__main__":
    # 用户指定的期望日期范围
    start_date = "2015-01-01"
    end_date = "2025-07-01"
    index_item = "000985.XSHG"

    # 因子配置：(factor_name, direction, neutralize)
    factor_configs = [
        ("market_cap", -1, True),
        ("high_low_std", -1, False),
    ]

    # 加载所有因子
    factors_dict = load_processed_factors(
        factor_configs, index_item, start_date, end_date
    )

    print()
    breakpoint()
