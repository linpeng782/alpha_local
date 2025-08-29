import sys
import os

sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils import *
from factor_utils.path_manager import get_data_path
import pandas as pd
from alpha_local.core.factor_config import get_factor_config
from alpha_local.core.feval_single_factor_analysis import get_stock_universe


def load_ic_dataframe(index_item, neutralize, ic_df_folder_path=None):
    """
    从指定的IC_df文件夹中读取所有IC值文件，合并为一个DataFrame

    :param index_item: 指数代码
    :param neutralize: 中性化状态 (True/False)
    :param ic_df_folder_path: IC_df文件夹路径，默认为None时自动生成
    :return: 合并后的IC DataFrame，列名为因子名称
    """
    import glob
    import re

    # 如果没有指定路径，则自动生成
    if ic_df_folder_path is None:
        neutralize_folder = "True" if neutralize else "False"
        ic_df_folder_path = (
            f"/Users/didi/KDCJ/alpha_local/data/IC_df/{index_item}/{neutralize_folder}"
        )

    print(f"✅从文件夹读取IC数据: {ic_df_folder_path}")

    # 查找所有IC值文件
    ic_files = glob.glob(os.path.join(ic_df_folder_path, "*_IC_values.pkl"))

    if not ic_files:
        print(f"⚠️ 在路径 {ic_df_folder_path} 中未找到IC文件")
        return pd.DataFrame()

    print(f"✅找到 {len(ic_files)} 个IC文件")

    ic_dataframes = {}

    for ic_file in ic_files:
        try:
            # 从文件名中提取因子名称
            filename = os.path.basename(ic_file)
            # 使用正则表达式提取因子名称（第一个下划线之前的部分）
            factor_match = re.match(r"^([^_]+(?:_[^_]+)*?)_\d{6}\.XSHG", filename)
            if factor_match:
                factor_name = factor_match.group(1)
            else:
                # 如果正则匹配失败，使用文件名的前部分
                factor_name = filename.split("_")[0]

            # 读取IC值数据
            ic_series = pd.read_pickle(ic_file)
            ic_dataframes[factor_name] = ic_series

            print(f"✅读取因子 {factor_name}: {len(ic_series)} 个数据点")

        except Exception as e:
            print(f"⚠️ 读取文件 {ic_file} 失败: {e}")
            continue

    if not ic_dataframes:
        print("⚠️ 没有成功读取任何IC数据")
        return pd.DataFrame()

    # 合并所有IC数据为一个DataFrame
    ic_df = pd.DataFrame(ic_dataframes)

    print(f"✅IC数据合并完成: {ic_df.shape}")
    print(f"✅因子列表: {list(ic_df.columns)}")

    return ic_df


if __name__ == "__main__":

    start_date = "2015-01-01"
    end_date = "2025-07-01"
    index_item = "000985.XSHG"
    neutralize = False  # 指定中性化状态

    # 读取指定文件夹下的所有IC数据
    ic_df = load_ic_dataframe(index_item, neutralize)

    if not ic_df.empty:
        factor_name = ic_df.columns.tolist()

        # 生成因子IC相关性热力图
        hot_corr_path = hot_corr(
            factor_name, ic_df, index_item, neutralize, start_date, end_date
        )
