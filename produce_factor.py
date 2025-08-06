import scipy as sp
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from pathlib import Path

from tqdm import *
import KDCJ

import warnings

warnings.filterwarnings("ignore")


def get_pre_trade_day(df_date, date, n):
    return df_date[df_date.index(pd.Timestamp(date)) - n]


def calculate_single_factor(factor_name, market_data_slice):
    """
    计算单个因子值

    参数:
        factor_name: 因子名称，如'alpha_001'
        market_data_slice: KDCJ_003实例，包含市场数据切片

    返回:
        因子计算结果
    """
    try:
        # 使用getattr安全地获取因子计算方法
        factor_method = getattr(market_data_slice, factor_name)
        return factor_method()
    except AttributeError:
        print(f"警告: 因子方法 {factor_name} 不存在")
        return None
    except Exception as e:
        print(f"计算因子 {factor_name} 时发生错误: {str(e)}")
        return None


def create_market_data_instance(data_slice_idx):
    """
    创建市场数据实例

    参数:
        data_slice_idx: 数据切片的起始索引

    返回:
            KDCJ_003实例
    """
    return KDCJ.KDCJ_003(
        open.iloc[data_slice_idx : data_slice_idx + n],
        high.iloc[data_slice_idx : data_slice_idx + n],
        low.iloc[data_slice_idx : data_slice_idx + n],
        close.iloc[data_slice_idx : data_slice_idx + n],
        prev_close.iloc[data_slice_idx : data_slice_idx + n],
        volume.iloc[data_slice_idx : data_slice_idx + n],
        amount.iloc[data_slice_idx : data_slice_idx + n],
        avg_price.iloc[data_slice_idx : data_slice_idx + n],
    )


if __name__ == "__main__":

    DATA_DIR = Path("./alpha_local/database")

    start_date = "2020-01-02"
    end_date = "2022-01-01"
    n = 5  # 调仓周期

    df = pd.read_pickle(DATA_DIR / "20140101_20221214_全A_日级别.pkl")
    whole_trade_datetime = sorted(list(set(df.index.get_level_values(1))))

    df = (
        df.reset_index()
        .set_index(["date"])
        .sort_index()
        .loc[get_pre_trade_day(whole_trade_datetime, start_date, 5) : end_date]
        .reset_index()
        .set_index(["order_book_id", "date"])
        .sort_index()  # 确保order_book_id和date都按顺序排列
    )
    # 只保留三只股票进行调试：000001、000002、000004
    #debug_stocks = ['000001.XSHE', '000002.XSHE', '000004.XSHE']
    #df = df[df.index.get_level_values('order_book_id').isin(debug_stocks)]

    open = df.open.unstack("order_book_id")
    high = df.high.unstack("order_book_id")
    low = df.low.unstack("order_book_id")
    close = df.close.unstack("order_book_id")
    prev_close = df.prev_close.unstack("order_book_id")
    volume = df.volume.unstack("order_book_id")
    amount = df.total_turnover.unstack("order_book_id")
    avg_price = amount.div(volume, fill_value=0)

    # 定义要计算的因子名称
    #alpha_names = ["alpha_{}".format(str(i).rjust(3, "0")) for i in range(1, 4)]
    alpha_names = ["alpha_001"]
    date_list = sorted(open.index.tolist())
    factor_get = []

    # 遍历每个因子进行计算
    for factor_name in alpha_names:
        print(f"正在计算因子: {factor_name}")
        factor_results = pd.DataFrame()

        try:
            # 滚动窗口计算因子
            for i in tqdm(range(0, len(open.index) - n), desc=f"计算{factor_name}"):
                # 创建市场数据实例
                market_data = create_market_data_instance(i)

                # 计算因子值
                factor_value = calculate_single_factor(factor_name, market_data)

                if factor_value is not None:
                    # 将结果添加到DataFrame
                    factor_df = pd.DataFrame(
                        factor_value, columns=[date_list[i + n - 1]]
                    )
                    factor_results = pd.concat([factor_results, factor_df], axis=1)

            alpha = factor_results.T
            print(alpha)

        except Exception as e:
            print(f"计算因子 {factor_name} 时发生错误: {str(e)}")
