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


# 离群值处理
def filter_extreme_MAD(series, n):
    median = series.median()
    new_median = ((series - median).abs()).median()
    if new_median == 0:
        new_median = np.inf
    return series.clip(median - n * new_median, median + n * new_median)


# 中性化处理
def neutralization(factor, market_cap, industry_exposure):
    factor_resid = pd.DataFrame()
    factor_ols = pd.concat(
        [factor.stack(), market_cap, industry_exposure], axis=1
    ).dropna()
    datetime_list = sorted(list(set(market_cap.index.get_level_values(0))))
    for i in datetime_list:
        try:
            factor_ols_temp = factor_ols.loc[i]  # 截面数据做回归
            x = factor_ols_temp.iloc[:, 1:]  # 市值/行业
            y = factor_ols_temp.iloc[:, 0]  # 因子值
            factor_ols_resid_temp = pd.DataFrame(
                sm.OLS(y.astype(float), x.astype(float), hasconst=False, missing="drop")
                .fit()
                .resid,
                columns=["{}".format(i)],
            )
            factor_resid = pd.concat([factor_resid, factor_ols_resid_temp], axis=1)
        except:
            factor_resid = pd.concat([factor_resid, pd.DataFrame()], axis=1)
    factor_resid = factor_resid.T
    factor_resid.index = pd.to_datetime(factor_resid.index)
    return factor_resid


# 单因子检测
def Factor_Return_N_IC(factor, n, close, Rank_IC=True):

    date_list_whole = sorted(list(set(factor.index.get_level_values(0))))
    start_date = date_list_whole[0]
    end_date = date_list_whole[-1]
    stock_list = sorted(list(set(factor.index.get_level_values(1))))
    close = close.pct_change(n).shift(-n).stack()
    close = pd.concat([close, factor], axis=1).dropna().reset_index()
    close.columns = ["date", "stock", "change_days", "factor"]
    if Rank_IC == True:
        rank_ic = (
            close.groupby("date")[["change_days", "factor"]]
            .corr(method="spearman")
            .reset_index()
            .set_index(["date"])
        )
        return rank_ic[rank_ic.level_1 == "factor"][["change_days"]]


# ic_ir检测
def ic_ir(x, name):
    t_stat, p_value = stats.ttest_1samp(x, 0)
    IC = {
        "name": name,
        "IC mean": round(x.mean()[0], 4),
        "IC std": round(x.std()[0], 4),
        "IR": round(x.mean()[0] / x.std()[0], 4),
        "t_stat": round(t_stat[0], 4),
        "p_value": round(p_value[0], 4),
        "IC>0": round(len(x[x > 0].dropna()) / len(x), 4),
        "ABS_IC>2%": round((len(x[abs(x) > 0.02].dropna()) / len(x)), 4),
    }
    return pd.DataFrame([IC])


if __name__ == "__main__":

    # 注意：原始数据的起始日期是2014-01-01，结束日期为2022-12-14
    # 但是我们研究的起始日期是2020-01-02，结束日期为2022-01-01
    start_date = "2020-01-02"
    end_date = "2022-01-01"
    change_day = 5

    # 数据目录
    DATA_DIR = Path("./alpha_local/database")

    # 市值数据
    market_cap = pd.DataFrame(
        pd.read_pickle(DATA_DIR / "market_cap.pkl"), columns=["market_cap"]
    ).loc[start_date:end_date]
    # 行业暴露数据
    industry_exposure = pd.read_pickle(DATA_DIR / "industry_exposure.pkl").loc[
        start_date:end_date
    ]
    # 新股过滤数据
    new_stock_filter = pd.read_pickle(DATA_DIR / "new_stock_filter.pkl")
    # ST过滤数据
    st_filter = pd.read_pickle(DATA_DIR / "st_filter.pkl")
    # 停牌过滤数据
    suspended_filter = pd.read_pickle(DATA_DIR / "suspended_filter.pkl")
    # 涨跌停过滤数据
    limit_up_down_filter = pd.read_pickle(DATA_DIR / "limit_up_down_filter.pkl")
    # 收盘价数据
    close = pd.read_pickle(DATA_DIR / "20140101_20221214_全A_日级别.pkl").close.unstack(
        "order_book_id"
    )
    factor_alpha = (
        pd.read_pickle(DATA_DIR / "alpha_001.pkl")
        .dropna(axis=1, how="all")
        .loc[start_date:end_date]
    )

    # 券池过滤：新股 涨停 ST 停牌筛选
    factor_alpha = (
        factor_alpha.mask(new_stock_filter)
        .mask(st_filter)
        .mask(suspended_filter)
        .mask(limit_up_down_filter)
    ).dropna(axis=1, how="all")

    # 离群值处理
    factor_alpha = factor_alpha.apply(lambda x: filter_extreme_MAD(x, 3), axis=1)

    # 标准化处理
    factor_alpha = factor_alpha.sub(factor_alpha.mean(axis=1), axis=0).div(
        factor_alpha.std(axis=1), axis=0
    )

    # 中性化处理
    factor_alpha = neutralization(factor_alpha,market_cap,industry_exposure)

    # 单因子检验
    Result = Factor_Return_N_IC(factor_alpha.stack(),change_day,close)

    # ICIR
    ic_summary = pd.DataFrame()
    ic_summary = pd.concat([ic_summary, ic_ir(Result, "alpha_001")], axis=0)


    print()
