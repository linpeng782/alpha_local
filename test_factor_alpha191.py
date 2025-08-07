import scipy as sp
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from pathlib import Path
import os
from rqdatac import *
from rqfactor import *
from rqfactor import Factor
from rqfactor.extension import *

init("13522652015", "123456")
import rqdatac

from tqdm import *
import KDCJ
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "PingFang SC",
    "Hiragino Sans GB",
    "STHeiti",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

import warnings

warnings.filterwarnings("ignore")


# 动态券池
def INDEX_FIX(start_date, end_date, index_item):
    """
    :param start_date: 开始日 -> str
    :param end_date: 结束日 -> str
    :param index_item: 指数代码 -> str
    :return index_fix: 动态因子值 -> unstack
    """

    index_fix = pd.DataFrame(
        {
            k: dict.fromkeys(v, True)
            for k, v in index_components(
                index_item, start_date=start_date, end_date=end_date
            ).items()
        }
    ).T

    index_fix.fillna(False, inplace=True)

    return index_fix


def get_new_stock_filter(stock_list, datetime_period, newly_listed_threshold=252):
    """
    :param stock_list: 股票队列 -> list
    :param datetime_period: 研究周期 -> list
    :param newly_listed_threshold: 新股日期阈值 -> int
    :return newly_listed_window: 新股过滤券池 -> unstack
    """

    datetime_period_tmp = datetime_period.copy()
    # 多添加一天
    datetime_period_tmp += [
        pd.to_datetime(get_next_trading_date(datetime_period[-1], 1))
    ]
    # 获取上市日期
    listed_datetime_period = [instruments(stock).listed_date for stock in stock_list]
    # 获取上市后的第252个交易日（新股和老股的分界点）
    newly_listed_window = pd.Series(
        index=stock_list,
        data=[
            pd.to_datetime(get_next_trading_date(listed_date, n=newly_listed_threshold))
            for listed_date in listed_datetime_period
        ],
    )
    # 防止分割日在研究日之后，后续填充不存在
    for k, v in enumerate(newly_listed_window):
        if v > datetime_period_tmp[-1]:
            newly_listed_window.iloc[k] = datetime_period_tmp[-1]

    # 标签新股，构建过滤表格
    newly_listed_window.index.names = ["order_book_id"]
    newly_listed_window = newly_listed_window.to_frame("date")
    newly_listed_window["signal"] = True
    newly_listed_window = (
        newly_listed_window.reset_index()
        .set_index(["date", "order_book_id"])
        .signal.unstack("order_book_id")
        .reindex(index=datetime_period_tmp)
    )
    newly_listed_window = newly_listed_window.shift(-1).bfill().fillna(False).iloc[:-1]

    return newly_listed_window


# new_stock_filter = get_new_stock_filter(stock_list,date_list, newly_listed_threshold = 252)


# 1.2 st过滤（风险警示标的默认不进行研究）
def get_st_filter(stock_list, datetime_period):
    """
    :param stock_list: 股票池 -> list
    :param datetime_period: 研究周期 -> list
    :return st_filter: st过滤券池 -> unstack
    """

    # 当st时返回1，非st时返回0
    st_filter = is_st_stock(
        stock_list, datetime_period[0], datetime_period[-1]
    ).reindex(columns=stock_list, index=datetime_period)
    st_filter = st_filter.shift(-1).ffill()

    return st_filter


# st_filter = get_st_filter(stock_list,date_list)


# 1.3 停牌过滤 （无法交易）
def get_suspended_filter(stock_list, date_list):
    """
    :param stock_list: 股票池 -> list
    :param date_list: 研究周期 -> list
    :return suspended_filter: 停牌过滤券池 -> unstack
    """

    # 当停牌时返回1，非停牌时返回0
    suspended_filter = is_suspended(stock_list, date_list[0], date_list[-1]).reindex(
        columns=stock_list, index=date_list
    )
    suspended_filter = suspended_filter.shift(-1).ffill()

    return suspended_filter


# suspended_filter = get_suspended_filter(stock_list,date_list)


# 1.4 涨停过滤 （开盘无法买入）
def get_limit_up_filter(stock_list, date_list):
    """
    :param stock_list: 股票池 -> list
    :param date_list: 研究周期 -> list
    :return limit_up_filter: 涨停过滤券池 -> unstack
    """
    # 涨停时返回为1,非涨停返回为0
    price = get_price(
        stock_list,
        date_list[0],
        date_list[-1],
        adjust_type="none",
        fields=["open", "limit_up"],
    )
    df = (
        (price["open"] == price["limit_up"])
        .unstack("order_book_id")
        .shift(-1)
        .fillna(False)
    )

    return df


# 离群值处理
def mad(df, n=3 * 1.4826):

    # MAD:中位数去极值
    def filter_extreme_MAD(series, n):
        median = series.median()
        new_median = ((series - median).abs()).median()
        return series.clip(median - n * new_median, median + n * new_median)

    # 离群值处理
    df = df.apply(lambda x: filter_extreme_MAD(x, n), axis=1)

    return df


# 离群值处理
def mad_vectorized(df, n=3 * 1.4826):

    # 计算每行的中位数
    median = df.median(axis=1)
    # 计算每行的MAD
    mad_values = (df.sub(median, axis=0).abs()).median(axis=1)
    # 计算上下界
    lower_bound = median - n * mad_values
    upper_bound = median + n * mad_values

    return df.clip(lower_bound, upper_bound, axis=0)


# 标准化处理
def standardize(df):
    df_standardize = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
    return df_standardize


# 中性化处理
def neutralization(factor, market_cap, industry_exposure):
    factor_resid = pd.DataFrame()
    factor_ols = pd.concat(
        [factor.stack(), market_cap, industry_exposure], axis=1
    ).dropna()
    datetime_list = sorted(list(set(market_cap.index.get_level_values(0))))
    # datetime_list = datetime_list[:3]
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


# 向量化中性化处理
def neutralization_vectorized(factor, market_cap, industry_exposure):

    # 1. 合并因子值和市值/行业暴露度
    factor_ols = pd.concat(
        [factor.stack(), market_cap, industry_exposure], axis=1
    ).dropna()
    factor_ols.columns = ["factor"] + list(factor_ols.columns[1:])

    # 2. 截面回归
    def neutralize_cross_section(group):

        try:
            y = group["factor"]
            x = group.iloc[:, 1:]
            model = sm.OLS(y, x, hasconst=False, missing="drop").fit()
            return pd.Series(model.resid)
        except:
            return pd.Series(dtype=float)

    factor_resid = factor_ols.groupby(level=0).apply(neutralize_cross_section)
    factor_resid = factor_resid.reset_index(level=0, drop=True)
    factor_resid = factor_resid.unstack(level=0).T
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


# 单因子检验
def calc_ic(df, n, index_item, name="", Rank_IC=True):

    # 基础数据获取
    order_book_ids = df.columns.tolist()
    datetime_period = df.index
    start = datetime_period.min().strftime("%F")
    end = datetime_period.max().strftime("%F")

    # 提取预存储数据
    try:
        # 开盘价
        open = pd.read_pickle(
            f"alpha_local/database/open_{index_item}_{start}_{end}.pkl"
        )
    except:
        # 新建预存储文档
        os.makedirs("alpha_local/database", exist_ok=True)
        # 拿一个完整的券池表格，防止有些股票在某些日期没有数据，导致缓存数据不全，影响其他因子计算
        index_fix = INDEX_FIX(start, end, index_item)
        order_book_ids = index_fix.columns.tolist()
        datetime_period = index_fix.index.tolist()
        # 获取开盘价
        open = get_price(
            order_book_ids,
            start_date=start,
            end_date=end,
            frequency="1d",
            fields="open",
        ).open.unstack("order_book_id")
        # 存储
        open.to_pickle(f"alpha_local/database/open_{index_item}_{start}_{end}.pkl")

    # 未来一段收益股票的累计收益率计算
    return_n = open.pct_change(n).shift(-n - 1)

    # 计算IC
    if Rank_IC == True:
        result = df.corrwith(return_n, axis=1, method="spearman").dropna(how="all")
    else:
        result = df.corrwith(return_n, axis=1, method="pearson").dropna(how="all")

    # t检验 单样本
    t_stat, _ = stats.ttest_1samp(result, 0)

    # 因子报告
    report = {
        "name": name,
        "IC mean": round(result.mean(), 2),
        "IC std": round(result.std(), 2),
        "IR": round(result.mean() / result.std(), 2),
        "IC>0": round(len(result[result > 0].dropna()) / len(result), 2),
        "ABS_IC>2%": round(len(result[abs(result) > 0.02].dropna()) / len(result), 2),
        "t_stat": round(t_stat, 2),
    }

    print(report)

    report = pd.DataFrame([report])

    return result, report


def group_g(df, n, g, index_item, name="", rebalance=False):
    """
    :param df: 因子值 -> unstack
    :param n: 调仓日 -> int
    :param g: 分组数量 -> int
    :param index_item: 券池名 -> str
    :param name: 因子名 -> str
    :param rebalance: 是否rebalance -> bool
    :return group_return: 各分组日收益率 -> dataframe
    :return turnover_ratio: 各分组日调仓日换手率 -> dataframe
    """

    # 信号向后移动一天
    df = df.shift(1).iloc[1:]

    # 基础数据获取
    order_book_ids = df.columns.tolist()
    datetime_period = df.index
    start = datetime_period.min().strftime("%F")
    end = datetime_period.max().strftime("%F")

    # 提取预存储数据
    try:
        # 未来一天收益率
        return_1d = pd.read_pickle(f"tmp/return_1d_{index_item}_{start}_{end}.pkl")
    except:
        # 新建预存储文档
        os.makedirs("tmp", exist_ok=True)
        # 拿一个完整的券池表格，防止有些股票在某些日期没有数据，导致缓存数据不全，影响其他因子计算
        index_fix = INDEX_FIX(start, end, index_item)
        order_book_ids = index_fix.columns.tolist()
        # 未来一天收益率
        open = get_price(
            order_book_ids,
            start,
            get_next_trading_date(end, 1),
            "1d",
            "open",
            "pre",
            False,
            True,
        ).open.unstack("order_book_id")
        return_1d = open.pct_change().shift(-1).dropna(axis=0, how="all").stack()
        # 存储
        return_1d.to_pickle(f"tmp/return_1d_{index_item}_{start}_{end}.pkl")

    # 数据喝收益合并
    group = df.stack().to_frame("factor")
    group["current_renturn"] = return_1d
    group = group.dropna()
    group.reset_index(inplace=True)
    group.columns = ["date", "stock", "factor", "current_renturn"]

    # 空换手率 和 空分组收益率表格
    turnover_ratio = pd.DataFrame()
    group_return = pd.DataFrame()

    datetime_period = pd.to_datetime(group.date.unique())

    # 按步长周期调仓
    for i in range(0, len(datetime_period) - 1, n):  # -1 防止刚好切到最后一天没法计算
        # 截面分组
        single = group[group.date == datetime_period[i]].sort_values(by="factor")
        # 根据值的大小进行切分
        single.loc[:, "group"] = pd.qcut(
            single.factor, g, list(range(1, g + 1))
        ).to_list()
        group_dict = {}
        # 分组内的股票
        for j in range(1, g + 1):
            group_dict[j] = single[single.group == j].stock.tolist()

        # 计算换手率
        turnover_ratio_temp = []
        if i == 0:
            # 首期分组成分股 存入历史
            temp_group_dict = group_dict
        else:
            # 分组计算换手率
            for j in range(1, g + 1):
                turnover_ratio_temp.append(
                    len(list(set(temp_group_dict[j]).difference(set(group_dict[j]))))
                    / len(set(temp_group_dict[j]))
                )
            # 存储分组换手率
            turnover_ratio = pd.concat(
                [
                    turnover_ratio,
                    pd.DataFrame(
                        turnover_ratio_temp,
                        index=["G{}".format(j) for j in list(range(1, g + 1))],
                        columns=[datetime_period[i]],
                    ).T,
                ],
                axis=0,
            )
            # 存入历史
            temp_group_dict = group_dict

        # 获取周期
        # 不够一个调仓周期，剩下的都是最后一个周期
        if i < len(datetime_period) - n:
            period = group[group.date.isin(datetime_period[i : i + n])]
        else:
            # 完整周期
            period = group[group.date.isin(datetime_period[i:])]

        # 计算各分组收益率（期间不rebalance权重）
        group_return_temp = []
        for j in range(1, g + 1):
            if rebalance:
                # 横截面汇总
                group_ret = period[period.stock.isin(group_dict[j])]
                group_ret = group_ret.set_index(
                    ["date", "stock"]
                ).current_renturn.unstack("stock")
                group_ret_combine_ret = group_ret.mean(axis=1)

            else:
                # 组内各标的数据
                group_ret = period[period.stock.isin(group_dict[j])]
                group_ret = group_ret.set_index(
                    ["date", "stock"]
                ).current_renturn.unstack("stock")
                # 标的累计收益
                group_ret_combine_cumnet = (1 + group_ret).cumprod().mean(axis=1)
                # 组合的逐期收益
                group_ret_combine_ret = group_ret_combine_cumnet.pct_change()
                # 第一期填补
                group_ret_combine_ret.iloc[0] = group_ret.iloc[0].mean()

            # 合并各分组
            group_return_temp.append(group_ret_combine_ret)

        # 每个步长期间的收益合并
        group_return = pd.concat(
            [
                group_return,
                pd.DataFrame(
                    group_return_temp,
                    index=["G{}".format(j) for j in list(range(1, g + 1))],
                ).T,
            ],
            axis=0,
        )
        # 进度
        print("\r 当前：{} / 总量：{}".format(i, len(datetime_period)), end="")

    # 基准，各组的平均收益
    group_return["Benchmark"] = group_return.mean(axis=1)
    group_return = (group_return + 1).cumprod()
    # 年化收益计算
    group_annual_ret = group_return.iloc[-1] ** (252 / len(group_return)) - 1
    group_annual_ret -= group_annual_ret.Benchmark
    group_annual_ret = group_annual_ret.drop("Benchmark").to_frame("annual_ret")
    group_annual_ret["group"] = list(range(1, g + 1))
    corr_value = round(group_annual_ret.corr(method="spearman").iloc[0, 1], 4)
    group_annual_ret.annual_ret.plot(
        kind="bar", figsize=(5, 3), title=f"{name}_分层超额年化收益_单调性{corr_value}"
    )

    group_return.plot(figsize=(5, 3), title=f"{name}_分层净值表现")

    yby_performance = (
        group_return.pct_change()
        .resample("Y")
        .apply(lambda x: (1 + x).cumprod().iloc[-1])
        .T
    )
    yby_performance -= yby_performance.loc["Benchmark"]
    yby_performance = yby_performance.replace(0, np.nan).dropna(how="all")
    yby_performance.plot(
        kind="bar",
        figsize=(5, 3),
        title=f"{name}_逐年分层年化收益",
        color=[
            "powderblue",
            "lightskyblue",
            "cornflowerblue",
            "steelblue",
            "royalblue",
        ],
    )

    return group_return, turnover_ratio


if __name__ == "__main__":

    # 注意：原始数据的起始日期是2014-01-01，结束日期为2022-12-14
    # 但是我们研究的起始日期是2020-01-02，结束日期为2022-01-01
    start_date = "2020-01-02"
    end_date = "2022-01-01"
    change_day = 5
    index_item = "000985.XSHG"

    stock_universe = INDEX_FIX('2014-01-01', '2022-12-14', index_item)
    stock_list = stock_universe.columns.tolist()
    start_date = stock_universe.index[0].strftime('%F')
    end_date = stock_universe.index[-1].strftime('%F')
    datetime_period = stock_universe.index.tolist()

    # 新股过滤
    new_stock_filter = get_new_stock_filter(stock_list,datetime_period)
    # st过滤
    st_filter = get_st_filter(stock_list,datetime_period)
    # 停牌过滤
    suspended_filter = get_suspended_filter(stock_list,datetime_period)
    # 涨停过滤
    limit_up_filter = get_limit_up_filter(stock_list,datetime_period)
    

    


    # 数据目录 - 修复路径问题
    # 获取当前脚本的目录
    script_dir = Path(__file__).parent
    DATA_DIR = script_dir / "database"

    # 读取市值数据
    market_cap = pd.DataFrame(
        pd.read_pickle(DATA_DIR / "market_cap.pkl"), columns=["market_cap"]
    ).loc[start_date:end_date]
    # 读取行业暴露数据
    industry_exposure = pd.read_pickle(DATA_DIR / "industry_exposure.pkl").loc[
        start_date:end_date
    ]
    # 读取新股过滤数据
    new_stock_filter = pd.read_pickle(DATA_DIR / "new_stock_filter.pkl")
    # 读取ST过滤数据
    st_filter = pd.read_pickle(DATA_DIR / "st_filter.pkl")
    # 读取停牌过滤数据
    suspended_filter = pd.read_pickle(DATA_DIR / "suspended_filter.pkl")
    # 读取涨跌停过滤数据
    limit_up_down_filter = pd.read_pickle(DATA_DIR / "limit_up_down_filter.pkl")
    # 读取收盘价数据
    close = pd.read_pickle(DATA_DIR / "20140101_20221214_全A_日级别.pkl").close.unstack(
        "order_book_id"
    )

    # 读取因子数据
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
    factor_alpha = mad_vectorized(factor_alpha)

    # 标准化处理
    factor_alpha = standardize(factor_alpha)

    # 中性化处理
    factor_alpha = neutralization_vectorized(
        factor_alpha, market_cap, industry_exposure
    )

    # 计算IC
    ic, performance = calc_ic(factor_alpha, change_day, index_item)
