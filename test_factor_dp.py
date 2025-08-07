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


# 获取行业暴露度矩阵
def get_industry_exposure(order_book_ids, datetime_period, industry_type="zx"):
    """
    :param order_book_ids: 股票池 -> list
    :param datetime_period: 研究日 -> list
    :param industry_type: 行业分类标准 二选一 中信/申万 zx/sw -> str
    :return result: 虚拟变量 -> df_unstack
    """

    # 获取行业特征数据
    if industry_type == "zx":
        industry_map_dict = rqdatac.client.get_client().execute(
            "__internal__zx2019_industry"
        )
        # 构建个股/行业map
        df = pd.DataFrame(
            industry_map_dict,
            columns=["first_industry_name", "order_book_id", "start_date"],
        )
        df.sort_values(["order_book_id", "start_date"], ascending=True, inplace=True)
        df = df.pivot(
            index="start_date", columns="order_book_id", values="first_industry_name"
        ).ffill()
    else:
        industry_map_dict = rqdatac.client.get_client().execute(
            "__internal__shenwan_industry"
        )
        df = pd.DataFrame(
            industry_map_dict,
            columns=["index_name", "order_book_id", "version", "start_date"],
        )
        df = df[df.version == 2]
        df = df.drop_duplicates()
        df = (
            df.set_index(["start_date", "order_book_id"])
            .index_name.unstack("order_book_id")
            .ffill()
        )

    # 匹配交易日
    datetime_period_base = pd.to_datetime(
        get_trading_dates(get_previous_trading_date(df.index[0], 2), df.index[-1])
    )
    df.index = datetime_period_base.take(
        datetime_period_base.searchsorted(df.index, side="right") - 1
    )
    # 切片所需日期
    df = df.reset_index().drop_duplicates(subset=["index"]).set_index("index")
    df = (
        df.reindex(index=datetime_period_base)
        .ffill()
        .reindex(index=datetime_period)
        .ffill()
    )
    inter_stock_list = list(set(df.columns) & set(order_book_ids))
    df = df[inter_stock_list].sort_index(axis=1)

    # 生成行业虚拟变量
    return df.stack()


# 向量化中性化处理
def neutralization_vectorized(
    factor,
    order_book_ids,
    index_item="",
    industry_type="zx",
):

    datetime_period = factor.index.tolist()
    start = datetime_period[0].strftime("%F")
    end = datetime_period[-1].strftime("%F")

    try:
        # 获取存储数据
        df_industry_market = pd.read_pickle(
            f"2025-08-07/df_industry_market_{industry_type}_{index_item}_{start}_{end}.pkl"
        )
    except:
        # 获取市值暴露度
        market_cap = (
            execute_factor(LOG(Factor("market_cap_3")), order_book_ids, start, end)
            .stack()
            .to_frame("market_cap")
        )
        # 获取行业暴露度
        industry_df = pd.get_dummies(
            get_industry_exposure(order_book_ids, datetime_period, industry_type)
        ).astype(int)
        # 合并市值行业暴露度
        industry_df["market_cap"] = market_cap
        df_industry_market = industry_df
        df_industry_market.index.names = ["datetime", "order_book_id"]
        df_industry_market.dropna(axis=0, inplace=True)
        os.makedirs("2025-08-07", exist_ok=True)
        df_industry_market.to_pickle(
            f"2025-08-07/df_industry_market_{industry_type}_{index_item}_{start}_{end}.pkl"
        )

    df_industry_market["factor"] = factor.stack()
    df_industry_market.dropna(subset="factor", inplace=True)

    # 将factor列移到第一列
    cols = df_industry_market.columns.tolist()
    cols.remove("factor")
    df_industry_market = df_industry_market[["factor"] + cols]

    # 截面回归
    def neutralize_cross_section(group):

        try:
            y = group["factor"]
            x = group.iloc[:, 1:]
            model = sm.OLS(y, x, hasconst=False, missing="drop").fit()
            return pd.Series(model.resid)
        except:
            return pd.Series(dtype=float)

    factor_resid = df_industry_market.groupby(level=0).apply(neutralize_cross_section)
    factor_resid = factor_resid.reset_index(level=0, drop=True)
    factor_resid = factor_resid.unstack(level=0).T
    factor_resid.index = pd.to_datetime(factor_resid.index)

    return factor_resid


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
        open = pd.read_pickle(f"2025-08-07/open_{index_item}_{start}_{end}.pkl")
    except:
        # 新建预存储文档
        os.makedirs("2025-08-07", exist_ok=True)
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
        open.to_pickle(f"2025-08-07/open_{index_item}_{start}_{end}.pkl")

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
        "IC mean": round(result.mean(), 4),
        "IC std": round(result.std(), 4),
        "IR": round(result.mean() / result.std(), 4),
        "IC>0": round(len(result[result > 0].dropna()) / len(result), 4),
        "ABS_IC>2%": round(len(result[abs(result) > 0.02].dropna()) / len(result), 4),
        "t_stat": round(t_stat, 4),
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
        return_1d = pd.read_pickle(
            f"2025-08-07/return_1d_{index_item}_{start}_{end}.pkl"
        )
    except:
        # 新建预存储文档
        os.makedirs("2025-08-07", exist_ok=True)
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
        return_1d.to_pickle(f"2025-08-07/return_1d_{index_item}_{start}_{end}.pkl")

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

    start_date = "2022-01-01"
    end_date = "2025-07-01"
    index_item = "000852.XSHG"
    change_day = 20

    stock_universe = INDEX_FIX(start_date, end_date, index_item)
    stock_list = stock_universe.columns.tolist()
    date_list = stock_universe.index.tolist()

    # dp因子
    f_dp = Factor("dividend_yield_ttm")
    f_dp = execute_factor(f_dp, stock_list, start_date, end_date)

    try:
        combo_mask = pd.read_pickle(
            f"2025-08-07/combo_mask_{index_item}_{start_date}_{end_date}.pkl"
        )
    except:
        #  新股过滤
        new_stock_filter = get_new_stock_filter(stock_list, date_list)
        # st过滤
        st_filter = get_st_filter(stock_list, date_list)
        # 停牌过滤
        suspended_filter = get_suspended_filter(stock_list, date_list)
        
        combo_mask = (
            new_stock_filter.astype(int)
            + st_filter.astype(int)
            + suspended_filter.astype(int)
            + (~stock_universe).astype(int)
        ) == 0

        os.makedirs("2025-08-07", exist_ok=True)
        combo_mask.to_pickle(
            f"2025-08-07/combo_mask_{index_item}_{start_date}_{end_date}.pkl"
        )

    f_dp = f_dp.mask(~combo_mask).dropna(axis=1, how="all")

    # 离群值处理
    f_dp = mad_vectorized(f_dp)

    # 标准化处理
    f_dp = standardize(f_dp)

    # 中性化处理
    f_dp = neutralization_vectorized(f_dp, stock_list)


    # 涨停过滤
    limit_up_filter = get_limit_up_filter(stock_list, date_list)
    f_dp = f_dp.mask(limit_up_filter)

    # 计算IC
    ic, performance = calc_ic(f_dp, change_day, index_item)
