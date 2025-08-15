import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
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


# 热力图
def hot_corr(name, ic_df):
    """
    :param name: 因子名称 -> list
    :param ic_df: ic序列表 -> dataframe
    :return fig: 热力图 -> plt
    """
    ax = plt.subplots(figsize=(len(name), len(name)))  # 调整画布大小
    ax = sns.heatmap(
        ic_df[name].corr(), vmin=0.4, square=True, annot=True, cmap="Greens"
    )  # annot=True 表示显示系数
    plt.title("Factors_IC_CORRELATION")
    # 设置刻度字体大小
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)


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


# 新股过滤
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


# st过滤（风险警示标的默认不进行研究）
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


# 停牌过滤 （无法交易）
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


# 涨停过滤 （开盘无法买入）
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


# 行业市值中性化
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
            f"factor_lib/df_industry_market_{industry_type}_{index_item}_{start}_{end}.pkl"
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
        os.makedirs("factor_lib", exist_ok=True)
        df_industry_market.to_pickle(
            f"factor_lib/df_industry_market_{industry_type}_{index_item}_{start}_{end}.pkl"
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
        open = pd.read_pickle(f"factor_lib/open_{index_item}_{start}_{end}.pkl")
    except:
        # 新建预存储文档
        os.makedirs("factor_lib", exist_ok=True)
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
        open.to_pickle(f"factor_lib/open_{index_item}_{start}_{end}.pkl")

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
        "change_day": n,
        "IC mean": round(result.mean(), 4),
        "IC std": round(result.std(), 4),
        "ICIR": round(result.mean() / result.std(), 4),
        "IC>0": round(len(result[result > 0].dropna()) / len(result), 4),
        "ABS_IC>2%": round(len(result[abs(result) > 0.02].dropna()) / len(result), 4),
        "t_stat": round(t_stat, 4),
    }

    print(report)

    report = pd.DataFrame([report])

    return result, report


# 分组回测
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
            f"factor_lib/return_1d_{index_item}_{start}_{end}.pkl"
        )
    except:
        # 新建预存储文档
        os.makedirs("factor_lib", exist_ok=True)
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
        return_1d.to_pickle(f"factor_lib/return_1d_{index_item}_{start}_{end}.pkl")

    # 数据和收益合并
    group = df.stack().to_frame("factor")
    group["current_renturn"] = return_1d
    group = group.dropna()
    group.reset_index(inplace=True)
    group.columns = ["date", "stock", "factor", "current_renturn"]

    # 换手率 和 分组收益率表格
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

        if i == 2540:
            breakpoint()

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
        kind="bar", figsize=(10, 5), title=f"{name}_分层超额年化收益_单调性{corr_value}"
    )

    group_return.plot(figsize=(10, 5), title=f"{name}_分层净值表现")

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
        figsize=(10, 5),
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


# 数据清洗封装函数
def preprocess_factor(factor, stock_universe, index_item):

    stock_list = stock_universe.columns.tolist()
    date_list = stock_universe.index.tolist()
    start_date = date_list[0].strftime("%F")
    end_date = date_list[-1].strftime("%F")

    try:
        combo_mask = pd.read_pickle(
            f"factor_lib/combo_mask_{index_item}_{start_date}_{end_date}.pkl"
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

        os.makedirs("factor_lib", exist_ok=True)
        combo_mask.to_pickle(
            f"factor_lib/combo_mask_{index_item}_{start_date}_{end_date}.pkl"
        )

    # axis=1,过滤掉所有日期截面都是nan的股票
    factor = factor.mask(~combo_mask).dropna(axis=1, how="all")

    # 离群值处理
    factor = mad_vectorized(factor)

    # 标准化处理
    factor = standardize(factor)

    # 中性化处理
    factor = neutralization_vectorized(factor, stock_list)

    # 涨停过滤
    limit_up_filter = get_limit_up_filter(stock_list, date_list)
    factor = factor.mask(limit_up_filter)

    return factor


def factor_layered_backtest(df, n, g, index_item, name="", rebalance=False):
    """
    因子分层回测函数（优化版）

    :param df: 因子值 DataFrame (unstack格式)
    :param n: 调仓频率（天数）
    :param g: 分组数量
    :param index_item: 券池名称
    :param name: 因子名称
    :param rebalance: 是否每日重新平衡权重
        - True: 每日等权重平均（高交易成本）
        - False: 买入持有策略（低交易成本）
    :return: (group_return, turnover_ratio)
        - group_return: 各分组的累计净值表现
        - turnover_ratio: 各分组的换手率
    """

    df = df.shift(1).iloc[1:]
    order_book_ids = df.columns.tolist()
    datetime_period = df.index
    start = datetime_period.min().strftime("%F")
    end = datetime_period.max().strftime("%F")

    try:
        return_1d = pd.read_pickle(
            f"factor_lib/return_1d_{index_item}_{start}_{end}.pkl"
        )
    except:
        os.makedirs("factor_lib", exist_ok=True)
        index_fix = INDEX_FIX(start, end, index_item)
        order_book_ids = index_fix.columns.tolist()
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
        return_1d.to_pickle(f"factor_lib/return_1d_{index_item}_{start}_{end}.pkl")

    # 数据合并，使用multiindex
    factor_data = df.stack().to_frame("factor")
    factor_data["current_return"] = return_1d
    factor_data = factor_data.dropna()

    # 获取调仓日期和数据边界
    actual_rebalance_dates = datetime_period[::n]  # 真正的调仓日期
    #data_end_date = datetime_period[-1]  # 数据结束边界
    
    # 批量分组
    all_groups = {}
    turnover_data = []

    ##########计算调仓日分组和换手率##########
    # 为所有调仓日期构建分组信息
    for i, date in enumerate(actual_rebalance_dates):
        # 获取当前因子值
        current_factors = factor_data.loc[date, "factor"]

        # 使用qcut进行分组
        groups = pd.qcut(current_factors, g, labels=range(1, g + 1))  
        current_groups = (
            current_factors.groupby(groups).apply(lambda x: x.index.tolist()).to_dict()
        )

        # 计算换手率（除了第一次）
        if i > 0:
            turnover_rates = []
            for group_id in range(1, g + 1):
                if group_id in all_groups[i - 1] and group_id in current_groups:
                    old_stocks = set(all_groups[i - 1][group_id])
                    new_stocks = set(current_groups[group_id])
                    turnover = (
                        len(old_stocks - new_stocks) / len(old_stocks)
                        if old_stocks
                        else 0
                    )
                    turnover_rates.append(turnover)
                else:
                    turnover_rates.append(np.nan)

            turnover_data.append({"date": date, "turnover": turnover_rates})

        all_groups[i] = current_groups
    
    if turnover_data:
        turnover_ratio = pd.DataFrame(
            [d["turnover"] for d in turnover_data],
            index=[d["date"] for d in turnover_data],
            columns=[f"G{i}" for i in range(1, g + 1)],
        )
    else:
        turnover_ratio = pd.DataFrame()
    
    print("turnover_ratio",turnover_ratio)

    ##########计算分组收益##########
    group_returns_list = []
    
    # 处理所有调仓周期
    for i, start_date in enumerate(actual_rebalance_dates):
        
        # 确定周期结束日期
        is_last_period = (i == len(actual_rebalance_dates) - 1)
        
        if not is_last_period:
            # 正常周期：到下一个调仓日
            end_date = actual_rebalance_dates[i + 1]
            period_mask = (factor_data.index.get_level_values(0) >= start_date) & (
                factor_data.index.get_level_values(0) < end_date
            )
        else:
            # 最后一个周期：到数据结束日
            period_mask = factor_data.index.get_level_values(0) >= start_date

        period_data = factor_data[period_mask]

        if i not in all_groups:
            continue

        group_dict = all_groups[i]
        period_returns = {}

        for group_id in range(1, g + 1):
            if group_id not in group_dict:
                continue

            stocks = group_dict[group_id]
            # 获取该组股票的收益率数据
            group_data = period_data[period_data.index.get_level_values(1).isin(stocks)]

            if len(group_data) == 0:
                continue

            # 根据rebalance参数选择计算方式
            if rebalance:
                # 每日等权重平均（每天重新平衡权重）
                portfolio_daily_returns = group_data.groupby(level=0)[
                    "current_return"
                ].mean()
            else:

                # 将数据重新整理为日期×股票的格式
                stock_daily_returns = group_data.reset_index().pivot(
                    index="datetime", columns="order_book_id", values="current_return"
                )
                # 等权重买入股票，不重新平衡
                group_cum_returns = (1 + stock_daily_returns).cumprod().mean(axis=1)
                portfolio_daily_returns = group_cum_returns.pct_change()  # 转为日收益率

                # 处理第一天的收益率
                if not portfolio_daily_returns.empty:
                    portfolio_daily_returns.iloc[0] = stock_daily_returns.iloc[0].mean()

            period_returns[f"G{group_id}"] = portfolio_daily_returns

        # 将该期间收益率添加到总列表中
        if period_returns:
            period_df = pd.DataFrame(period_returns)
            group_returns_list.append(period_df)

    if group_returns_list:
        group_return = pd.concat(group_returns_list, axis=0)
    else:
        group_return = pd.DataFrame()

    if not group_return.empty:
        # 基准收益
        group_return["Benchmark"] = group_return.mean(axis=1)
        group_return = (group_return + 1).cumprod()
        print(group_return)

        # 计算年化收益率
        group_annual_ret = group_return.iloc[-1] ** (252 / len(group_return)) - 1
        group_annual_ret = group_annual_ret - group_annual_ret.Benchmark
        group_annual_ret = group_annual_ret.drop("Benchmark").to_frame("annual_ret")
        group_annual_ret["group"] = list(range(1, g + 1))
        corr_value = round(group_annual_ret.corr(method="spearman").iloc[0, 1], 4)

        group_annual_ret.annual_ret.plot(
            kind="bar",
            figsize=(10, 5),
            title=f"{name}_分层超额年化收益_单调性{corr_value}",
        )

        group_return.plot(figsize=(10, 5), title=f"{name}_分层净值表现")

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
            figsize=(10, 5),
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


# 获取买入队列
def get_buy_list(df, top_type="rank", rank_n=100, quantile_q=0.8):
    """
    :param df: 因子值 -> dataframe/unstack
    :param top_type: 选择买入队列方式，从['rank','quantile']选择一种方式 -> str
    :param rank_n: 值最大的前n只的股票 -> int
    :param quantile_q: 值最大的前n分位数的股票 -> float
    :return df: 买入队列 -> dataframe/unstack
    """

    if top_type == "rank":
        df = df.rank(axis=1, ascending=False) <= rank_n
    elif top_type == "quantile":
        df = df.sub(df.quantile(quantile_q, axis=1), axis=0) > 0
    else:
        print("select one from ['rank','quantile']")

    df = df.astype(int)
    df = df.replace(0, np.nan).dropna(how="all", axis=1)

    return df


# 4.2 获取标的收益
def get_bar(df, adjust):
    """
    :param df: 买入队列 -> dataframe/unstack
    :return ret: 基准的逐日收益 -> dataframe
    """
    start_date = get_previous_trading_date(df.index.min(), 1).strftime("%F")
    end_date = df.index.max().strftime("%F")
    stock_list = df.columns.tolist()
    price_open = get_price(
        stock_list, start_date, end_date, fields=["open"], adjust_type=adjust
    ).open.unstack("order_book_id")

    return price_open


# 4.3 获取基准收益
def get_benchmark(df, benchmark, benchmark_type="mcw"):
    """
    :param df: 买入队列 -> dataframe/unstack
    :param benchmark: 基准指数 -> str
    :return ret: 基准的逐日收益 -> dataframe
    """
    start_date = get_previous_trading_date(df.index.min(), 1).strftime("%F")
    end_date = df.index.max().strftime("%F")
    if benchmark_type == "mcw":
        price_open = get_price(
            [benchmark], start_date, end_date, fields=["open"]
        ).open.unstack("order_book_id")
    else:
        index_fix = INDEX_FIX(start_date, end_date, benchmark)
        stock_list = index_fix.columns.tolist()
        price_open = get_price(
            stock_list, start_date, end_date, fields=["open"]
        ).open.unstack("order_book_id")
        price_open = price_open.pct_change().mask(~index_fix).mean(axis=1)
        price_open = (1 + price_open).cumprod().to_frame(benchmark)

    return price_open


# 4.4 回测框架
def backtest(
    portfolio_weights,
    rebalance_frequency=20,
    initial_capital=10000 * 10000,
    stamp_tax_rate=0.0005,
    transfer_fee_rate=0.0001,
    commission_rate=0.0002,
    min_transaction_fee=5,
    cash_annual_yield=0.02,
):
    """
    量化策略回测框架

    :param portfolio_weights: 投资组合权重矩阵 -> DataFrame
    :param rebalance_frequency: 调仓频率（天数） -> int
    :param initial_capital: 初始资本 -> float
    :param stamp_tax_rate: 印花税率 -> float
    :param transfer_fee_rate: 过户费率 -> float
    :param commission_rate: 佣金率 -> float
    :param min_transaction_fee: 最低交易手续费 -> float
    :param cash_annual_yield: 现金年化收益率 -> float
    :return: 账户历史记录 -> DataFrame
    """

    # =========================== 基础参数初始化 ===========================
    # 保存初始资本备份，用于最后的统计计算
    cash = initial_capital
    # 初始化历史持仓，第一次调仓时为0
    previous_holdings = 0
    # 买入成本费率：过户费 + 佣金
    buy_cost_rate = transfer_fee_rate + commission_rate
    # 卖出成本费率：印花税 + 过户费 + 佣金
    sell_cost_rate = stamp_tax_rate + transfer_fee_rate + commission_rate
    # 现金账户日利率（年化收益率转换为日收益率）
    daily_cash_yield = (1 + cash_annual_yield) ** (1 / 252) - 1

    # =========================== 数据结构初始化 ===========================
    # 创建账户历史记录表，索引为所有交易日
    account_history = pd.DataFrame(
        index=portfolio_weights.index,
        # 列1：账户总资产
        # 列2：持仓市值
        # 列3：现金账户余额
        columns=["total_account_asset", "holding_market_cap", "cash_account"],
    )
    # 获取所有股票的开盘价格数据（未复权）
    open_prices = get_bar(portfolio_weights, "none")
    # 获取所有股票的后复权价格数据
    adjusted_prices = get_bar(portfolio_weights, "post")
    # 获取每只股票的最小交易单位（通常为100股）
    min_trade_units = pd.Series(
        dict(
            [
                (stock, instruments(stock).round_lot)
                for stock in portfolio_weights.columns.tolist()
            ]
        )
    )
    # 生成调仓日期列表：每 rebalance_frequency 天调仓一次
    # 确保最后一天也被包含在调仓日中
    rebalance_dates = sorted(
        set(
            portfolio_weights.index.tolist()[::rebalance_frequency]
            + [portfolio_weights.index[-1]]
        )
    )

    # =========================== 开始逐期调仓循环 ===========================
    for i in tqdm(range(0, len(rebalance_dates) - 1)):
        rebalance_date = rebalance_dates[i]  # 当前调仓日期
        next_rebalance_date = rebalance_dates[i + 1]  # 下一个调仓日期

        # =========================== 获取当前调仓日的目标权重 ===========================
        # 获取当前调仓日的目标权重，并删除缺失值
        current_target_weights = portfolio_weights.loc[rebalance_date].dropna()
        # 获取目标股票列表
        target_stocks = current_target_weights.index.tolist()

        # =========================== 计算目标持仓数量 ===========================
        # 计算公式：目标持仓 = 向下取整(权重 * 可用资金 / (调整后价格 * 最小交易单位)) * 最小交易单位
        # 步骤1：按权重分配资金
        # 步骤2：乘以可用资金
        # 步骤3：除以调整后的股价（预留卖出手续费）
        # 步骤4：向下取整到最小交易单位的整数倍
        # 步骤5：转换为实际可交易的股数
        target_holdings = (
            current_target_weights
            * cash
            / (open_prices.loc[rebalance_date, target_stocks] * (1 + sell_cost_rate))
            // min_trade_units.loc[target_stocks]
        ) * min_trade_units.loc[target_stocks]

        # =========================== 仓位变动计算 ===========================
        ## 步骤1：计算持仓变动量（目标持仓 - 历史持仓）
        # fill_value=0 确保新增股票（历史持仓为空）和清仓股票（目标持仓为空）都能正确计算
        holdings_change_raw = target_holdings.sub(
            previous_holdings, fill_value=0
        )  # 计算原始持仓变动量

        ## 步骤2：过滤掉无变动的股票（变动量为0的股票）
        # 将变动量为0的股票标记为NaN，然后删除，只保留需要调仓的股票
        holdings_change_filtered = holdings_change_raw.replace(
            0, np.nan
        )  # 将无变动的股票标记为NaN

        ## 步骤3：获取最终的交易执行列表
        # 正数表示需要买入的股数，负数表示需要卖出的股数
        trades_to_execute = (
            holdings_change_filtered.dropna()
        )  # 删除NaN，只保留需要执行的交易

        # =========================== 获取当前交易日价格 ===========================
        current_prices = open_prices.loc[
            rebalance_date
        ]  # 获取当前调仓日的所有股票开盘价

        # =========================== 计算交易成本 ===========================
        def calc_transaction_fee(transaction_value, min_transaction_fee):
            """
            计算单笔交易的手续费
            :param transaction_value: 交易金额（正数为买入，负数为卖出）
            :param min_transaction_fee: 最低交易手续费
            :return: 交易手续费
            """
            if pd.isna(transaction_value) or transaction_value == 0:
                return 0  # 无交易时手续费为0
            elif transaction_value < 0:  # 卖出交易（负数）
                fee = (
                    -transaction_value * sell_cost_rate
                )  # 卖出手续费：印花税 + 过户费 + 佣金
            else:  # 买入交易（正数）
                fee = transaction_value * buy_cost_rate  # 买入手续费：过户费 + 佣金

            # 应用最低手续费限制
            return max(fee, min_transaction_fee)  # 返回实际手续费和最低手续费中的较大值

        # 计算总交易成本：交易金额 = 价格 * 交易股数
        # 计算每笔交易的交易金额
        # 对每笔交易计算手续费
        # 求和得到总手续费
        total_transaction_cost = (
            (current_prices * trades_to_execute)
            .apply(lambda x: calc_transaction_fee(x, min_transaction_fee))
            .sum()
        )

        # =========================== 价格复权调整 ===========================
        # 计算从调仓日到下一调仓日的价格复权比率
        price_adjustment_ratio = (
            adjusted_prices.loc[rebalance_date:next_rebalance_date]
            / adjusted_prices.loc[rebalance_date]
        )  # 复权比率 = 期间价格 / 起始价格

        # 将复权比率应用到当日开盘价，得到期间调整后价格
        period_adjusted_prices = (
            price_adjustment_ratio.T.mul(  # 转置以便于计算
                current_prices, axis=0
            )  # 乘以当日开盘价
            .dropna(how="all")
            .T
        )  # 删除全为NaN的行并转置回来

        # =========================== 计算投资组合市值 ===========================
        # 投资组合市值 = 每只股票的(调整后价格 * 持仓数量)的总和
        portfolio_market_value = (period_adjusted_prices * target_holdings).sum(
            axis=1
        )  # 按日计算投资组合市值

        # =========================== 计算现金账户余额 ===========================
        # 初始现金余额 = 可用资金 - 交易成本 - 初始投资金额
        initial_cash_balance = (
            cash - total_transaction_cost - portfolio_market_value.loc[rebalance_date]
        )

        # 计算期间现金账户的复利增长（按日计息）
        cash_balance = pd.Series(
            [
                initial_cash_balance
                * ((1 + daily_cash_yield) ** (day + 1))  # 复利计息公式
                for day in range(0, len(portfolio_market_value))
            ],  # 对每一天计算
            index=portfolio_market_value.index,
        )  # 使用相同的日期索引

        # =========================== 计算账户总资产 ===========================
        total_portfolio_value = (
            portfolio_market_value + cash_balance
        )  # 总资产 = 持仓市值 + 现金余额

        # =========================== 更新历史数据为下一次调仓做准备 ===========================
        previous_holdings = target_holdings  # 更新历史持仓为当前目标持仓
        cash = total_portfolio_value.loc[
            next_rebalance_date
        ]  # 更新可用资金为下一调仓日的账户总值

        # =========================== 保存账户历史记录 ===========================
        # 将当前期间的账户数据保存到历史记录中（保疙2位小数）
        account_history.loc[
            rebalance_date:next_rebalance_date, "total_account_asset"
        ] = round(total_portfolio_value, 2)
        account_history.loc[
            rebalance_date:next_rebalance_date, "holding_market_cap"
        ] = round(portfolio_market_value, 2)
        account_history.loc[rebalance_date:next_rebalance_date, "cash_account"] = round(
            cash_balance, 2
        )

    # =========================== 添加初始日记录并排序 ===========================
    # 在第一个交易日之前添加初始资本记录
    initial_date = pd.to_datetime(
        get_previous_trading_date(account_history.index.min(), 1)
    )
    account_history.loc[initial_date] = [
        initial_capital,
        0,
        initial_capital,
    ]  # [总资产, 持仓市值, 现金余额]
    account_history = account_history.sort_index()  # 按日期排序

    return account_history  # 返回完整的账户历史记录


# 4.5 回测绩效指标绘制
def get_performance_analysis(account_result, rf=0.03, benchmark_index="000985.XSHG"):

    # 加入基准
    performance = pd.concat(
        [
            account_result["total_account_asset"].to_frame("strategy"),
            get_benchmark(account_result, benchmark_index),
        ],
        axis=1,
    )
    performance_net = performance.pct_change().dropna(how="all")  # 清算至当日开盘
    performance_cumnet = (1 + performance_net).cumprod()
    performance_cumnet["alpha"] = (
        performance_cumnet["strategy"] / performance_cumnet[benchmark_index]
    )
    performance_cumnet = performance_cumnet.fillna(1)

    # 指标计算
    performance_pct = performance_cumnet.pct_change().dropna()

    # 策略收益
    strategy_name, benchmark_name, alpha_name = performance_cumnet.columns.tolist()
    Strategy_Final_Return = performance_cumnet[strategy_name].iloc[-1] - 1

    # 策略年化收益
    Strategy_Annualized_Return_EAR = (1 + Strategy_Final_Return) ** (
        252 / len(performance_cumnet)
    ) - 1

    # 基准收益
    Benchmark_Final_Return = performance_cumnet[benchmark_name].iloc[-1] - 1

    # 基准年化收益
    Benchmark_Annualized_Return_EAR = (1 + Benchmark_Final_Return) ** (
        252 / len(performance_cumnet)
    ) - 1

    # alpha
    ols_result = sm.OLS(
        performance_pct[strategy_name] * 252 - rf,
        sm.add_constant(performance_pct[benchmark_name] * 252 - rf),
    ).fit()
    Alpha = ols_result.params[0]

    # beta
    Beta = ols_result.params[1]

    # beta_2 = np.cov(performance_pct[strategy_name],performance_pct[benchmark_name])[0,1]/performance_pct[benchmark_name].var()
    # 波动率
    Strategy_Volatility = performance_pct[strategy_name].std() * np.sqrt(252)

    # 夏普
    Strategy_Sharpe = (Strategy_Annualized_Return_EAR - rf) / Strategy_Volatility

    # 下行波动率
    strategy_ret = performance_pct[strategy_name]
    Strategy_Down_Volatility = strategy_ret[strategy_ret < 0].std() * np.sqrt(252)

    # sortino
    Sortino = (Strategy_Annualized_Return_EAR - rf) / Strategy_Down_Volatility

    # 跟踪误差
    Tracking_Error = (
        performance_pct[strategy_name] - performance_pct[benchmark_name]
    ).std() * np.sqrt(252)

    # 信息比率
    Information_Ratio = (
        Strategy_Annualized_Return_EAR - Benchmark_Annualized_Return_EAR
    ) / Tracking_Error

    # 最大回测
    i = np.argmax(
        (
            np.maximum.accumulate(performance_cumnet[strategy_name])
            - performance_cumnet[strategy_name]
        )
        / np.maximum.accumulate(performance_cumnet[strategy_name])
    )
    j = np.argmax(performance_cumnet[strategy_name][:i])
    Max_Drawdown = (
        1 - performance_cumnet[strategy_name][i] / performance_cumnet[strategy_name][j]
    )

    # 卡玛比率
    Calmar = (Strategy_Annualized_Return_EAR) / Max_Drawdown

    # 超额收益
    Alpha_Final_Return = performance_cumnet[alpha_name].iloc[-1] - 1

    # 超额年化收益
    Alpha_Annualized_Return_EAR = (1 + Alpha_Final_Return) ** (
        252 / len(performance_cumnet)
    ) - 1

    # 超额波动率
    Alpha_Volatility = performance_pct[alpha_name].std() * np.sqrt(252)

    # 超额夏普
    Alpha_Sharpe = (Alpha_Annualized_Return_EAR - rf) / Alpha_Volatility

    # 超额最大回测
    i = np.argmax(
        (
            np.maximum.accumulate(performance_cumnet[alpha_name])
            - performance_cumnet[alpha_name]
        )
        / np.maximum.accumulate(performance_cumnet[alpha_name])
    )
    j = np.argmax(performance_cumnet[alpha_name][:i])
    Alpha_Max_Drawdown = (
        1 - performance_cumnet[alpha_name][i] / performance_cumnet[alpha_name][j]
    )

    # 胜率
    performance_pct["win"] = performance_pct[alpha_name] > 0
    Win_Ratio = performance_pct["win"].value_counts().loc[True] / len(performance_pct)

    # 盈亏比
    profit_lose = performance_pct.groupby("win")[alpha_name].mean()
    Profit_Lose_Ratio = abs(profit_lose[True] / profit_lose[False])

    result = {
        "策略累计收益": round(Strategy_Final_Return, 4),
        "策略年化收益": round(Strategy_Annualized_Return_EAR, 4),
        "基准累计收益": round(Benchmark_Final_Return, 4),
        "基准年化收益": round(Benchmark_Annualized_Return_EAR, 4),
        "阿尔法": round(Alpha, 4),
        "贝塔": round(Beta, 4),
        "波动率": round(Strategy_Volatility, 4),
        "夏普比率": round(Strategy_Sharpe, 4),
        "下行波动率": round(Strategy_Down_Volatility, 4),
        "索提诺比率": round(Sortino, 4),
        "跟踪误差": round(Tracking_Error, 4),
        "信息比率": round(Information_Ratio, 4),
        "最大回撤": round(Max_Drawdown, 4),
        "卡玛比率": round(Calmar, 4),
        "超额累计收益": round(Alpha_Final_Return, 4),
        "超额年化收益": round(Alpha_Annualized_Return_EAR, 4),
        "超额波动率": round(Alpha_Volatility, 4),
        "超额夏普": round(Alpha_Sharpe, 4),
        "超额最大回撤": round(Alpha_Max_Drawdown, 4),
        "胜率": round(Win_Ratio, 4),
        "盈亏比": round(Profit_Lose_Ratio, 4),
    }

    performance_cumnet.plot(secondary_y="alpha", figsize=(10, 6))
    print(pd.DataFrame([result]).T)

    return performance_cumnet, result
