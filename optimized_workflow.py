"""
优化后的因子测试工作流程
支持大规模因子测试、自动化策略构建
"""

import pandas as pd
import numpy as np
from rqfactor import *
from factor_processing_utils import *
from data_manager import data_manager
from factor_engine import factor_engine
from strategy_tester import strategy_tester


def define_factor_library():
    """定义因子库"""

    # 基本面因子
    fundamental_factors = {
        "cfoa_mrq": {
            "func": Factor("cash_flow_from_operating_activities_mrq_0")
            / Factor("total_assets_mrq_0"),
            "type": "fundamental",
        },
        "atdy_mrq": {
            "func": Factor("operating_revenue_mrq_0") / Factor("total_assets_mrq_0")
            - Factor("operating_revenue_mrq_4") / Factor("total_assets_mrq_4"),
            "type": "fundamental",
        },
        "ccr_mrq": {
            "func": Factor("cash_flow_from_operating_activities_mrq_0")
            / Factor("current_liabilities_mrq_0"),
            "type": "fundamental",
        },
        "op_q_yoy_mrq": {
            "func": (
                Factor("profit_from_operation_mrq_0")
                - Factor("profit_from_operation_mrq_4")
            )
            / ABS(Factor("profit_from_operation_mrq_4")),
            "type": "fundamental",
        },
        "dividend_yield_ttm": {
            "func": Factor("dividend_yield_ttm"),
            "type": "fundamental",
        },
        "pe_ttm": {
            "func": Factor("market_cap_3")
            / Factor("net_profit_parent_company_lyr_0")
            * -1,
            "type": "fundamental",
        },
        "ep_ttm": {
            "func": Factor("net_profit_parent_company_lyr_0") / Factor("market_cap_3"),
            "type": "fundamental",
        },
        "opr_mrq": {
            "func": Factor("profit_from_operation_mrq_0")
            / Factor("operating_revenue_mrq_0"),
            "type": "fundamental",
        },
    }

    # 技术面因子
    HIGH = Factor("high")
    LOW = Factor("low")
    OPEN = Factor("open")
    CLOSE = Factor("close")
    VOLUME = Factor("volume")
    TURNOVER = Factor("total_turnover")

    def turnover_rate(order_book_ids, start_date, end_date):
        return (
            get_turnover_rate(order_book_ids, start_date, end_date, fields="today")
            .today.unstack("order_book_id")
            .reindex(
                columns=order_book_ids,
                index=pd.to_datetime(get_trading_dates(start_date, end_date)),
            )
        )

    DAILY_TURNOVER_RATE = UserDefinedLeafFactor("DAILY_TURNOVER_RATE", turnover_rate)

    technical_factors = {
        "liq_turn_avg": {"func": MA(DAILY_TURNOVER_RATE, 20), "type": "technical"},
        "liq_turn_std": {"func": STD(DAILY_TURNOVER_RATE, 20), "type": "technical"},
        "liq_zamount": {
            "func": MA(TURNOVER, 20) / STD(TURNOVER, 20),
            "type": "technical",
        },
        "corr_price_turn": {
            "func": CORR(CLOSE, DAILY_TURNOVER_RATE, 20) * -1,
            "type": "technical",
        },
        "vol_highlow_std": {"func": STD(HIGH / LOW, 20) * -1, "type": "technical"},
        "vol_up_shadow_std": {
            "func": STD((HIGH - MAX(OPEN, CLOSE)) / HIGH, 20),
            "type": "technical",
        },
        "mmt_normal_M": {
            "func": (CLOSE / DELAY(CLOSE, 20) - 1) * -1,
            "type": "technical",
        },
        "mmt_reverse_5d": {
            "func": (CLOSE / DELAY(CLOSE, 5) - 1) * -1,
            "type": "technical",
        },
    }

    # 合并所有因子
    all_factors = {}
    all_factors.update(fundamental_factors)
    all_factors.update(technical_factors)

    return all_factors


def run_complete_factor_test(
    start_date="2015-01-01",
    end_date="2025-07-01",
    index_item="000852.XSHG",
    change_day=20,
):
    """运行完整的因子测试流程"""

    print("=== 开始完整因子测试流程 ===")

    # 1. 初始化基础数据
    print("1. 初始化基础数据...")
    stock_universe = INDEX_FIX(start_date, end_date, index_item)
    stock_list = stock_universe.columns.tolist()
    date_list = stock_universe.index.tolist()

    print(f"股票池大小: {len(stock_list)}")
    print(f"时间范围: {start_date} 到 {end_date}")
    print(f"交易日数量: {len(date_list)}")

    # 2. 定义因子库
    print("2. 定义因子库...")
    factor_library = define_factor_library()
    print(f"因子总数: {len(factor_library)}")

    # 按类型统计
    factor_types = {}
    for name, info in factor_library.items():
        factor_type = info["type"]
        if factor_type not in factor_types:
            factor_types[factor_type] = 0
        factor_types[factor_type] += 1

    for factor_type, count in factor_types.items():
        print(f"  {factor_type}: {count}个")

    # 3. 单因子测试
    print("3. 开始单因子测试...")
    factor_results, ic_results, ic_summary = strategy_tester.test_single_factors(
        factor_library, stock_list, start_date, end_date, index_item, change_day
    )

    print("单因子测试完成!")
    print(f"成功计算的因子数量: {len(factor_results)}")

    # 显示IC摘要
    if not ic_summary.empty:
        print("\n=== IC摘要 (按IR排序) ===")
        ic_summary_sorted = ic_summary.sort_values("IR", ascending=False)
        print(ic_summary_sorted[["IC mean", "IR", "IC>0", "t_stat"]].round(4))

    # 4. 因子筛选和合成
    print("\n4. 因子筛选和合成...")

    # 筛选优质因子 (IC绝对值>0.02, IR>0.3)
    good_factors_mask = (ic_summary["IC mean"].abs() > 0.02) & (ic_summary["IR"] > 0.3)
    good_factors = ic_summary[good_factors_mask].index.tolist()
    print(f"筛选出优质因子: {len(good_factors)}个")
    print(f"因子名称: {good_factors}")

    # 5. 策略测试
    print("\n5. 开始策略测试...")

    if len(good_factors) >= 2:
        # 批量测试不同策略配置
        strategy_results = strategy_tester.batch_test_strategies(
            composite_methods=["equal_weight", "ic_weight", "ir_weight"],
            rank_ns=[100, 200, 300],
            index_item=index_item,
        )

        print(f"完成策略测试: {len(strategy_results)}个策略")

        # 6. 策略对比
        print("\n6. 生成策略对比报告...")
        comparison_df = strategy_tester.generate_strategy_comparison()

        # 7. 找出最佳策略
        best_strategy_name, best_strategy_result = strategy_tester.get_best_strategy(
            "夏普比率"
        )
        print(f"\n=== 最佳策略 (按夏普比率) ===")
        print(f"策略名称: {best_strategy_name}")
        print(
            f"夏普比率: {best_strategy_result['performance_metrics']['夏普比率']:.4f}"
        )
        print(
            f"年化收益: {best_strategy_result['performance_metrics']['策略年化收益']:.4f}"
        )
        print(
            f"最大回撤: {best_strategy_result['performance_metrics']['最大回撤']:.4f}"
        )

        return {
            "factor_results": factor_results,
            "ic_results": ic_results,
            "ic_summary": ic_summary,
            "strategy_results": strategy_results,
            "best_strategy": (best_strategy_name, best_strategy_result),
            "comparison_df": comparison_df,
        }
    else:
        print("优质因子数量不足，无法进行策略测试")
        return {
            "factor_results": factor_results,
            "ic_results": ic_results,
            "ic_summary": ic_summary,
        }


def run_incremental_test(
    new_factors,
    start_date="2015-01-01",
    end_date="2025-07-01",
    index_item="000852.XSHG",
):
    """增量测试新因子"""

    print("=== 开始增量因子测试 ===")

    # 基础数据
    stock_universe = INDEX_FIX(start_date, end_date, index_item)
    stock_list = stock_universe.columns.tolist()

    # 注册新因子
    factor_engine.register_factors_batch(new_factors)

    # 计算新因子
    new_factor_names = list(new_factors.keys())
    new_results = factor_engine.compute_factors_batch(
        new_factor_names, stock_list, start_date, end_date, index_item
    )

    # 计算IC
    ic_df, ic_summary = factor_engine.compute_ic_batch(new_results, 20, index_item)

    print(f"新增因子测试完成: {len(new_results)}个")
    print("\n=== 新因子IC摘要 ===")
    print(ic_summary[["IC mean", "IR", "IC>0", "t_stat"]].round(4))

    return new_results, ic_df, ic_summary


if __name__ == "__main__":
    # 运行完整测试
    results = run_complete_factor_test()

    print("\n=== 测试完成 ===")
    print("结果已保存到 strategy_results/ 目录")
