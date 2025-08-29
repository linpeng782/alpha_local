"""
策略对比分析工具
用于对比两个策略的相对表现，而不是与指数基准的对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import pickle
import os
import sys
import statsmodels.api as sm
from datetime import datetime

# 添加路径以导入自定义模块
sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils.path_manager import get_data_path
from alpha_local.core.factor_config import get_factor_config
from alpha_local.core.analyze_single_factor import get_stock_universe

# 设置中文字体
rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def load_account_result(file_path):
    """加载保存的回测结果"""
    return pd.read_pickle(file_path)


def compare_two_strategies(
    benchmark_result,
    strategy_result,
    benchmark_name="基准策略",
    strategy_name="测试策略",
    benchmark_neutralize=False,
    strategy_neutralize=True,
    index_item="000985.XSHG",
    start_date=None,
    end_date=None,
    rf=0.03,
    show_plot=True,
    save_plot=True,
):
    """
    对比两个策略的表现

    参数:
    benchmark_result: 基准策略的account_result
    strategy_result: 测试策略的account_result
    benchmark_name: 基准策略名称
    strategy_name: 测试策略名称
    rf: 无风险利率
    show_plot: 是否显示图表
    save_plot: 是否保存图表
    output_dir: 图表保存目录
    """

    # 构建对比数据，使用更具描述性的列名
    benchmark_label = f"{benchmark_name}_{'neutral' if benchmark_neutralize else 'raw'}"
    strategy_label = f"{strategy_name}_{'neutral' if strategy_neutralize else 'raw'}"

    performance = pd.concat(
        [
            benchmark_result["total_account_asset"].to_frame(benchmark_label),
            strategy_result["total_account_asset"].to_frame(strategy_label),
        ],
        axis=1,
    )

    # 计算日收益率
    daily_returns = performance.pct_change().dropna()

    # 计算累计收益（从1开始）
    cumulative_returns = (1 + daily_returns).cumprod()

    # 计算超额收益（策略相对基准）
    cumulative_returns["超额收益"] = (
        cumulative_returns[strategy_label] / cumulative_returns[benchmark_label]
    )

    # 计算各项指标
    daily_pct_change = cumulative_returns.pct_change().dropna()

    # 策略表现指标
    strategy_final_return = cumulative_returns[strategy_label].iloc[-1] - 1
    benchmark_final_return = cumulative_returns[benchmark_label].iloc[-1] - 1
    excess_final_return = cumulative_returns["超额收益"].iloc[-1] - 1

    # 年化收益率
    trading_days = len(cumulative_returns)
    strategy_annual_return = (1 + strategy_final_return) ** (252 / trading_days) - 1
    benchmark_annual_return = (1 + benchmark_final_return) ** (252 / trading_days) - 1
    excess_annual_return = (1 + excess_final_return) ** (252 / trading_days) - 1

    # 波动率
    strategy_volatility = daily_pct_change[strategy_label].std() * np.sqrt(252)
    benchmark_volatility = daily_pct_change[benchmark_label].std() * np.sqrt(252)

    # 夏普比率
    strategy_sharpe = (strategy_annual_return - rf) / strategy_volatility
    benchmark_sharpe = (benchmark_annual_return - rf) / benchmark_volatility

    # 前三大回撤计算
    def calculate_top3_drawdowns(cumulative_series):
        running_max = np.maximum.accumulate(cumulative_series)
        drawdown_ratio = (running_max - cumulative_series) / running_max
        drawdown_periods = []

        # 找到所有局部峰值点（新高点）
        is_new_high = cumulative_series == running_max
        peak_indices = cumulative_series[is_new_high].index

        # 对每个峰值，找到后续的最大回撤
        for k in range(len(peak_indices) - 1):
            peak_idx = peak_indices[k]
            next_peak_idx = peak_indices[k + 1]

            # 在这个峰值到下个峰值之间找最大回撤
            period_mask = (cumulative_series.index >= peak_idx) & (
                cumulative_series.index < next_peak_idx
            )
            if period_mask.sum() > 1:  # 确保有足够的数据点
                period_cumulative = cumulative_series[period_mask]
                period_drawdown = drawdown_ratio[period_mask]

                if (
                    len(period_drawdown) > 0 and period_drawdown.max() > 0.01
                ):  # 只考虑回撤超过1%的情况
                    trough_idx = period_drawdown.idxmax()
                    peak_value = cumulative_series[peak_idx]
                    trough_value = cumulative_series[trough_idx]
                    drawdown_pct = period_drawdown.max()

                    drawdown_periods.append(
                        {
                            "peak_date": peak_idx,
                            "trough_date": trough_idx,
                            "peak_value": peak_value,
                            "trough_value": trough_value,
                            "drawdown": drawdown_pct,
                        }
                    )

        # 处理最后一个峰值到结束的回撤
        if len(peak_indices) > 0:
            last_peak_idx = peak_indices[-1]
            period_mask = cumulative_series.index >= last_peak_idx
            period_cumulative = cumulative_series[period_mask]
            period_drawdown = drawdown_ratio[period_mask]

            if len(period_drawdown) > 0 and period_drawdown.max() > 0.01:
                trough_idx = period_drawdown.idxmax()
                peak_value = cumulative_series[last_peak_idx]
                trough_value = cumulative_series[trough_idx]
                drawdown_pct = period_drawdown.max()

                drawdown_periods.append(
                    {
                        "peak_date": last_peak_idx,
                        "trough_date": trough_idx,
                        "peak_value": peak_value,
                        "trough_value": trough_value,
                        "drawdown": drawdown_pct,
                    }
                )

        # 按回撤幅度排序，返回前三大
        drawdown_periods.sort(key=lambda x: x["drawdown"], reverse=True)
        return drawdown_periods[:5]

    # 计算前三大回撤
    strategy_top3_dd = calculate_top3_drawdowns(cumulative_returns[strategy_label])
    benchmark_top3_dd = calculate_top3_drawdowns(cumulative_returns[benchmark_label])

    # 构建结果字典
    results = {
        "收益对比": {
            "策略年化收益率": f"{strategy_annual_return:.2%}",
            "基准年化收益率": f"{benchmark_annual_return:.2%}",
        },
        "波动对比": {
            "策略年化波动率": f"{strategy_volatility:.2%}",
            "基准年化波动率": f"{benchmark_volatility:.2%}",
        },
        "夏普比率对比": {
            "策略夏普比率": f"{strategy_sharpe:.3f}",
            "基准夏普比率": f"{benchmark_sharpe:.3f}",
        },
    }

    # 基准策略回撤总结
    benchmark_drawdown_summary = {}
    for i, dd in enumerate(benchmark_top3_dd, 1):
        benchmark_drawdown_summary[f"第{i}大回撤"] = (
            f"{dd['drawdown']:.2%} ({dd['peak_date'].strftime('%Y-%m-%d')} ~ {dd['trough_date'].strftime('%Y-%m-%d')})"
        )

    # 测试策略回撤总结
    strategy_drawdown_summary = {}
    for i, dd in enumerate(strategy_top3_dd, 1):
        strategy_drawdown_summary[f"第{i}大回撤"] = (
            f"{dd['drawdown']:.2%} ({dd['peak_date'].strftime('%Y-%m-%d')} ~ {dd['trough_date'].strftime('%Y-%m-%d')})"
        )

    # 添加到结果字典
    results[f"{benchmark_name}回撤分析"] = benchmark_drawdown_summary
    results[f"{strategy_name}回撤分析"] = strategy_drawdown_summary

    # 打印结果
    print("=" * 60)
    print(f"策略对比分析报告: {strategy_name} vs {benchmark_name}")
    print("=" * 60)

    for category, metrics in results.items():
        print(f"\n【{category}】")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    # 绘制对比图表
    if show_plot or save_plot:
        # 创建图表，左侧为统一表格，右侧为收益曲线
        fig = plt.figure(figsize=(26, 12))

        # 创建网格布局：1行2列，最大化右侧图表空间
        gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 3.2], hspace=0.1, wspace=0.05)

        # 左侧：统计表格
        ax_table = fig.add_subplot(gs[0, 0])
        ax_table.axis("off")  # 隐藏轴

        # 准备统一表格数据：16行4列
        table_data = []

        # 表头
        table_data.append(["指标", f"{benchmark_name}", f"{strategy_name}", "备注"])

        # 基本指标
        table_data.append(
            [
                "年化收益",
                f"{benchmark_annual_return:.2%}",
                f"{strategy_annual_return:.2%}",
                "",
            ]
        )
        table_data.append(
            [
                "年化波动率",
                f"{benchmark_volatility:.2%}",
                f"{strategy_volatility:.2%}",
                "",
            ]
        )
        table_data.append(
            ["夏普比率", f"{benchmark_sharpe:.3f}", f"{strategy_sharpe:.3f}", ""]
        )

        # 空行分隔
        table_data.append(["", "", "", ""])

        # market_cap策略回撤分析
        table_data.append([f"{benchmark_name}", "回撤幅度", "起始日期", "结束日期"])
        for i, dd in enumerate(benchmark_top3_dd[:5], 1):
            table_data.append(
                [
                    f"第{i}大回撤",
                    f"{dd['drawdown']:.2%}",
                    dd["peak_date"].strftime("%Y-%m-%d"),
                    dd["trough_date"].strftime("%Y-%m-%d"),
                ]
            )

        # combo策略回撤分析
        table_data.append([f"{strategy_name}", "回撤幅度", "起始日期", "结束日期"])
        for i, dd in enumerate(strategy_top3_dd[:5], 1):
            table_data.append(
                [
                    f"第{i}大回撤",
                    f"{dd['drawdown']:.2%}",
                    dd["peak_date"].strftime("%Y-%m-%d"),
                    dd["trough_date"].strftime("%Y-%m-%d"),
                ]
            )

        # 创建表格
        table = ax_table.table(
            cellText=table_data,
            cellLoc="center",
            loc="center",
            colWidths=[0.3, 0.25, 0.25, 0.2],
        )

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        # 设置表头样式
        for i in range(4):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # 设置回撤分析标题样式
        # market_cap回撤分析标题行（第6行）
        for i in range(4):
            table[(5, i)].set_facecolor("blue")
            table[(5, i)].set_text_props(weight="bold", color="white")

        # combo回撤分析标题行（第12行）
        for i in range(4):
            table[(11, i)].set_facecolor("red")
            table[(11, i)].set_text_props(weight="bold", color="white")

        # 右侧：收益曲线图
        ax_chart = fig.add_subplot(gs[0, 1])
        ax2 = ax_chart.twinx()

        # 绘制曲线
        line1 = ax_chart.plot(
            cumulative_returns.index,
            cumulative_returns[benchmark_label],
            label=benchmark_name,
            linewidth=2.5,
            color="blue",
            alpha=0.8,
        )
        line2 = ax_chart.plot(
            cumulative_returns.index,
            cumulative_returns[strategy_label],
            label=strategy_name,
            linewidth=2.5,
            color="red",
            alpha=0.8,
        )

        # 超额收益曲线
        line3 = ax2.plot(
            cumulative_returns.index,
            cumulative_returns["超额收益"],
            label="超额收益",
            linewidth=2,
            color="green",
            alpha=0.7,
            linestyle="--",
        )
        ax2.axhline(y=1, color="gray", linestyle=":", alpha=0.5, linewidth=1)

        # 设置标题和标签
        ax_chart.set_title(
            f"策略对比分析: {strategy_name} vs {benchmark_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax_chart.set_xlabel("时间", fontsize=12)
        ax_chart.set_ylabel("累计净值", fontsize=12, color="black", labelpad=-5)
        ax2.set_ylabel("累计超额收益", fontsize=12, color="green")

        # 合并图例
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax_chart.legend(lines, labels, loc="upper left", fontsize=11)

        # 网格和格式
        ax_chart.grid(True, alpha=0.3)
        ax_chart.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax_chart.xaxis.set_major_locator(mdates.YearLocator())

        # 设置右轴颜色
        ax2.tick_params(axis="y", labelcolor="green")

        plt.tight_layout()

    if save_plot:
        # 使用get_data_path生成保存路径
        if start_date and end_date:
            filepath = get_data_path(
                "strategy_comparison",
                benchmark_name=benchmark_name,
                benchmark_neutralize=benchmark_neutralize,
                strategy_name=strategy_name,
                strategy_neutralize=strategy_neutralize,
                index_item=index_item,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            # 如果没有提供日期，使用时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"strategy_comparison_{timestamp}.png"
            filepath = get_data_path(
                "strategy_comparison",
                filename=filename,
                index_item=index_item,
            )
        
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"\n✅对比图表已保存到: {filepath}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return results, cumulative_returns


def get_comparison_config(scenario="scenario1"):
    """
    获取策略对比配置
    
    参数:
        scenario: 对比场景
            - "scenario1": 同一因子的中性化 vs 非中性化
            - "scenario2": 不同因子的对比
            - "scenario3": 多因子策略 vs 单因子策略
            - "custom": 自定义配置
    
    返回:
        (benchmark_name, benchmark_neutralize, strategy_name, strategy_neutralize)
    """
    
    configs = {
        "scenario1": (
            "high_low_std_504", False,  # 基准: 非中性化版本
            "high_low_std_504", True    # 策略: 中性化版本
        ),
        "scenario2": (
            "market_cap", False,        # 基准: 市值因子
            "high_low_std_504", True    # 策略: 波动率因子
        ),
        "scenario3": (
            "combo_2", True,            # 基准: 多因子策略
            "market_cap", False         # 策略: 单因子策略
        )
    }
    
    if scenario in configs:
        return configs[scenario]
    else:
        # 自定义配置，返回默认值
        return "high_low_std_504", False, "high_low_std_504", True


def main():
    """主函数：执行策略对比分析"""

    # 基本参数设置
    index_item = "000985.XSHG"
    start_date = "2015-01-01"
    end_date = "2025-07-01"

    # 获取股票池和实际回测日期
    stock_universe = get_stock_universe(start_date, end_date, index_item)
    universe_start = stock_universe.index[0].strftime("%F")
    universe_end = stock_universe.index[-1].strftime("%F")
    backtest_start_date = universe_start

    # =================================================================
    # 策略对比配置区域
    # =================================================================
    
    # 选择对比场景：
    # "scenario1": 同一因子的中性化 vs 非中性化
    # "scenario2": 不同因子的对比 (market_cap vs high_low_std_504)
    # "scenario3": 多因子策略 vs 单因子策略 (combo_2 vs market_cap)
    
    current_scenario = "scenario1"  # 修改这里来切换不同的对比场景
    
    # 获取配置
    benchmark_name, benchmark_neutralize, strategy_name, strategy_neutralize = get_comparison_config(current_scenario)
    
    # 也可以手动覆盖配置（取消注释即可）：
    # benchmark_name, benchmark_neutralize = "high_low_std_504", False
    # strategy_name, strategy_neutralize = "high_low_std_504", True
    
    # =================================================================
    
    # 从配置文件获取因子信息
    benchmark_config = get_factor_config(
        benchmark_name, neutralize=benchmark_neutralize
    )
    strategy_config = get_factor_config(strategy_name, neutralize=strategy_neutralize)
    benchmark_direction = benchmark_config["direction"]
    strategy_direction = strategy_config["direction"]
    
    # 显示对比配置
    print(f"\n=== 策略对比配置 ===")
    print(f"基准策略: {benchmark_name} (中性化: {benchmark_neutralize}, 方向: {benchmark_direction})")
    print(f"对比策略: {strategy_name} (中性化: {strategy_neutralize}, 方向: {strategy_direction})")
    print(f"==================\n")

    # 使用get_data_path生成文件路径
    benchmark_file = get_data_path(
        "account_result",
        start_date=backtest_start_date,
        end_date=end_date,
        factor_name=benchmark_name,
        index_item=index_item,
        direction=benchmark_direction,
        neutralize=benchmark_neutralize,
    )

    strategy_file = get_data_path(
        "account_result",
        start_date=backtest_start_date,
        end_date=end_date,
        factor_name=strategy_name,
        index_item=index_item,
        direction=strategy_direction,
        neutralize=strategy_neutralize,
    )

    # 显示生成的文件路径
    print(f"基准策略文件路径: {benchmark_file}")
    print(f"对比策略文件路径: {strategy_file}")

    # 检查文件是否存在
    if not os.path.exists(benchmark_file):
        print(f"⚠️  警告：基准策略文件不存在: {benchmark_file}")
        print(f"    请先运行 {benchmark_name} 因子的回测")
        return

    if not os.path.exists(strategy_file):
        print(f"⚠️  警告：对比策略文件不存在: {strategy_file}")
        print(f"    请先运行 {strategy_name} 因子的回测")
        return

    # 加载数据
    print("加载策略回测结果...")
    benchmark_result = load_account_result(benchmark_file)
    strategy_result = load_account_result(strategy_file)

    # 执行对比分析
    print("执行策略对比分析...")
    results, performance_data = compare_two_strategies(
        benchmark_result=benchmark_result,
        strategy_result=strategy_result,
        benchmark_name=benchmark_name,
        strategy_name=strategy_name,
        benchmark_neutralize=benchmark_neutralize,
        strategy_neutralize=strategy_neutralize,
        index_item=index_item,
        start_date=backtest_start_date,
        end_date=end_date,
        show_plot=False,
        save_plot=True,
    )

    print("\n✅策略对比分析完成！")


if __name__ == "__main__":
    main()
