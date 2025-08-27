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
import statsmodels.api as sm
from datetime import datetime

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
    rf=0.03,
    show_plot=True,
    save_plot=True,
    output_dir="/Users/didi/KDCJ/alpha_local/outputs/images",
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

    # 构建对比数据
    performance = pd.concat(
        [
            benchmark_result["total_account_asset"].to_frame(benchmark_name),
            strategy_result["total_account_asset"].to_frame(strategy_name),
        ],
        axis=1,
    )

    # 计算日收益率
    daily_returns = performance.pct_change().dropna()

    # 计算累计收益（从1开始）
    cumulative_returns = (1 + daily_returns).cumprod()

    # 计算超额收益（策略相对基准）
    cumulative_returns["超额收益"] = (
        cumulative_returns[strategy_name] / cumulative_returns[benchmark_name]
    )

    # 计算各项指标
    daily_pct_change = cumulative_returns.pct_change().dropna()

    # 策略表现指标
    strategy_final_return = cumulative_returns[strategy_name].iloc[-1] - 1
    benchmark_final_return = cumulative_returns[benchmark_name].iloc[-1] - 1
    excess_final_return = cumulative_returns["超额收益"].iloc[-1] - 1

    # 年化收益率
    trading_days = len(cumulative_returns)
    strategy_annual_return = (1 + strategy_final_return) ** (252 / trading_days) - 1
    benchmark_annual_return = (1 + benchmark_final_return) ** (252 / trading_days) - 1
    excess_annual_return = (1 + excess_final_return) ** (252 / trading_days) - 1

    # 波动率
    strategy_volatility = daily_pct_change[strategy_name].std() * np.sqrt(252)
    benchmark_volatility = daily_pct_change[benchmark_name].std() * np.sqrt(252)

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
    strategy_top3_dd = calculate_top3_drawdowns(cumulative_returns[strategy_name])
    benchmark_top3_dd = calculate_top3_drawdowns(cumulative_returns[benchmark_name])

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
        gs = fig.add_gridspec(1, 2, 
                            width_ratios=[0.8, 3.2], 
                            hspace=0.1, wspace=0.05)

        # 左侧：统计表格
        ax_table = fig.add_subplot(gs[0, 0])
        ax_table.axis("off")  # 隐藏轴

        # 准备统一表格数据：16行4列
        table_data = []
        
        # 表头
        table_data.append(["指标", f"{benchmark_name}", f"{strategy_name}", "备注"])
        
        # 基本指标
        table_data.append(["年化收益", f"{benchmark_annual_return:.2%}", f"{strategy_annual_return:.2%}", ""])
        table_data.append(["年化波动率", f"{benchmark_volatility:.2%}", f"{strategy_volatility:.2%}", ""])
        table_data.append(["夏普比率", f"{benchmark_sharpe:.3f}", f"{strategy_sharpe:.3f}", ""])
        
        # 空行分隔
        table_data.append(["", "", "", ""])
        
        # market_cap策略回撤分析
        table_data.append([f"{benchmark_name}", "回撤幅度", "起始日期", "结束日期"])
        for i, dd in enumerate(benchmark_top3_dd[:5], 1):
            table_data.append([
                f"第{i}大回撤",
                f"{dd['drawdown']:.2%}",
                dd['peak_date'].strftime('%Y-%m-%d'),
                dd['trough_date'].strftime('%Y-%m-%d')
            ])
        
        # combo策略回撤分析  
        table_data.append([f"{strategy_name}", "回撤幅度", "起始日期", "结束日期"])
        for i, dd in enumerate(strategy_top3_dd[:5], 1):
            table_data.append([
                f"第{i}大回撤",
                f"{dd['drawdown']:.2%}",
                dd['peak_date'].strftime('%Y-%m-%d'),
                dd['trough_date'].strftime('%Y-%m-%d')
            ])

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
            cumulative_returns[benchmark_name],
            label=benchmark_name,
            linewidth=2.5,
            color="blue",
            alpha=0.8,
        )
        line2 = ax_chart.plot(
            cumulative_returns.index,
            cumulative_returns[strategy_name],
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
        ax_chart.set_ylabel("累计净值", fontsize=12, color="black")
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
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"strategy_comparison_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"\n✅对比图表已保存到: {filepath}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return results, cumulative_returns


def main():
    """主函数：执行策略对比分析"""

    # 数据文件路径
    result_dir = "/Users/didi/KDCJ/alpha_local/data/account_result"
    backtest_start_date = "2015-01-01"
    end_date = "2025-07-01"
    benchmark_name = "market_cap"
    strategy_name = "combo"

    benchmark_file = os.path.join(
        result_dir,
        f"{backtest_start_date}_{end_date}_{benchmark_name}_account_result.pkl",
    )
    strategy_file = os.path.join(
        result_dir,
        f"{backtest_start_date}_{end_date}_{strategy_name}_account_result.pkl",
    )

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
        show_plot=False,
        save_plot=True,
    )

    print("\n✅策略对比分析完成！")


if __name__ == "__main__":
    main()
