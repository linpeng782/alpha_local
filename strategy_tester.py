"""
策略测试框架 - 自动化因子测试和策略构建
支持单因子测试、多因子合成、策略回测
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from factor_engine import factor_engine
from factor_processing_utils import *


class StrategyTester:
    """策略测试框架"""
    
    def __init__(self, results_dir="strategy_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # 测试结果存储
        self.factor_results = {}
        self.ic_results = {}
        self.ic_summary = pd.DataFrame()
        self.strategy_results = {}
    
    def test_single_factors(self, factor_dict, stock_list, start_date, end_date, 
                          index_item, change_day=20, force_recompute=False):
        """测试单因子表现"""
        print("=== 开始单因子测试 ===")
        
        # 注册因子
        factor_engine.register_factors_batch(factor_dict)
        
        # 批量计算因子
        factor_names = list(factor_dict.keys())
        self.factor_results = factor_engine.compute_factors_batch(
            factor_names, stock_list, start_date, end_date, index_item
        )
        
        # 计算IC
        ic_df, ic_summary_df = factor_engine.compute_ic_batch(
            self.factor_results, change_day, index_item
        )
        
        self.ic_results = ic_df
        self.ic_summary = ic_summary_df
        
        # 保存结果
        self._save_single_factor_results()
        
        # 生成报告
        self._generate_single_factor_report()
        
        return self.factor_results, self.ic_results, self.ic_summary
    
    def create_composite_factor(self, factor_names=None, method="equal_weight", 
                              ic_threshold=0.02, ir_threshold=0.3):
        """创建合成因子"""
        if factor_names is None:
            # 根据IC和IR筛选因子
            if self.ic_summary.empty:
                raise ValueError("请先运行单因子测试")
            
            # 筛选条件
            mask = (
                (self.ic_summary['IC mean'].abs() > ic_threshold) & 
                (self.ic_summary['IR'] > ir_threshold)
            )
            factor_names = self.ic_summary[mask].index.tolist()
            print(f"根据IC>{ic_threshold}, IR>{ir_threshold}筛选出因子: {factor_names}")
        
        if not factor_names:
            raise ValueError("没有符合条件的因子")
        
        # 获取因子数据
        selected_factors = {name: self.factor_results[name] for name in factor_names if name in self.factor_results}
        
        if method == "equal_weight":
            # 等权重合成
            composite_factor = self._equal_weight_composite(selected_factors)
        elif method == "ic_weight":
            # IC加权合成
            composite_factor = self._ic_weight_composite(selected_factors, factor_names)
        elif method == "ir_weight":
            # IR加权合成
            composite_factor = self._ir_weight_composite(selected_factors, factor_names)
        else:
            raise ValueError("不支持的合成方法")
        
        return composite_factor, factor_names
    
    def _equal_weight_composite(self, factor_dict):
        """等权重合成因子"""
        composite = pd.DataFrame()
        for name, factor_data in factor_dict.items():
            standardized = standardize(factor_data)
            composite = composite.add(standardized, fill_value=0)
        
        return standardize(composite)
    
    def _ic_weight_composite(self, factor_dict, factor_names):
        """IC加权合成因子"""
        # 获取IC权重
        ic_weights = self.ic_summary.loc[factor_names, 'IC mean'].abs()
        ic_weights = ic_weights / ic_weights.sum()
        
        composite = pd.DataFrame()
        for name, factor_data in factor_dict.items():
            if name in ic_weights.index:
                weight = ic_weights[name]
                standardized = standardize(factor_data) * weight
                composite = composite.add(standardized, fill_value=0)
        
        return standardize(composite)
    
    def _ir_weight_composite(self, factor_dict, factor_names):
        """IR加权合成因子"""
        # 获取IR权重
        ir_weights = self.ic_summary.loc[factor_names, 'IR']
        ir_weights = ir_weights / ir_weights.sum()
        
        composite = pd.DataFrame()
        for name, factor_data in factor_dict.items():
            if name in ir_weights.index:
                weight = ir_weights[name]
                standardized = standardize(factor_data) * weight
                composite = composite.add(standardized, fill_value=0)
        
        return standardize(composite)
    
    def test_strategy(self, composite_factor, index_item, rank_n=200, 
                     rebalance_frequency=20, initial_capital=100000000):
        """测试策略表现"""
        print("=== 开始策略测试 ===")
        
        # 生成买入列表
        buy_list = get_buy_list(composite_factor, rank_n=rank_n)
        
        # 计算权重
        weights = buy_list.div(buy_list.sum(axis=1), axis=0)
        weights = weights.shift(1).dropna(how="all")
        
        # 回测
        account_result = backtest(
            weights, 
            rebalance_frequency=rebalance_frequency,
            initial_capital=initial_capital
        )
        
        # 绩效分析
        performance_cumnet, performance_metrics = get_performance_analysis(
            account_result, benchmark_index=index_item
        )
        
        # 保存结果
        strategy_result = {
            'account_result': account_result,
            'performance_cumnet': performance_cumnet,
            'performance_metrics': performance_metrics,
            'weights': weights,
            'buy_list': buy_list
        }
        
        self.strategy_results['composite'] = strategy_result
        
        return strategy_result
    
    def batch_test_strategies(self, composite_methods=['equal_weight', 'ic_weight', 'ir_weight'],
                            rank_ns=[100, 200, 300], index_item="000852.XSHG"):
        """批量测试不同策略配置"""
        print("=== 开始批量策略测试 ===")
        
        results = {}
        
        for method in composite_methods:
            for rank_n in rank_ns:
                strategy_name = f"{method}_rank{rank_n}"
                print(f"测试策略: {strategy_name}")
                
                try:
                    # 创建合成因子
                    composite_factor, factor_names = self.create_composite_factor(method=method)
                    
                    # 测试策略
                    strategy_result = self.test_strategy(
                        composite_factor, index_item, rank_n=rank_n
                    )
                    
                    strategy_result['method'] = method
                    strategy_result['rank_n'] = rank_n
                    strategy_result['factor_names'] = factor_names
                    
                    results[strategy_name] = strategy_result
                    
                except Exception as e:
                    print(f"策略 {strategy_name} 测试失败: {e}")
        
        self.strategy_results.update(results)
        return results
    
    def _save_single_factor_results(self):
        """保存单因子测试结果"""
        # 保存IC结果
        if not self.ic_results.empty:
            self.ic_results.to_pickle(self.results_dir / "ic_results.pkl")
        
        if not self.ic_summary.empty:
            self.ic_summary.to_pickle(self.results_dir / "ic_summary.pkl")
            self.ic_summary.to_csv(self.results_dir / "ic_summary.csv")
    
    def _generate_single_factor_report(self):
        """生成单因子测试报告"""
        if self.ic_summary.empty:
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # IC均值分布
        self.ic_summary['IC mean'].plot(kind='bar', ax=axes[0,0], title='IC均值分布')
        axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # IR分布
        self.ic_summary['IR'].plot(kind='bar', ax=axes[0,1], title='IR分布')
        axes[0,1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        
        # IC>0占比
        self.ic_summary['IC>0'].plot(kind='bar', ax=axes[1,0], title='IC>0占比')
        axes[1,0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        
        # t统计量
        self.ic_summary['t_stat'].plot(kind='bar', ax=axes[1,1], title='t统计量')
        axes[1,1].axhline(y=2, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "single_factor_report.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_strategy_comparison(self):
        """生成策略对比报告"""
        if not self.strategy_results:
            print("没有策略结果可供对比")
            return
        
        # 收集所有策略的绩效指标
        comparison_data = []
        for strategy_name, result in self.strategy_results.items():
            metrics = result['performance_metrics'].copy()
            metrics['strategy_name'] = strategy_name
            metrics['method'] = result.get('method', 'unknown')
            metrics['rank_n'] = result.get('rank_n', 'unknown')
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data).set_index('strategy_name')
        
        # 保存对比结果
        comparison_df.to_csv(self.results_dir / "strategy_comparison.csv")
        
        # 生成对比图表
        key_metrics = ['策略年化收益', '夏普比率', '最大回撤', '信息比率']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(key_metrics):
            if metric in comparison_df.columns:
                comparison_df[metric].plot(kind='bar', ax=axes[i], title=metric)
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "strategy_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
   

# 全局策略测试器实例
strategy_tester = StrategyTester()
