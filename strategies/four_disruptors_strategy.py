"""
四大搅屎棍策略 - KDCJ版本重构
基于YCZZ项目的7_1_四大搅屎棍_选.py重构而来

策略逻辑：
1. 计算全市场股票乖离率
2. 按行业分组计算宽度比例
3. 过滤"搅屎棍"行业（银行、有色金属、钢铁、煤炭）
4. 选择小市值股票并应用ROE/ROA筛选

重构适配：
- 使用rqdatac数据接口替代hkcodex
- 采用KDCJ项目的代码风格和函数结构
- 添加向量化操作优化性能
- 集成KDCJ的数据路径管理系统
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from rqdatac import *
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")

# 导入KDCJ工具函数
from factor_utils.factor_analysis import (
    get_data_path, 
    get_new_stock_filter, 
    get_st_filter,
    get_suspended_filter,
    get_limit_up_filter
)

class FourDisruptorsStrategy:
    """四大搅屎棍策略类"""
    
    def __init__(self, start_date, end_date, index_item='000985.SH'):
        """
        初始化策略参数
        
        参数:
            start_date: 开始日期
            end_date: 结束日期  
            index_item: 基准指数代码
        """
        self.start_date = start_date
        self.end_date = end_date
        self.index_item = index_item
        
        # 策略参数
        self.bias_window = 20  # 乖离率计算窗口
        self.top_industries = 5  # 选择行业宽度前N名
        self.roe_threshold = 0.15  # ROE阈值
        self.roa_threshold = 0.1   # ROA阈值
        self.new_stock_threshold = 375  # 新股过滤天数
        
        # 搅屎棍行业代码（申万一级）
        self.disruptor_industries = [
            '801780',  # 银行
            '801050',  # 有色金属  
            '801950',  # 钢铁
            '801040'   # 煤炭
        ]
        
        # 申万一级行业映射
        self.sw_industry_map = self._get_sw_industry_mapping()
        
    def _get_sw_industry_mapping(self):
        """获取申万一级行业映射"""
        # 这里应该根据实际情况获取申万行业映射
        # 暂时使用示例映射
        return {
            '801010': '农林牧渔',
            '801020': '采掘', 
            '801030': '化工',
            '801040': '钢铁',
            '801050': '有色金属',
            '801080': '电子',
            '801110': '家用电器',
            '801120': '食品饮料',
            '801130': '纺织服装',
            '801140': '轻工制造',
            '801150': '医药生物',
            '801160': '公用事业',
            '801170': '交通运输',
            '801180': '房地产',
            '801200': '商业贸易',
            '801210': '休闲服务',
            '801230': '综合',
            '801710': '建筑材料',
            '801720': '建筑装饰',
            '801730': '电气设备',
            '801740': '国防军工',
            '801750': '计算机',
            '801760': '传媒',
            '801770': '通信',
            '801780': '银行',
            '801790': '非银金融',
            '801880': '汽车',
            '801890': '机械设备',
            '801950': '钢铁'
        }
    
    def calculate_bias_ratio(self, stock_list, trade_dates):
        """
        计算股票乖离率
        
        参数:
            stock_list: 股票列表
            trade_dates: 交易日列表
            
        返回:
            乖离率DataFrame
        """
        print("正在计算股票乖离率...")
        
        # 获取股票价格数据
        price_data = get_price(
            stock_list,
            start_date=trade_dates[0] - timedelta(days=self.bias_window + 10),
            end_date=trade_dates[-1],
            fields=['close']
        )['close']
        
        # 计算移动平均线
        ma_data = price_data.rolling(window=self.bias_window).mean()
        
        # 计算乖离率：(当前价格 - 移动平均) / 移动平均
        bias_ratio = (price_data - ma_data) / ma_data
        
        # 只保留交易日的数据
        bias_ratio = bias_ratio.reindex(trade_dates)
        
        # 将乖离率转换为0/1信号（大于0为1，小于等于0为0）
        bias_signal = (bias_ratio > 0).astype(int)
        
        return bias_signal
    
    def get_industry_width_ratio(self, bias_signal, stock_list, trade_date):
        """
        计算行业宽度比例
        
        参数:
            bias_signal: 乖离率信号
            stock_list: 股票列表
            trade_date: 交易日期
            
        返回:
            行业宽度比例Series
        """
        # 获取股票行业分类
        industry_data = shenwan_instrument_industry(stock_list, date=trade_date)
        
        # 创建当日数据DataFrame
        day_data = pd.DataFrame({
            'bias_signal': bias_signal.loc[trade_date, stock_list],
            'industry': [industry_data.get(stock, '') for stock in stock_list]
        }).dropna()
        
        # 按行业分组计算宽度比例
        industry_stats = day_data.groupby('industry').agg({
            'bias_signal': ['sum', 'count']
        })
        
        # 计算比例：乖离率大于0的股票数 / 该行业总股票数
        industry_width = (industry_stats[('bias_signal', 'sum')] / 
                         industry_stats[('bias_signal', 'count')] * 100).round()
        
        return industry_width
    
    def filter_disruptor_industries(self, industry_width):
        """
        过滤搅屎棍行业
        
        参数:
            industry_width: 行业宽度Series
            
        返回:
            是否包含搅屎棍行业的布尔值
        """
        top_industries = industry_width.nlargest(self.top_industries).index.tolist()
        
        # 检查是否包含搅屎棍行业
        has_disruptors = any(industry in top_industries for industry in self.disruptor_industries)
        
        if has_disruptors:
            disruptor_names = [self.sw_industry_map.get(ind, ind) 
                             for ind in self.disruptor_industries if ind in top_industries]
            print(f"发现搅屎棍行业: {disruptor_names}, 跳过本日")
            return True
        
        return False
    
    def select_small_cap_stocks(self, trade_date, previous_date):
        """
        选择小市值股票并应用基本面筛选
        
        参数:
            trade_date: 交易日期
            previous_date: 前一交易日
            
        返回:
            筛选后的股票列表
        """
        # 获取中证1000成分股作为初始股票池
        stock_pool = index_components('000852.CSI', date=trade_date)
        
        if not stock_pool:
            print(f"{trade_date}: 获取指数成分股失败")
            return []
        
        # 应用各种过滤器
        stock_pool = self._apply_stock_filters(stock_pool, [trade_date], previous_date)
        
        if not stock_pool:
            return []
        
        # 获取基本面数据进行筛选
        fundamental_data = get_fundamentals(
            query(fundamentals.eod_derivative_indicator.roe,
                  fundamentals.eod_derivative_indicator.roa)
            .filter(fundamentals.eod_derivative_indicator.stockcode.in_(stock_pool)),
            date=previous_date
        )
        
        # 应用ROE和ROA筛选
        qualified_stocks = fundamental_data[
            (fundamental_data['roe'] > self.roe_threshold) & 
            (fundamental_data['roa'] > self.roa_threshold)
        ]['stockcode'].tolist()
        
        if not qualified_stocks:
            return []
        
        # 获取市值数据并排序
        market_cap_data = get_fundamentals(
            query(fundamentals.eod_derivative_indicator.market_cap)
            .filter(fundamentals.eod_derivative_indicator.stockcode.in_(qualified_stocks)),
            date=previous_date
        )
        
        # 按市值升序排序，选择小市值股票
        sorted_stocks = market_cap_data.sort_values('market_cap')['stockcode'].tolist()
        
        return sorted_stocks[:50]  # 返回前50只小市值股票
    
    def _apply_stock_filters(self, stock_list, date_list, previous_date):
        """
        应用股票过滤器
        
        参数:
            stock_list: 股票列表
            date_list: 日期列表  
            previous_date: 前一交易日
            
        返回:
            过滤后的股票列表
        """
        # 过滤新股
        new_stock_filter = get_new_stock_filter(
            stock_list, date_list, self.new_stock_threshold
        )
        
        # 过滤ST股票
        st_filter = get_st_filter(stock_list, date_list)
        
        # 过滤停牌股票
        suspended_filter = get_suspended_filter(stock_list, date_list)
        
        # 过滤涨停股票
        limit_up_filter = get_limit_up_filter(stock_list, date_list)
        
        # 综合过滤条件
        combined_filter = (new_stock_filter & st_filter & 
                          suspended_filter & limit_up_filter)
        
        if combined_filter.empty:
            return []
        
        # 获取通过过滤的股票
        filtered_stocks = combined_filter.loc[date_list[0]]
        qualified_stocks = filtered_stocks[filtered_stocks].index.tolist()
        
        return qualified_stocks
    
    def run_strategy(self):
        """
        运行策略主逻辑
        
        返回:
            选股结果字典
        """
        print(f"开始运行四大搅屎棍策略: {self.start_date} 至 {self.end_date}")
        
        # 获取交易日历
        trade_dates = get_trading_dates(self.start_date, self.end_date)
        
        # 获取股票池
        all_stocks = all_instruments(type='CS', date=self.end_date)['order_book_id'].tolist()
        
        # 计算全市场股票乖离率
        bias_signal = self.calculate_bias_ratio(all_stocks, trade_dates)
        
        # 存储选股结果
        selection_results = {}
        
        for i, trade_date in enumerate(trade_dates):
            if i == 0:  # 跳过第一个交易日
                continue
                
            previous_date = trade_dates[i-1]
            
            print(f"\n处理日期: {trade_date}")
            
            # 获取当日可交易股票
            available_stocks = [stock for stock in all_stocks 
                              if not pd.isna(bias_signal.loc[trade_date, stock])]
            
            # 计算行业宽度比例
            industry_width = self.get_industry_width_ratio(
                bias_signal, available_stocks, trade_date
            )
            
            # 显示行业宽度信息
            top_industries = industry_width.nlargest(self.top_industries)
            industry_names = [self.sw_industry_map.get(ind, ind) for ind in top_industries.index]
            print(f"行业宽度前{self.top_industries}名: {dict(zip(industry_names, top_industries.values))}")
            print(f"全市场宽度: {industry_width.mean():.2f}")
            
            # 检查是否包含搅屎棍行业
            if self.filter_disruptor_industries(industry_width):
                selection_results[trade_date] = []
                continue
            
            # 选择小市值股票
            selected_stocks = self.select_small_cap_stocks(trade_date, previous_date)
            
            selection_results[trade_date] = selected_stocks
            print(f"选中股票数量: {len(selected_stocks)}")
            
            # 保存当日选股结果
            self._save_daily_results(trade_date, selected_stocks)
        
        print("\n策略运行完成!")
        return selection_results
    
    def _save_daily_results(self, trade_date, selected_stocks):
        """
        保存每日选股结果
        
        参数:
            trade_date: 交易日期
            selected_stocks: 选中的股票列表
        """
        # 使用KDCJ的数据路径管理
        output_dir = "/Users/didi/KDCJ/alpha_local/outputs/stock_selection/four_disruptors"
        os.makedirs(output_dir, exist_ok=True)
        
        file_path = os.path.join(output_dir, f"{trade_date.strftime('%Y%m%d')}.txt")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for stock in selected_stocks:
                f.write(f"{stock}\n")

def main():
    """主函数示例"""
    # 初始化策略
    strategy = FourDisruptorsStrategy(
        start_date='2024-01-01',
        end_date='2024-12-31',
        index_item='000985.SH'
    )
    
    # 运行策略
    results = strategy.run_strategy()
    
    # 输出统计信息
    total_days = len(results)
    active_days = len([day for day, stocks in results.items() if stocks])
    
    print(f"\n=== 策略统计 ===")
    print(f"总交易日数: {total_days}")
    print(f"有效选股日数: {active_days}")
    print(f"选股成功率: {active_days/total_days*100:.2f}%")

if __name__ == "__main__":
    main()
