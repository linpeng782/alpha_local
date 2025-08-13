"""
数据管理器 - 统一管理基础数据获取和缓存
解决API调用频繁和重复计算问题
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import hashlib
from rqdatac import *
from rqfactor import *
import os
from tqdm import tqdm
import time


class DataManager:
    """统一的数据管理器"""
    
    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._base_data_cache = {}
        
    def _get_cache_key(self, *args):
        """生成缓存键"""
        key_str = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _load_cache(self, cache_key):
        """加载缓存数据"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def _save_cache(self, cache_key, data):
        """保存缓存数据"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    
    def get_base_data(self, stock_list, start_date, end_date, fields=None):
        """获取基础数据（价格、成交量等）"""
        cache_key = self._get_cache_key("base_data", len(stock_list), start_date, end_date)
        
        # 尝试从缓存加载
        cached_data = self._load_cache(cache_key)
        if cached_data is not None:
            print(f"从缓存加载基础数据: {cache_key}")
            return cached_data
        
        print("获取基础数据...")
        if fields is None:
            fields = ["open", "high", "low", "close", "volume", "total_turnover"]
        
        base_data = {}
        for field in tqdm(fields, desc="获取基础数据"):
            try:
                data = get_price(
                    stock_list, start_date, end_date, 
                    fields=[field], adjust_type="post"
                )[field].unstack("order_book_id")
                base_data[field] = data
                time.sleep(0.5)  # 避免API限制
            except Exception as e:
                print(f"获取{field}数据失败: {e}")
                base_data[field] = None
        
        # 保存到缓存
        self._save_cache(cache_key, base_data)
        return base_data
    
    def get_fundamental_data(self, stock_list, start_date, end_date, fields=None):
        """获取基本面数据"""
        cache_key = self._get_cache_key("fundamental", len(stock_list), start_date, end_date)
        
        cached_data = self._load_cache(cache_key)
        if cached_data is not None:
            print(f"从缓存加载基本面数据: {cache_key}")
            return cached_data
        
        print("获取基本面数据...")
        if fields is None:
            # 常用基本面字段
            fields = [
                "cash_flow_from_operating_activities_mrq_0",
                "total_assets_mrq_0",
                "operating_revenue_mrq_0",
                "current_liabilities_mrq_0",
                "profit_from_operation_mrq_0",
                "market_cap_3",
                "net_profit_parent_company_lyr_0",
                "dividend_yield_ttm"
            ]
        
        fundamental_data = {}
        for field in tqdm(fields, desc="获取基本面数据"):
            try:
                factor = Factor(field)
                data = execute_factor(factor, stock_list, start_date, end_date)
                fundamental_data[field] = data
                time.sleep(1)  # 基本面数据获取间隔更长
            except Exception as e:
                print(f"获取{field}数据失败: {e}")
                fundamental_data[field] = None
        
        self._save_cache(cache_key, fundamental_data)
        return fundamental_data
    
    def get_market_data(self, stock_list, start_date, end_date):
        """获取市场数据（换手率等）"""
        cache_key = self._get_cache_key("market", len(stock_list), start_date, end_date)
        
        cached_data = self._load_cache(cache_key)
        if cached_data is not None:
            print(f"从缓存加载市场数据: {cache_key}")
            return cached_data
        
        print("获取市场数据...")
        market_data = {}
        
        try:
            # 换手率
            turnover_rate_data = get_turnover_rate(
                stock_list, start_date, end_date, fields="today"
            ).today.unstack("order_book_id")
            market_data["turnover_rate"] = turnover_rate_data
            time.sleep(1)
            
            # 其他市场数据可以在这里添加
            
        except Exception as e:
            print(f"获取市场数据失败: {e}")
        
        self._save_cache(cache_key, market_data)
        return market_data
    
    def clear_cache(self):
        """清空缓存"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
        print("缓存已清空")


# 全局数据管理器实例
data_manager = DataManager()
