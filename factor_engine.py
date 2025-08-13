"""
因子计算引擎 - 批量计算和管理因子
支持并行计算、结果缓存、增量更新
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from data_manager import data_manager
from factor_processing_utils import *


class FactorEngine:
    """因子计算引擎"""
    
    def __init__(self, cache_dir="factor_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.raw_dir = self.cache_dir / "raw"
        self.processed_dir = self.cache_dir / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # 因子注册表
        self.factor_registry = {}
        
    def register_factor(self, name, factor_func, factor_type="technical"):
        """注册因子计算函数"""
        self.factor_registry[name] = {
            'func': factor_func,
            'type': factor_type,
            'computed': False
        }
    
    def register_factors_batch(self, factor_dict):
        """批量注册因子"""
        for name, factor_info in factor_dict.items():
            if isinstance(factor_info, dict):
                self.register_factor(name, factor_info['func'], factor_info.get('type', 'technical'))
            else:
                # 兼容旧格式
                self.register_factor(name, factor_info)
    
    def _get_factor_cache_path(self, factor_name, data_params, processed=False):
        """获取因子缓存路径"""
        cache_key = hashlib.md5(str(data_params).encode()).hexdigest()[:16]
        base_dir = self.processed_dir if processed else self.raw_dir
        return base_dir / f"{factor_name}_{cache_key}.pkl"
    
    def _load_factor_cache(self, factor_name, data_params, processed=False):
        """加载因子缓存"""
        cache_path = self._get_factor_cache_path(factor_name, data_params, processed)
        if cache_path.exists():
            try:
                return pd.read_pickle(cache_path)
            except:
                return None
        return None
    
    def _save_factor_cache(self, factor_name, data, data_params, processed=False):
        """保存因子缓存"""
        cache_path = self._get_factor_cache_path(factor_name, data_params, processed)
        data.to_pickle(cache_path)
    
    def compute_single_factor(self, factor_name, stock_list, start_date, end_date, 
                            index_item, force_recompute=False):
        """计算单个因子"""
        data_params = {
            'stock_list_len': len(stock_list),
            'start_date': start_date,
            'end_date': end_date,
            'index_item': index_item
        }
        
        # 检查缓存
        if not force_recompute:
            # 先检查处理后的缓存
            processed_data = self._load_factor_cache(factor_name, data_params, processed=True)
            if processed_data is not None:
                print(f"从缓存加载已处理因子: {factor_name}")
                return processed_data
            
            # 再检查原始缓存
            raw_data = self._load_factor_cache(factor_name, data_params, processed=False)
            if raw_data is not None:
                print(f"从缓存加载原始因子: {factor_name}")
                # 进行数据清洗
                stock_universe = INDEX_FIX(start_date, end_date, index_item)
                processed_data = data_clean(raw_data, stock_universe, index_item)
                # 保存处理后的数据
                self._save_factor_cache(factor_name, processed_data, data_params, processed=True)
                return processed_data
        
        # 计算因子
        if factor_name not in self.factor_registry:
            raise ValueError(f"因子 {factor_name} 未注册")
        
        print(f"计算因子: {factor_name}")
        try:
            factor_func = self.factor_registry[factor_name]['func']
            raw_data = execute_factor(factor_func, stock_list, start_date, end_date)
            
            # 保存原始数据
            self._save_factor_cache(factor_name, raw_data, data_params, processed=False)
            
            # 数据清洗
            stock_universe = INDEX_FIX(start_date, end_date, index_item)
            processed_data = data_clean(raw_data, stock_universe, index_item)
            
            # 保存处理后的数据
            self._save_factor_cache(factor_name, processed_data, data_params, processed=True)
            
            return processed_data
            
        except Exception as e:
            print(f"计算因子 {factor_name} 失败: {e}")
            return None
    
    def compute_factors_batch(self, factor_names, stock_list, start_date, end_date, 
                            index_item, max_workers=2, batch_size=3):
        """批量计算因子（带并发控制）"""
        results = {}
        
        # 分批处理
        factor_batches = [factor_names[i:i+batch_size] for i in range(0, len(factor_names), batch_size)]
        
        for batch_idx, batch in enumerate(factor_batches):
            print(f"处理第 {batch_idx + 1}/{len(factor_batches)} 批因子")
            
            # 使用线程池处理当前批次
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 创建任务
                future_to_factor = {
                    executor.submit(
                        self.compute_single_factor, 
                        factor_name, stock_list, start_date, end_date, index_item
                    ): factor_name 
                    for factor_name in batch
                }
                
                # 收集结果
                for future in tqdm(future_to_factor, desc=f"批次 {batch_idx + 1}"):
                    factor_name = future_to_factor[future]
                    try:
                        result = future.result(timeout=300)  # 5分钟超时
                        if result is not None:
                            results[factor_name] = result
                    except Exception as e:
                        print(f"因子 {factor_name} 计算失败: {e}")
            
            # 批次间等待
            if batch_idx < len(factor_batches) - 1:
                print("批次间等待...")
                time.sleep(10)
        
        return results
    
    def compute_ic_batch(self, factor_results, change_day=20, index_item="000852.XSHG"):
        """批量计算IC"""
        ic_results = {}
        ic_summary_list = []
        
        for factor_name, factor_data in tqdm(factor_results.items(), desc="计算IC"):
            try:
                ic, ic_summary = calc_ic(factor_data, change_day, index_item, factor_name)
                ic_results[factor_name] = ic
                ic_summary_list.append(ic_summary.set_index("name"))
            except Exception as e:
                print(f"计算 {factor_name} IC失败: {e}")
        
        # 合并IC结果
        if ic_results:
            ic_df = pd.concat(ic_results.values(), axis=1)
            ic_df.columns = list(ic_results.keys())
        else:
            ic_df = pd.DataFrame()
        
        if ic_summary_list:
            ic_summary_df = pd.concat(ic_summary_list, axis=0)
        else:
            ic_summary_df = pd.DataFrame()
        
        return ic_df, ic_summary_df
    
    def get_factor_summary(self):
        """获取因子注册摘要"""
        summary = []
        for name, info in self.factor_registry.items():
            summary.append({
                'name': name,
                'type': info['type'],
                'computed': info['computed']
            })
        return pd.DataFrame(summary)
    
    def clear_cache(self, factor_names=None):
        """清空缓存"""
        if factor_names is None:
            # 清空所有缓存
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
                self.raw_dir.mkdir(exist_ok=True)
                self.processed_dir.mkdir(exist_ok=True)
            print("所有因子缓存已清空")
        else:
            # 清空指定因子缓存
            for factor_name in factor_names:
                for cache_file in self.cache_dir.rglob(f"{factor_name}_*.pkl"):
                    cache_file.unlink()
            print(f"已清空因子缓存: {factor_names}")


# 全局因子引擎实例
factor_engine = FactorEngine()
