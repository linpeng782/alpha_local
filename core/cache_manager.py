#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子库缓存管理工具
"""

import os
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


class FactorCacheManager:
    """因子缓存管理器"""
    
    def __init__(self, base_path="factor_lib"):
        self.base_path = Path(base_path)
        self.cache_dirs = {
            'combo_masks': self.base_path / 'cache' / 'combo_masks',
            'market_data': self.base_path / 'cache' / 'market_data', 
            'returns': self.base_path / 'cache' / 'returns',
            'industry': self.base_path / 'cache' / 'industry',
            'processed': self.base_path / 'processed',
            'raw': self.base_path / 'raw'
        }
        
        # 创建目录结构
        for cache_dir in self.cache_dirs.values():
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_cache_key(self, data_type, stock_code, start_date, end_date, **kwargs):
        """生成缓存键"""
        key_str = f"{data_type}_{stock_code}_{start_date}_{end_date}"
        if kwargs:
            # 添加额外参数到键中
            extra_params = "_".join([f"{k}_{v}" for k, v in sorted(kwargs.items())])
            key_str += f"_{extra_params}"
        return key_str
    
    def _get_cache_path(self, cache_type, cache_key):
        """获取缓存文件路径"""
        return self.cache_dirs[cache_type] / f"{cache_key}.pkl"
    
    def save_cache(self, data, cache_type, data_type, stock_code, start_date, end_date, **kwargs):
        """保存缓存数据"""
        cache_key = self._generate_cache_key(data_type, stock_code, start_date, end_date, **kwargs)
        cache_path = self._get_cache_path(cache_type, cache_key)
        
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'data': data,
                'timestamp': datetime.now(),
                'metadata': {
                    'data_type': data_type,
                    'stock_code': stock_code,
                    'start_date': start_date,
                    'end_date': end_date,
                    **kwargs
                }
            }, f)
        
        print(f"缓存已保存: {cache_path}")
        return cache_path
    
    def load_cache(self, cache_type, data_type, stock_code, start_date, end_date, max_age_days=30, **kwargs):
        """加载缓存数据"""
        cache_key = self._generate_cache_key(data_type, stock_code, start_date, end_date, **kwargs)
        cache_path = self._get_cache_path(cache_type, cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # 检查缓存是否过期
            if 'timestamp' in cached_data:
                cache_age = datetime.now() - cached_data['timestamp']
                if cache_age.days > max_age_days:
                    print(f"缓存已过期 ({cache_age.days}天): {cache_path}")
                    return None
            
            print(f"缓存命中: {cache_path}")
            return cached_data['data']
            
        except Exception as e:
            print(f"加载缓存失败: {cache_path}, 错误: {e}")
            return None
    
    def clean_expired_cache(self, max_age_days=30):
        """清理过期缓存"""
        cleaned_count = 0
        total_size_cleaned = 0
        
        for cache_type, cache_dir in self.cache_dirs.items():
            if not cache_dir.exists():
                continue
                
            for cache_file in cache_dir.glob("*.pkl"):
                try:
                    # 检查文件修改时间
                    file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                    
                    if file_age.days > max_age_days:
                        file_size = cache_file.stat().st_size
                        cache_file.unlink()
                        cleaned_count += 1
                        total_size_cleaned += file_size
                        print(f"删除过期缓存: {cache_file}")
                        
                except Exception as e:
                    print(f"清理缓存失败: {cache_file}, 错误: {e}")
        
        print(f"清理完成: 删除 {cleaned_count} 个文件, 释放 {total_size_cleaned/1024/1024:.2f} MB")
        return cleaned_count, total_size_cleaned
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        stats = {}
        
        for cache_type, cache_dir in self.cache_dirs.items():
            if not cache_dir.exists():
                stats[cache_type] = {'count': 0, 'size_mb': 0}
                continue
                
            files = list(cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in files)
            
            stats[cache_type] = {
                'count': len(files),
                'size_mb': total_size / 1024 / 1024
            }
        
        return stats
    
    def migrate_old_cache(self, old_cache_dir="factor_lib"):
        """迁移旧的缓存文件到新的目录结构"""
        old_path = Path(old_cache_dir)
        if not old_path.exists():
            return
        
        migrated_count = 0
        
        # 迁移规则映射
        migration_rules = {
            'combo_mask_': 'combo_masks',
            'df_industry_market_': 'industry', 
            'market_cap_': 'market_data',
            'open_': 'market_data',
            'return_1d_': 'returns'
        }
        
        for old_file in old_path.glob("*.pkl"):
            if old_file.name.startswith(tuple(migration_rules.keys())):
                # 确定目标目录
                target_cache_type = None
                for prefix, cache_type in migration_rules.items():
                    if old_file.name.startswith(prefix):
                        target_cache_type = cache_type
                        break
                
                if target_cache_type:
                    target_path = self.cache_dirs[target_cache_type] / old_file.name
                    old_file.rename(target_path)
                    migrated_count += 1
                    print(f"迁移文件: {old_file} -> {target_path}")
        
        print(f"迁移完成: {migrated_count} 个文件")
        return migrated_count


def clean_factor_cache(max_age_days=30):
    """清理因子缓存的便捷函数"""
    manager = FactorCacheManager()
    return manager.clean_expired_cache(max_age_days)


def show_cache_stats():
    """显示缓存统计信息的便捷函数"""
    manager = FactorCacheManager()
    stats = manager.get_cache_stats()
    
    print("=== 因子缓存统计 ===")
    total_size = 0
    total_count = 0
    
    for cache_type, stat in stats.items():
        print(f"{cache_type:15}: {stat['count']:4d} 文件, {stat['size_mb']:8.2f} MB")
        total_size += stat['size_mb']
        total_count += stat['count']
    
    print("-" * 40)
    print(f"{'总计':15}: {total_count:4d} 文件, {total_size:8.2f} MB")


if __name__ == "__main__":
    # 示例用法
    show_cache_stats()
    
    # 清理30天以上的缓存
    # clean_factor_cache(30)
    
    # 迁移旧缓存文件
    # manager = FactorCacheManager()
    # manager.migrate_old_cache()
