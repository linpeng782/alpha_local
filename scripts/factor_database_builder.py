"""
因子数据库构建器 - 简化版本
结合增量更新和批量处理的最佳实践
"""

import os
import sys
import pandas as pd
import numpy as np

# 添加core目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from factor_processing_utils import *



class FactorDatabaseBuilder:
    def __init__(self, cache_dir="factor_lib"):
        """
        初始化因子数据库构建器

        参数:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        self.setup_directories()

    def setup_directories(self):
        """创建必要的目录结构"""
        directories = [
            f"{self.cache_dir}/raw", 
            f"{self.cache_dir}/processed",
            f"{self.cache_dir}/final"
        ]
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
        # 确保主目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

    def build_single_factor(
        self,
        factor_name,
        factor_expr,
        stock_list,
        start_date,
        end_date,
        stock_universe,
        index_item,
        force_rebuild=False,
    ):
        """
        构建单个因子（带缓存机制）

        参数:
            factor_name: 因子名称
            factor_expr: 因子表达式
            stock_list: 股票列表
            start_date: 开始日期
            end_date: 结束日期
            stock_universe: 股票池
            index_item: 指数代码
            force_rebuild: 是否强制重建

        返回:
            factor_stacked: 处理后的因子数据（长格式）
        """

        # 定义缓存路径
        raw_path = f"{self.cache_dir}/raw/{factor_name}_{index_item}_{start_date}_{end_date}.pkl"
        processed_path = f"{self.cache_dir}/processed/{factor_name}_{index_item}_{start_date}_{end_date}.pkl"

        # 检查是否已存在处理后的因子
        if not force_rebuild and os.path.exists(processed_path):
            print(f"加载缓存因子: {factor_name}")
            return pd.read_pickle(processed_path)

        # 检查原始因子缓存
        if not force_rebuild and os.path.exists(raw_path):
            print(f"加载原始因子: {factor_name}")
            raw_factor = pd.read_pickle(raw_path)
        else:
            print(f"计算原始因子: {factor_name}")
            raw_factor = execute_factor(factor_expr, stock_list, start_date, end_date)
            raw_factor.to_pickle(raw_path)

        # 因子预处理
        print(f"预处理因子: {factor_name}")
        processed_factor = preprocess_factor(raw_factor, stock_universe, index_item)

        # 转换为长格式
        factor_stacked = (
            processed_factor.stack("order_book_id")
            .swaplevel()
            .sort_index()
            .to_frame(factor_name)
        )

        # 保存处理后的因子
        factor_stacked.to_pickle(processed_path)
        print(f"完成因子: {factor_name}")

        return factor_stacked

    def build_factor_database(
        self,
        factor_dict,
        stock_list,
        start_date,
        end_date,
        stock_universe,
        index_item,
        daily_data,
        force_rebuild=False,
        join_type="inner",
    ):
        """
        构建完整的因子数据库

        参数:
            factor_dict: 因子字典 {因子名: 因子表达式}
            stock_list: 股票列表
            start_date: 开始日期
            end_date: 结束日期
            stock_universe: 股票池
            index_item: 指数代码
            daily_data: 市场数据(包含价格、成交量、涨跌停等)
            force_rebuild: 是否强制重建
            join_type: 拼接方式 ('inner', 'left', 'outer')

        返回:
            final_df: 最终的因子数据库
            quality_report: 质量报告
        """

        print(f"开始构建因子数据库，共 {len(factor_dict)} 个因子")
        print(f"时间范围: {start_date} 到 {end_date}")
        print(f"股票数量: {len(stock_list)}")

        # 检查最终结果缓存
        final_path = (
            f"{self.cache_dir}/final/factor_db_{index_item}_{start_date}_{end_date}.pkl"
        )
        if not force_rebuild and os.path.exists(final_path):
            print(f"加载完整缓存数据库")
            return pd.read_pickle(final_path), None

        # 逐个构建因子
        factor_list = []
        for i, (factor_name, factor_expr) in enumerate(factor_dict.items(), 1):
            print(f"\n[{i}/{len(factor_dict)}] 处理因子: {factor_name}")

            try:
                factor_stacked = self.build_single_factor(
                    factor_name,
                    factor_expr,
                    stock_list,
                    start_date,
                    end_date,
                    stock_universe,
                    index_item,
                    force_rebuild,
                )
                factor_list.append(factor_stacked)

            except Exception as e:
                print(f"因子 {factor_name} 构建失败: {str(e)}")
                continue

        if not factor_list:
            print("没有成功构建的因子")
            return pd.DataFrame(), pd.DataFrame()

        # 批量拼接所有因子
        print(f"\n拼接 {len(factor_list)} 个因子...")
        all_factors = pd.concat(factor_list, axis=1)

        # 与价格数据合并
        #print(f"合并价格数据，使用 {join_type} 连接...")
        final_df = pd.concat([daily_data, all_factors], axis=1, join=join_type)

        # 生成质量报告
        #print("生成质量报告...")
        quality_report = self.generate_quality_report(final_df)

        # 保存最终结果
        print("保存最终数据库...")
        final_df.to_pickle(final_path)

        file_size_mb = os.path.getsize(final_path) / 1024**2
        print(f"构建完成！")
        print(f"   最终数据: {len(final_df)} 行 × {len(final_df.columns)} 列")
        print(f"   文件大小: {file_size_mb:.1f} MB")
        print(f"   保存路径: {final_path}")

        return final_df, quality_report

    def generate_quality_report(self, df):
        """
        生成简化的质量报告

        参数:
            df: 因子数据库DataFrame

        返回:
            quality_report: 质量报告DataFrame
        """

        reports = []

        for col in df.columns:
            if col not in ["open", "close", "high", "low", "volume"]:  # 跳过价格列

                total_count = len(df[col])
                valid_count = df[col].notna().sum()
                missing_rate = (total_count - valid_count) / total_count

                if valid_count > 0:
                    unique_count = df[col].nunique()
                    mean_val = df[col].mean()
                    std_val = df[col].std()

                    # 简单的质量评分
                    quality_score = 100
                    if missing_rate > 0.5:
                        quality_score -= 40
                    elif missing_rate > 0.2:
                        quality_score -= 20

                    if unique_count < total_count * 0.1:  # 唯一值太少
                        quality_score -= 20

                else:
                    unique_count = mean_val = std_val = 0
                    quality_score = 0

                reports.append(
                    {
                        "factor_name": col,
                        "total_count": total_count,
                        "valid_count": valid_count,
                        "missing_rate": round(missing_rate, 4),
                        "unique_count": unique_count,
                        "mean": round(mean_val, 4) if pd.notna(mean_val) else None,
                        "std": round(std_val, 4) if pd.notna(std_val) else None,
                        "quality_score": max(0, quality_score),
                    }
                )

        quality_df = pd.DataFrame(reports)
        quality_df = quality_df.sort_values("quality_score", ascending=False)

        # 打印摘要
        if len(quality_df) > 0:
            avg_score = quality_df["quality_score"].mean()
            good_factors = (quality_df["quality_score"] >= 70).sum()
            print(
                f"质量摘要: 平均分 {avg_score:.1f}, 优良因子 {good_factors}/{len(quality_df)}"
            )

        return quality_df



