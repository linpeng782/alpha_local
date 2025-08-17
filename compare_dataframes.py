"""
DataFrame比较工具
用于详细比较两个看似相同的DataFrame的差异
"""

import pandas as pd
import numpy as np

def detailed_dataframe_comparison(df1, df2, name1="df1", name2="df2"):
    """
    详细比较两个DataFrame的差异
    
    参数:
        df1, df2: 要比较的DataFrame
        name1, name2: DataFrame的名称，用于输出
    """
    
    print(f"=== 详细比较 {name1} 和 {name2} ===\n")
    
    # 1. 基本信息比较
    print("1. 基本信息比较:")
    print(f"   {name1} 形状: {df1.shape}")
    print(f"   {name2} 形状: {df2.shape}")
    print(f"   形状相同: {df1.shape == df2.shape}")
    
    # 2. 数据类型比较
    print(f"\n2. 数据类型比较:")
    print(f"   {name1} dtypes: {df1.dtypes.unique()}")
    print(f"   {name2} dtypes: {df2.dtypes.unique()}")
    
    # 检查每列的数据类型
    if df1.shape == df2.shape:
        dtype_diff = (df1.dtypes != df2.dtypes).sum()
        print(f"   数据类型不同的列数: {dtype_diff}")
        if dtype_diff > 0:
            print("   数据类型不同的列:")
            for col in df1.columns:
                if df1[col].dtype != df2[col].dtype:
                    print(f"     {col}: {df1[col].dtype} vs {df2[col].dtype}")
    
    # 3. 索引比较
    print(f"\n3. 索引比较:")
    print(f"   {name1} 索引类型: {type(df1.index)}")
    print(f"   {name2} 索引类型: {type(df2.index)}")
    print(f"   索引相同: {df1.index.equals(df2.index)}")
    
    if not df1.index.equals(df2.index):
        print("   索引差异详情:")
        # 检查索引长度
        print(f"     {name1} 索引长度: {len(df1.index)}")
        print(f"     {name2} 索引长度: {len(df2.index)}")
        
        # 检查索引名称
        print(f"     {name1} 索引名称: {df1.index.names}")
        print(f"     {name2} 索引名称: {df2.index.names}")
        
        # 找出不同的索引值
        if len(df1.index) == len(df2.index):
            diff_mask = df1.index != df2.index
            if hasattr(diff_mask, 'any') and diff_mask.any():
                print(f"     不同索引位置数量: {diff_mask.sum()}")
                print(f"     前5个不同的索引:")
                diff_indices = np.where(diff_mask)[0][:5]
                for i in diff_indices:
                    print(f"       位置 {i}: {df1.index[i]} vs {df2.index[i]}")
    
    # 4. 列名比较
    print(f"\n4. 列名比较:")
    print(f"   {name1} 列数: {len(df1.columns)}")
    print(f"   {name2} 列数: {len(df2.columns)}")
    print(f"   列名相同: {df1.columns.equals(df2.columns)}")
    
    if not df1.columns.equals(df2.columns):
        print("   列名差异:")
        cols1_only = set(df1.columns) - set(df2.columns)
        cols2_only = set(df2.columns) - set(df1.columns)
        if cols1_only:
            print(f"     只在{name1}中: {cols1_only}")
        if cols2_only:
            print(f"     只在{name2}中: {cols2_only}")
    
    # 5. 数值比较（如果形状相同）
    if df1.shape == df2.shape and df1.columns.equals(df2.columns) and df1.index.equals(df2.index):
        print(f"\n5. 数值比较:")
        
        # 检查NaN值
        nan_count1 = df1.isna().sum().sum()
        nan_count2 = df2.isna().sum().sum()
        print(f"   {name1} NaN数量: {nan_count1}")
        print(f"   {name2} NaN数量: {nan_count2}")
        
        # 检查无穷值
        inf_count1 = np.isinf(df1.select_dtypes(include=[np.number])).sum().sum()
        inf_count2 = np.isinf(df2.select_dtypes(include=[np.number])).sum().sum()
        print(f"   {name1} 无穷值数量: {inf_count1}")
        print(f"   {name2} 无穷值数量: {inf_count2}")
        
        # 数值差异分析
        try:
            # 对于数值列进行比较
            numeric_cols = df1.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # 检查该列是否完全相等
                if not df1[col].equals(df2[col]):
                    print(f"\n   列 '{col}' 存在差异:")
                    
                    # 忽略NaN的比较
                    mask = ~(df1[col].isna() | df2[col].isna())
                    if mask.any():
                        diff = np.abs(df1.loc[mask, col] - df2.loc[mask, col])
                        max_diff = diff.max()
                        mean_diff = diff.mean()
                        
                        print(f"     最大差异: {max_diff}")
                        print(f"     平均差异: {mean_diff}")
                        print(f"     差异大于1e-10的数量: {(diff > 1e-10).sum()}")
                        
                        # 显示前几个有差异的值
                        if (diff > 1e-15).any():
                            diff_indices = diff[diff > 1e-15].index[:3]
                            print(f"     前3个有差异的值:")
                            for idx in diff_indices:
                                print(f"       {idx}: {df1.loc[idx, col]} vs {df2.loc[idx, col]} (差异: {diff.loc[idx]})")
                    
                    # 检查NaN位置是否相同
                    nan_diff = df1[col].isna() != df2[col].isna()
                    if nan_diff.any():
                        print(f"     NaN位置不同的数量: {nan_diff.sum()}")
                        
        except Exception as e:
            print(f"   数值比较时出错: {e}")
    
    # 6. 内存使用比较
    print(f"\n6. 内存使用:")
    try:
        mem1 = df1.memory_usage(deep=True).sum() / 1024**2
        mem2 = df2.memory_usage(deep=True).sum() / 1024**2
        print(f"   {name1} 内存使用: {mem1:.2f} MB")
        print(f"   {name2} 内存使用: {mem2:.2f} MB")
    except:
        print("   无法计算内存使用")
    
    print(f"\n=== 比较完成 ===")

# 使用示例
if __name__ == "__main__":
    print("DataFrame比较工具已准备就绪")
    print("使用方法:")
    print("from compare_dataframes import detailed_dataframe_comparison")
    print("detailed_dataframe_comparison(df1, df2, 'return_group', 'group_return')")
