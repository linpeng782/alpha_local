"""
因子数据库执行文件
用于运行因子数据库构建任务
"""

from factor_database_builder import FactorDatabaseBuilder
from factor_config import FACTOR_DICT, DEFAULT_CONFIG
from factor_processing_utils import *


def factor_factory():
    """
    主函数 - 演示如何使用因子数据库构建器
    """

    # 从配置文件加载基础参数
    start_date = DEFAULT_CONFIG["start_date"]
    end_date = DEFAULT_CONFIG["end_date"]
    index_item = DEFAULT_CONFIG["index_item"]

    # 获取股票池和价格数据
    print("准备动态券池")
    stock_universe = INDEX_FIX(start_date, end_date, index_item)
    stock_list = stock_universe.columns.tolist()

    print("获取市场数据")
    market_data = get_price(
        stock_list,
        start_date,
        end_date,
        fields=[
            "open",
            "close",
            "high",
            "low",
            "limit_up",
            "limit_down",
            "total_turnover",
            "volume",
        ],
        adjust_type="none",
        skip_suspended=True,
    ).sort_index()

    # 从配置文件加载因子
    print("加载因子配置文件")
    factor_dict = FACTOR_DICT
    print(f"加载了 {len(factor_dict)} 个因子: {list(factor_dict.keys())}")

    # 创建构建器并执行
    builder = FactorDatabaseBuilder()

    final_df, quality_report = builder.build_factor_database(
        factor_dict=factor_dict,
        stock_list=stock_list,
        start_date=start_date,
        end_date=end_date,
        stock_universe=stock_universe,
        index_item=index_item,
        daily_data=market_data,
        force_rebuild=DEFAULT_CONFIG["force_rebuild"],  # 使用缓存
        join_type=DEFAULT_CONFIG["join_type"],  # 只保留有价格和因子的数据
    )

    # 显示结果
    print(f"\n最终结果:")
    print(f"数据形状: {final_df.shape}")
    print(f"列名: {list(final_df.columns)}")

    if quality_report is not None and len(quality_report) > 0:
        print(f"\n质量报告:")
        print(quality_report[["factor_name", "missing_rate", "quality_score"]].head())

    return final_df, quality_report


if __name__ == "__main__":
    # 执行主函数
    df, report = factor_factory()
    print(df)
