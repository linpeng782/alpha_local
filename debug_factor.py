from factor_processing_utils import *
import os


if __name__ == "__main__":
    start_date = "2015-01-01"
    end_date = "2025-07-01"
    index_item = "000852.XSHG"

    change_day = 20
    month_day = 20
    year_day = 252

    stock_universe = INDEX_FIX(start_date, end_date, index_item)
    stock_list = stock_universe.columns.tolist()
    date_list = stock_universe.index.tolist()
    factor_name = "cfoa_ttm_0"

    # 使用os.path.join处理路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(
        current_dir,
        "factor_lib",
        "raw",
        f"{factor_name}_{index_item}_{start_date}_{end_date}.pkl",
    )
    processed_path = os.path.join(
        current_dir,
        "factor_lib",
        "processed",
        f"{factor_name}_{index_item}_{start_date}_{end_date}.pkl",
    )

    try:
        factor_clean = pd.read_pickle(processed_path)

    except:
        factor_definition = Factor(
            "cash_flow_from_operating_activities_ttm_0"
        ) / Factor("total_assets_ttm_0")
        raw_factor = execute_factor(factor_definition, stock_list, start_date, end_date)
        factor_clean = data_clean(raw_factor, stock_universe, index_item)


    return_group_hold, _ = factor_layered_backtest(
        factor_clean, change_day, 10, index_item, name=factor_name
    )

    print("\n=== Rebalance=False (买入持有) ===")
    print(return_group_hold)
