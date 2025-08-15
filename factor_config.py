"""
因子配置文件 
"""

from factor_processing_utils import Factor

# 因子定义
FACTOR_DICT = {
    "cfoa_mrq": Factor("cash_flow_from_operating_activities_mrq_0")
    / Factor("total_assets_mrq_0"),
    "atdy_mrq": Factor("operating_revenue_mrq_0") / Factor("total_assets_mrq_0")
    - Factor("operating_revenue_mrq_4") / Factor("total_assets_mrq_4"),
    "ccr_mrq": Factor("cash_flow_from_operating_activities_mrq_0")
    / Factor("current_liabilities_mrq_0"),
}

# 基础配置
DEFAULT_CONFIG = {
    "start_date": "2015-01-01",
    "end_date": "2025-07-01",
    "index_item": "000852.XSHG",
    "join_type": "inner",
    "force_rebuild": False,
    "cache_dir": "factor_lib",
}
