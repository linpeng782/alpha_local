# 简化版因子配置系统使用说明

## 重构完成！✅

现在因子配置和测试代码已经完全分离，使用更加简洁。

## 文件结构

- `factor_research_config.py`: 因子配置文件
- `test_single_factor.py`: 因子测试脚本
- `factor_research_log.py`: 因子研究记录（可选使用）

## 使用方法

### 1. 测试现有因子

在 `test_single_factor.py` 中修改这一行：
```python
current_factor = "high_low_std_504"  # 可选: high_low_std_504, market_cap
```

### 2. 添加新因子

在 `factor_research_config.py` 中：

1. 在 `FACTOR_DEFINITIONS` 中添加因子定义：
```python
FACTOR_DEFINITIONS = {
    "high_low_std_504": STD(Factor("high") / Factor("low"), 504),
    "market_cap": Factor("market_cap"),
    "your_new_factor": YOUR_FACTOR_EXPRESSION,  # 添加这里
}
```

2. 在 `FACTOR_CONFIGS` 中添加因子配置：
```python
FACTOR_CONFIGS = {
    # ... 现有配置 ...
    "your_new_factor": {
        "direction": 1,  # 或 -1
        "neutralize": True,  # 或 False
        "category": "技术指标",
        "description": "你的因子描述",
        "status": "研究中",
        "create_date": "2025-08-26",
        "last_test_date": "2025-08-26",
        "performance": {
            "ic": None,
            "rank_ic": None,
            "monotonicity": None,
            "notes": "待测试"
        }
    }
}
```

## 优势

1. **配置分离**: 因子定义和测试逻辑完全分离
2. **简单切换**: 只需修改一行代码就能切换因子
3. **无需eval**: 直接使用因子对象，避免字符串执行的复杂性
4. **易于维护**: 参考了 `factor_config.py` 的成熟模式
5. **清晰结构**: 因子定义和配置信息分开管理

## 工作流程

1. 在 `FACTOR_DEFINITIONS` 中定义新因子
2. 在 `FACTOR_CONFIGS` 中配置因子参数
3. 修改 `test_single_factor.py` 中的 `current_factor`
4. 运行测试
5. 查看结果并记录

现在你可以轻松管理和测试所有因子了！🎉
