# 因子配置管理系统使用说明

## 快速开始

### 1. 测试因子
在 `test_single_factor.py` 中修改这一行来切换要测试的因子：
```python
current_factor = "high_low_std_504"  # 改为你要测试的因子名称
```

### 2. 添加新因子
在 `test_single_factor.py` 的 `factor_configs` 字典中添加：
```python
"your_new_factor": {
    "definition": YOUR_FACTOR_EXPRESSION,
    "direction": 1,  # 或 -1
    "neutralize": True,  # 或 False
    "description": "因子描述",
    "category": "因子分类"
}
```

### 3. 记录测试结果
测试完成后，使用以下代码记录结果：
```python
from factor_research_log import add_test_result, update_factor_status

# 添加测试结果
add_test_result(
    "your_factor_name", 
    "2025-08-26",           # 测试日期
    "2015-2025",            # 测试期间
    "000985.XSHG",          # 指数
    ic=0.045,               # IC值
    rank_ic=0.052,          # Rank IC值
    monotonicity=0.85,      # 单调性
    notes="测试备注"
)

# 更新因子状态
update_factor_status("your_factor_name", "已完成")  # 或 "已废弃"
```

### 4. 查看研究记录
```python
from factor_research_log import print_research_summary, print_factor_detail

# 查看所有因子总结
print_research_summary()

# 查看特定因子详情
print_factor_detail("market_cap")
```

## 文件说明

- `test_single_factor.py`: 主测试文件，包含因子配置
- `factor_research_log.py`: 研究记录管理，记录所有测试结果
- `README_factor_config.md`: 本使用说明

## 工作流程

1. **添加新因子** → 在 `factor_configs` 中定义
2. **运行测试** → 修改 `current_factor` 并运行
3. **查看结果** → 检查生成的图表和报告
4. **记录结果** → 使用 `add_test_result` 记录指标
5. **更新状态** → 使用 `update_factor_status` 更新状态

## 因子状态说明

- **研究中**: 正在测试的因子
- **已完成**: 测试完成，效果良好的因子  
- **已废弃**: 测试完成，效果不佳的因子

这样你就能系统地管理所有因子研究，避免遗忘和重复工作！
