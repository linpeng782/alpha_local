# 因子配置管理系统使用说明

## 概述

这个系统帮助你统一管理和跟踪所有研究过的因子，避免重复工作和遗忘重要因子。

## 文件结构

- `factor_research_config.py`: 因子配置文件，存储所有因子定义和元数据
- `factor_manager.py`: 因子管理工具，提供添加、更新、查看功能
- `test_single_factor.py`: 修改后的测试脚本，使用配置文件

## 使用方法

### 1. 测试现有因子

在 `test_single_factor.py` 中，只需修改一行代码：

```python
current_factor = "high_low_std_504"  # 改为你要测试的因子名称
```

### 2. 添加新因子

#### 方法一：使用交互式工具

```bash
cd /Users/didi/KDCJ/alpha_local/core
python factor_manager.py
```

选择选项3，按提示输入因子信息。

#### 方法二：直接编辑配置文件

在 `factor_research_config.py` 的 `FACTOR_CONFIGS` 字典中添加：

```python
"your_factor_name": {
    "definition": lambda: YOUR_FACTOR_DEFINITION,
    "direction": 1,  # 或 -1
    "neutralize": True,  # 或 False
    "category": "技术指标",
    "description": "因子描述",
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
```

### 3. 更新因子表现

测试完成后，更新因子的表现数据：

```python
from factor_manager import quick_update_performance

quick_update_performance(
    "your_factor_name",
    ic=0.045,
    rank_ic=0.052,
    monotonicity=0.85,
    notes="表现良好"
)
```

### 4. 查看因子总结

```python
from factor_research_config import print_factor_summary, print_factor_detail

# 查看所有因子总结
print_factor_summary()

# 查看特定因子详情
print_factor_detail("market_cap")
```

### 5. 管理因子状态

```python
from factor_manager import FactorManager

manager = FactorManager()
manager.change_status("your_factor_name", "已完成")  # 或 "已废弃"
```

## 因子分类

- **技术指标**: 基于价格、成交量的技术分析因子
- **基本面**: 基于财务数据的基本面因子
- **量价关系**: 成交量与价格关系的因子
- **风险收益**: 风险调整收益相关因子
- **其他**: 其他类型因子

## 因子状态

- **研究中**: 正在研究测试的因子
- **已完成**: 测试完成，效果良好的因子
- **已废弃**: 测试完成，效果不佳的因子

## 最佳实践

1. **及时记录**: 每次测试新因子后，立即更新表现数据
2. **详细描述**: 为每个因子写清楚的描述和经济含义
3. **分类管理**: 按照因子类型进行分类，便于查找
4. **状态更新**: 及时更新因子状态，避免重复研究
5. **定期备份**: 使用 `factor_manager.py` 导出配置备份

## 示例工作流

1. 有新的因子想法
2. 在配置文件中添加因子定义
3. 修改 `test_single_factor.py` 中的 `current_factor`
4. 运行测试
5. 更新因子表现数据
6. 根据结果修改因子状态

这样你就能系统地管理所有因子研究，避免遗忘和重复工作！
