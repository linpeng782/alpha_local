# Alpha_Local 数据目录说明

## 目录结构

```
data/
├── 📁 cache/                   # 缓存数据（按类型分类）
│   ├── combo_masks/           # 组合掩码文件 (7个文件)
│   ├── industry/              # 行业数据文件 (4个文件)
│   ├── market_data/           # 市场数据文件 (8个文件)
│   └── returns/               # 收益率数据文件 (5个文件)
└── 📁 factor_lib/             # 因子数据库
    ├── raw/                   # 原始因子数据
    ├── processed/             # 处理后因子数据
    └── README.md              # 因子库说明文档
```

## 数据类型说明

### 1. cache/ - 缓存数据
**用途**：存储计算过程中的中间结果和常用数据，提高计算效率

- **combo_masks/**：股票过滤掩码（新股、ST、停牌、股票池过滤的组合）
- **industry/**：行业分类和市值数据
- **market_data/**：开盘价、市值等基础市场数据  
- **returns/**：日收益率数据

### 2. factor_lib/ - 因子数据库
**用途**：存储因子计算的原始数据和处理结果

- **raw/**：原始因子数据（未经清洗）
- **processed/**：处理后因子数据（已清洗、标准化、中性化）

## 使用建议

### 路径引用
在scripts目录下的Python文件中，使用相对路径：

```python
import os

# 获取当前脚本目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 因子数据路径
raw_path = os.path.join(current_dir, "..", "data", "factor_lib", "raw")
processed_path = os.path.join(current_dir, "..", "data", "factor_lib", "processed")

# 缓存数据路径
cache_path = os.path.join(current_dir, "..", "data", "cache")
```

### 数据管理
1. **定期清理cache目录**：删除过期的缓存文件
2. **备份重要因子**：processed目录下的重要因子数据建议备份
3. **监控磁盘空间**：cache目录可能会占用较大空间

## 磁盘使用统计

- **cache目录**：约6.3GB（24个文件）
- **factor_lib目录**：根据存储的因子数量而定
- **总计**：约6.3GB+（不含factor_lib中的因子数据）
