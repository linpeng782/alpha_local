# Alpha_Local 因子研究工作区

## 目录结构

```
alpha_local/
├── 📁 core/                    # 核心工具和配置
│   ├── factor_processing_utils.py  # 因子处理工具函数
│   ├── factor_config.py            # 因子配置
│   ├── cache_manager.py             # 缓存管理
│   └── compare_dataframes.py        # 数据框对比工具
├── 📁 notebooks/               # Jupyter notebooks
│   ├── research/              # 研究性notebook
│   │   ├── all_factor_test.ipynb   # 全因子测试
│   │   ├── test_factor.ipynb       # 因子测试
│   │   └── small_cap.ipynb         # 小盘股研究
│   └── factors/               # 具体因子分析
│       ├── cfoa.ipynb             # CFOA因子分析
│       └── quality_composite.ipynb # 质量复合因子
├── 📁 scripts/                # Python脚本
│   ├── debug_factor.py            # 因子调试脚本
│   ├── produce_factor.py          # 因子生产脚本
│   ├── factor_database_builder.py # 因子数据库构建
│   └── feval_factor_database.py   # 因子数据库评估
├── 📁 data/                   # 数据文件
│   ├── database/              # 因子数据库
│   ├── factor_lib/            # 因子库缓存
│   └── cache/                 # 其他缓存文件
├── 📁 outputs/                # 输出文件
│   ├── images/                # 图片输出
│   └── reports/               # 报告输出
└── 📁 docs/                   # 文档
    └── 因子数据库构建流程.md    # 构建流程文档
```

## 使用说明

### 导入核心工具

在notebooks中使用：
```python
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'core'))
from factor_processing_utils import *
```

在scripts中使用：
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from factor_processing_utils import *
```

### 项目工作流

1. **研究阶段**：在 `notebooks/research/` 中进行探索性分析
2. **因子开发**：在 `notebooks/factors/` 中开发具体因子
3. **脚本化**：将成熟的逻辑移到 `scripts/` 中
4. **数据管理**：所有数据文件存储在 `data/` 中
5. **结果输出**：图片和报告保存在 `outputs/` 中

## 注意事项

- 所有缓存文件统一存储在 `data/` 目录下
- 图片输出统一保存到 `outputs/images/`
- 重要的分析结果可以导出报告到 `outputs/reports/`
- 核心工具函数统一维护在 `core/` 目录下
