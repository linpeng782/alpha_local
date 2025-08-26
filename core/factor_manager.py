"""
因子管理工具
用于添加、更新、查看因子配置
"""

import json
from datetime import datetime
from factor_research_config import (
    FACTOR_CONFIGS, add_factor_config, update_factor_performance,
    print_factor_summary, print_factor_detail, get_factor_config
)

class FactorManager:
    def __init__(self):
        pass
    
    def add_new_factor(self, factor_name, definition_code, direction, neutralize, 
                      category, description, status="研究中"):
        """添加新因子配置"""
        config = {
            "definition_code": definition_code,  # 保存代码字符串，便于查看
            "direction": direction,
            "neutralize": neutralize,
            "category": category,
            "description": description,
            "status": status,
            "create_date": datetime.now().strftime("%Y-%m-%d"),
            "last_test_date": datetime.now().strftime("%Y-%m-%d"),
            "performance": {
                "ic": None,
                "rank_ic": None,
                "monotonicity": None,
                "notes": "待测试"
            }
        }
        
        # 这里需要手动添加到配置文件中
        print(f"请将以下配置添加到 factor_research_config.py 中:")
        print(f'"{factor_name}": {{')
        print(f'    "definition": lambda: {definition_code},')
        print(f'    "direction": {direction},')
        print(f'    "neutralize": {neutralize},')
        print(f'    "category": "{category}",')
        print(f'    "description": "{description}",')
        print(f'    "status": "{status}",')
        print(f'    "create_date": "{config["create_date"]}",')
        print(f'    "last_test_date": "{config["last_test_date"]}",')
        print(f'    "performance": {{')
        print(f'        "ic": None,')
        print(f'        "rank_ic": None,')
        print(f'        "monotonicity": None,')
        print(f'        "notes": "待测试"')
        print(f'    }}')
        print(f'}},')
        
    def update_performance(self, factor_name, ic=None, rank_ic=None, 
                          monotonicity=None, notes=None):
        """更新因子表现"""
        performance_data = {}
        if ic is not None:
            performance_data["ic"] = ic
        if rank_ic is not None:
            performance_data["rank_ic"] = rank_ic
        if monotonicity is not None:
            performance_data["monotonicity"] = monotonicity
        if notes is not None:
            performance_data["notes"] = notes
            
        update_factor_performance(factor_name, performance_data)
        print(f"已更新因子 {factor_name} 的表现数据")
        
    def change_status(self, factor_name, new_status):
        """修改因子状态"""
        if factor_name in FACTOR_CONFIGS:
            FACTOR_CONFIGS[factor_name]["status"] = new_status
            print(f"已将因子 {factor_name} 状态修改为: {new_status}")
        else:
            print(f"未找到因子: {factor_name}")
    
    def export_to_json(self, filename="factor_configs_backup.json"):
        """导出配置到JSON文件"""
        # 由于lambda函数无法序列化，这里只导出基本信息
        export_data = {}
        for name, config in FACTOR_CONFIGS.items():
            export_data[name] = {
                "direction": config["direction"],
                "neutralize": config["neutralize"],
                "category": config["category"],
                "description": config["description"],
                "status": config["status"],
                "create_date": config["create_date"],
                "last_test_date": config["last_test_date"],
                "performance": config["performance"]
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"配置已导出到: {filename}")
    
    def interactive_menu(self):
        """交互式菜单"""
        while True:
            print("\n=== 因子管理工具 ===")
            print("1. 查看所有因子总结")
            print("2. 查看因子详情")
            print("3. 添加新因子")
            print("4. 更新因子表现")
            print("5. 修改因子状态")
            print("6. 导出配置")
            print("0. 退出")
            
            choice = input("\n请选择操作 (0-6): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                print_factor_summary()
            elif choice == "2":
                factor_name = input("请输入因子名称: ").strip()
                print_factor_detail(factor_name)
            elif choice == "3":
                self._add_factor_interactive()
            elif choice == "4":
                self._update_performance_interactive()
            elif choice == "5":
                self._change_status_interactive()
            elif choice == "6":
                filename = input("请输入导出文件名 (默认: factor_configs_backup.json): ").strip()
                if not filename:
                    filename = "factor_configs_backup.json"
                self.export_to_json(filename)
            else:
                print("无效选择，请重试")
    
    def _add_factor_interactive(self):
        """交互式添加因子"""
        print("\n=== 添加新因子 ===")
        factor_name = input("因子名称: ").strip()
        definition_code = input("因子定义代码 (如: STD(Factor('high')/Factor('low'), 20)): ").strip()
        direction = int(input("因子方向 (1 或 -1): ").strip())
        neutralize = input("是否中性化 (y/n): ").strip().lower() == 'y'
        category = input("因子分类: ").strip()
        description = input("因子描述: ").strip()
        status = input("因子状态 (默认: 研究中): ").strip() or "研究中"
        
        self.add_new_factor(factor_name, definition_code, direction, neutralize, 
                           category, description, status)
    
    def _update_performance_interactive(self):
        """交互式更新表现"""
        print("\n=== 更新因子表现 ===")
        factor_name = input("因子名称: ").strip()
        
        if factor_name not in FACTOR_CONFIGS:
            print(f"未找到因子: {factor_name}")
            return
        
        ic_str = input("IC值 (回车跳过): ").strip()
        rank_ic_str = input("Rank IC值 (回车跳过): ").strip()
        mono_str = input("单调性 (回车跳过): ").strip()
        notes = input("备注 (回车跳过): ").strip()
        
        ic = float(ic_str) if ic_str else None
        rank_ic = float(rank_ic_str) if rank_ic_str else None
        monotonicity = float(mono_str) if mono_str else None
        notes = notes if notes else None
        
        self.update_performance(factor_name, ic, rank_ic, monotonicity, notes)
    
    def _change_status_interactive(self):
        """交互式修改状态"""
        print("\n=== 修改因子状态 ===")
        factor_name = input("因子名称: ").strip()
        print("可选状态: 研究中, 已完成, 已废弃")
        new_status = input("新状态: ").strip()
        
        self.change_status(factor_name, new_status)

# 快捷函数
def quick_add_factor():
    """快速添加因子的便捷函数"""
    manager = FactorManager()
    manager._add_factor_interactive()

def quick_update_performance(factor_name, ic=None, rank_ic=None, monotonicity=None, notes=None):
    """快速更新表现的便捷函数"""
    manager = FactorManager()
    manager.update_performance(factor_name, ic, rank_ic, monotonicity, notes)

if __name__ == "__main__":
    manager = FactorManager()
    manager.interactive_menu()
