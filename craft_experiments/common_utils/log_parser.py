"""
log_parser.py

日志解析工具 - 用于从评估日志中提取成功率和关键指标

功能说明：
1. 从 LIBERO 评估日志中提取总体成功率
2. 从训练目录中查找最新的 checkpoint
3. 解析包含多个任务套件结果的日志文件
4. 将结果格式化为 Markdown 表格

使用场景：
- 自动化实验脚本中提取评估结果
- 生成论文表格
- 快速查看训练效果

作者：VLA-Adapter + CRaFT 团队
"""

import re
from pathlib import Path
from typing import Dict, Optional


def extract_success_rate_from_log(log_file_path: str) -> Optional[float]:
    """
    从 LIBERO 评估日志文件中提取总体成功率
    
    工作原理：
    1. 读取整个日志文件内容
    2. 使用正则表达式匹配成功率行（格式：Overall success rate: 0.8500 (85.0%)）
    3. 提取浮点数值（0.0 到 1.0 之间）
    
    Args:
        log_file_path: 评估日志文件的路径（通常在 eval_logs/ 目录下）
    
    Returns:
        成功率（浮点数，范围 0.0-1.0），如果未找到则返回 None
        
    示例：
        >>> extract_success_rate_from_log("eval_logs/spatial_eval.log")
        0.8500  # 表示 85% 的成功率
    """
    try:
        # 步骤 1: 读取日志文件内容
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 步骤 2: 使用正则表达式匹配成功率行
        # 匹配模式：Overall success rate: 0.8500 (85.0%)
        # 捕获组 1: 小数形式的成功率（0.8500）
        # 捕获组 2: 百分比形式的成功率（85.0）
        pattern = r"Overall success rate:\s+([\d.]+)\s+\(([\d.]+)%\)"
        match = re.search(pattern, content)
        
        if match:
            # 步骤 3: 提取并转换为浮点数
            success_rate = float(match.group(1))
            return success_rate
        else:
            # 未找到匹配的成功率行
            print(f"⚠️  警告：无法在 {log_file_path} 中找到成功率")
            return None
    
    except Exception as e:
        # 文件读取或解析错误
        print(f"❌ 错误：读取日志文件 {log_file_path} 时出错: {e}")
        return None


def extract_checkpoint_path(run_dir: str) -> Optional[str]:
    """
    从训练运行目录中提取最新 checkpoint 的路径
    
    工作原理：
    1. 检查运行目录是否存在
    2. 查找所有 checkpoint 目录（格式：run_dir--XXXXX_chkpt）
    3. 按步数排序，返回最新的 checkpoint
    
    Args:
        run_dir: 训练运行目录的路径（例如：runs/craft-libero_spatial-table1）
    
    Returns:
        最新 checkpoint 目录的路径，如果未找到则返回 None
        
    示例：
        >>> extract_checkpoint_path("runs/craft-spatial")
        "runs/craft-spatial--20000_chkpt"  # 表示第 20000 步的 checkpoint
    """
    # 步骤 1: 检查运行目录是否存在
    run_path = Path(run_dir)
    
    if not run_path.exists():
        print(f"⚠️  警告：运行目录 {run_dir} 不存在")
        return None
    
    # 步骤 2: 查找所有 checkpoint 目录
    # Checkpoint 目录格式：run_dir--XXXXX_chkpt（例如：craft-spatial--5000_chkpt）
    checkpoint_dirs = list(run_path.parent.glob(f"{run_path.name}--*_chkpt"))
    
    if not checkpoint_dirs:
        print(f"⚠️  警告：未找到 {run_dir} 的 checkpoint 目录")
        return None
    
    # 步骤 3: 按步数排序并获取最新的 checkpoint
    def get_step_number(path):
        """从 checkpoint 目录名中提取步数"""
        match = re.search(r'--(\d+)_chkpt', path.name)
        return int(match.group(1)) if match else 0
    
    latest_checkpoint = max(checkpoint_dirs, key=get_step_number)
    return str(latest_checkpoint)


def parse_all_results(results_log_path: str) -> Dict[str, float]:
    """
    解析包含多个任务套件结果的日志文件
    
    工作原理：
    1. 逐行读取结果日志文件
    2. 使用正则表达式匹配每一行的任务名称和成功率
    3. 构建字典存储所有结果
    
    Args:
        results_log_path: 结果日志文件的路径（例如：table1_results.log）
    
    Returns:
        字典，键为任务套件名称，值为成功率（浮点数）
        
    示例：
        >>> parse_all_results("table1_results.log")
        {
            'libero_spatial': 0.9780,
            'libero_object': 0.9920,
            'libero_goal': 0.9720,
            'libero_10': 0.9500
        }
    """
    results = {}
    
    try:
        # 步骤 1: 逐行读取日志文件
        with open(results_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 步骤 2: 匹配每一行的格式（例如：libero_spatial: 0.8500）
                match = re.match(r"(\w+):\s+([\d.]+)", line.strip())
                if match:
                    task_suite = match.group(1)      # 任务套件名称
                    success_rate = float(match.group(2))  # 成功率
                    results[task_suite] = success_rate
    
    except Exception as e:
        print(f"❌ 错误：解析结果日志 {results_log_path} 时出错: {e}")
    
    return results


def format_results_table(results: Dict[str, float]) -> str:
    """
    将结果格式化为 Markdown 表格
    
    工作原理：
    1. 创建表格头部（任务套件 | 成功率）
    2. 按字母顺序遍历所有结果
    3. 计算并添加平均成功率
    
    Args:
        results: 字典，键为任务套件名称，值为成功率
    
    Returns:
        格式化的 Markdown 表格字符串
        
    示例输出：
        | Task Suite | Success Rate |
        |------------|-------------|
        | libero_spatial | 0.9780 (97.8%) |
        | libero_object | 0.9920 (99.2%) |
        |------------|-------------|
        | **Average** | **0.9850 (98.5%)** |
    """
    # 步骤 1: 创建表格头部
    table = "| 任务套件 (Task Suite) | 成功率 (Success Rate) |\n"
    table += "|----------------------|----------------------|\n"
    
    # 步骤 2: 按字母顺序添加每个任务的结果
    for task_suite, success_rate in sorted(results.items()):
        table += f"| {task_suite} | {success_rate:.4f} ({success_rate*100:.1f}%) |\n"
    
    # 步骤 3: 计算并添加平均成功率
    if results:
        avg_success_rate = sum(results.values()) / len(results)
        table += "|----------------------|----------------------|\n"
        table += f"| **平均 (Average)** | **{avg_success_rate:.4f} ({avg_success_rate*100:.1f}%)** |\n"
    
    return table


if __name__ == "__main__":
    """
    命令行测试接口
    
    使用方法：
        python log_parser.py <日志文件路径>
        
    示例：
        python log_parser.py eval_logs/spatial_eval.log
    """
    import sys
    
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
        print(f"📄 正在解析日志文件: {log_path}")
        success_rate = extract_success_rate_from_log(log_path)
        if success_rate is not None:
            print(f"✅ 成功率: {success_rate:.4f} ({success_rate*100:.1f}%)")
        else:
            print("❌ 无法提取成功率")
    else:
        print("用法: python log_parser.py <日志文件路径>")
        print("示例: python log_parser.py eval_logs/spatial_eval.log")

