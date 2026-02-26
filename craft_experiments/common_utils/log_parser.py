"""
log_parser.py

Utility functions for parsing evaluation logs and extracting success rates.
"""

import re
from pathlib import Path
from typing import Dict, Optional


def extract_success_rate_from_log(log_file_path: str) -> Optional[float]:
    """
    Extract the overall success rate from a LIBERO evaluation log file.
    
    Args:
        log_file_path: Path to the evaluation log file
    
    Returns:
        Success rate as a float (0.0 to 1.0), or None if not found
    """
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
        
        # Look for the final success rate line
        # Pattern: "Overall success rate: 0.8500 (85.0%)"
        pattern = r"Overall success rate:\s+([\d.]+)\s+\(([\d.]+)%\)"
        match = re.search(pattern, content)
        
        if match:
            success_rate = float(match.group(1))
            return success_rate
        else:
            print(f"Warning: Could not find success rate in {log_file_path}")
            return None
    
    except Exception as e:
        print(f"Error reading log file {log_file_path}: {e}")
        return None


def extract_checkpoint_path(run_dir: str) -> Optional[str]:
    """
    Extract the path to the latest checkpoint from a training run directory.
    
    Args:
        run_dir: Path to the training run directory
    
    Returns:
        Path to the latest checkpoint directory, or None if not found
    """
    run_path = Path(run_dir)
    
    if not run_path.exists():
        print(f"Warning: Run directory {run_dir} does not exist")
        return None
    
    # Look for checkpoint directories (format: run_dir--XXXXX_chkpt)
    checkpoint_dirs = list(run_path.parent.glob(f"{run_path.name}--*_chkpt"))
    
    if not checkpoint_dirs:
        print(f"Warning: No checkpoint directories found for {run_dir}")
        return None
    
    # Sort by step number and get the latest
    def get_step_number(path):
        match = re.search(r'--(\d+)_chkpt', path.name)
        return int(match.group(1)) if match else 0
    
    latest_checkpoint = max(checkpoint_dirs, key=get_step_number)
    return str(latest_checkpoint)


def parse_all_results(results_log_path: str) -> Dict[str, float]:
    """
    Parse a results log file containing multiple task suite results.
    
    Args:
        results_log_path: Path to the results log file
    
    Returns:
        Dictionary mapping task suite names to success rates
    """
    results = {}
    
    try:
        with open(results_log_path, 'r') as f:
            for line in f:
                # Pattern: "libero_spatial: 0.8500"
                match = re.match(r"(\w+):\s+([\d.]+)", line.strip())
                if match:
                    task_suite = match.group(1)
                    success_rate = float(match.group(2))
                    results[task_suite] = success_rate
    
    except Exception as e:
        print(f"Error parsing results log {results_log_path}: {e}")
    
    return results


def format_results_table(results: Dict[str, float]) -> str:
    """
    Format results as a markdown table.
    
    Args:
        results: Dictionary mapping task suite names to success rates
    
    Returns:
        Formatted markdown table string
    """
    table = "| Task Suite | Success Rate |\n"
    table += "|------------|-------------|\n"
    
    for task_suite, success_rate in sorted(results.items()):
        table += f"| {task_suite} | {success_rate:.4f} ({success_rate*100:.1f}%) |\n"
    
    # Add average
    if results:
        avg_success_rate = sum(results.values()) / len(results)
        table += "|------------|-------------|\n"
        table += f"| **Average** | **{avg_success_rate:.4f} ({avg_success_rate*100:.1f}%)** |\n"
    
    return table


if __name__ == "__main__":
    # Test the parser
    import sys
    
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
        success_rate = extract_success_rate_from_log(log_path)
        if success_rate is not None:
            print(f"Success rate: {success_rate:.4f} ({success_rate*100:.1f}%)")
    else:
        print("Usage: python log_parser.py <log_file_path>")

