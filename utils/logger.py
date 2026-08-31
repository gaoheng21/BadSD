"""
BadSD 实验日志系统
功能：
1. 记录所有运行输出到日志文件
2. 同时在控制台显示
3. 记录代码修改历史
4. 自动创建带时间戳的日志文件
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """实验日志记录器"""

    def __init__(self, log_dir='logs', experiment_name=None, to_file=False):
        """
        初始化日志记录器

        参数:
            log_dir: 日志目录
            experiment_name: 实验名称（可选）
            to_file: 是否将日志写入带时间戳的文件（默认 False，仅输出控制台）
        """
        self.log_dir = Path(log_dir)
        self.to_file = to_file

        # 生成日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if experiment_name:
            log_filename = f"{experiment_name}_{timestamp}.log"
        else:
            log_filename = f"experiment_{timestamp}.log"

        self.log_file = self.log_dir / log_filename
        self.change_log_file = self.log_dir / "code_changes.log"

        # 配置日志记录器
        self.logger = logging.getLogger('BadSD')
        self.logger.setLevel(logging.DEBUG)

        # 清除已有的处理器（避免重复）
        self.logger.handlers.clear()

        # 文件处理器 - 记录所有信息（仅当 to_file=True 时启用）
        if self.to_file:
            self.log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # 控制台处理器 - 显示INFO及以上级别
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        # 添加处理器
        self.logger.addHandler(console_handler)

        # 记录日志文件位置
        self.logger.info(f"=" * 80)
        self.logger.info(f"实验日志已启动")
        if self.to_file:
            self.logger.info(f"日志文件: {self.log_file.absolute()}")
        self.logger.info(f"=" * 80)

    def info(self, message):
        """记录INFO级别信息"""
        self.logger.info(message)

    def debug(self, message):
        """记录DEBUG级别信息"""
        self.logger.debug(message)

    def warning(self, message):
        """记录WARNING级别信息"""
        self.logger.warning(message)

    def error(self, message):
        """记录ERROR级别信息"""
        self.logger.error(message)

    def section(self, title):
        """记录分节标题"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f" {title}")
        self.logger.info("=" * 80)

    def subsection(self, title):
        """记录子节标题"""
        self.logger.info("")
        self.logger.info(f"--- {title} ---")

    def log_config(self, config_dict):
        """记录配置信息"""
        self.section("实验配置")
        for key, value in config_dict.items():
            self.logger.info(f"  {key:20s}: {value}")

    def log_results(self, results_dict):
        """记录实验结果"""
        self.section("实验结果")
        for key, value in results_dict.items():
            if isinstance(value, float):
                self.logger.info(f"  {key:20s}: {value:.4f}")
            else:
                self.logger.info(f"  {key:20s}: {value}")

    def log_code_change(self, file_path, change_description):
        """记录代码修改历史"""
        if not self.to_file:
            self.logger.info(f"代码修改: {file_path} - {change_description}")
            return
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.change_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"时间: {timestamp}\n")
            f.write(f"文件: {file_path}\n")
            f.write(f"修改: {change_description}\n")

        self.logger.info(f"代码修改已记录: {file_path}")

    def close(self):
        """关闭日志记录器"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("实验日志已关闭")
        self.logger.info("=" * 80)

        # 关闭所有处理器
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


# 全局日志实例
_global_logger = None


def get_logger(log_dir='logs', experiment_name=None, to_file=False):
    """
    获取全局日志实例

    参数:
        log_dir: 日志目录
        experiment_name: 实验名称
        to_file: 是否将日志写入带时间戳的文件（默认 False，仅输出控制台）

    返回:
        ExperimentLogger实例
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = ExperimentLogger(log_dir, experiment_name, to_file=to_file)
    return _global_logger


def close_logger():
    """关闭全局日志实例"""
    global _global_logger
    if _global_logger is not None:
        _global_logger.close()
        _global_logger = None
