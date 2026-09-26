"""
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """
    """

    def __init__(self, log_dir='logs', experiment_name=None, to_file=False):
        """
        """
        self.log_dir = Path(log_dir)
        self.to_file = to_file

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if experiment_name:
            log_filename = f"{experiment_name}_{timestamp}.log"
        else:
            log_filename = f"experiment_{timestamp}.log"

        self.log_file = self.log_dir / log_filename
        self.change_log_file = self.log_dir / "code_changes.log"

        self.logger = logging.getLogger('BadSD')
        self.logger.setLevel(logging.DEBUG)

        self.logger.handlers.clear()

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

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(console_handler)

        self.logger.info(f"=" * 80)
        self.logger.info(f"实验日志已启动")
        if self.to_file:
            self.logger.info(f"日志文件: {self.log_file.absolute()}")
        self.logger.info(f"=" * 80)

    def info(self, message):
        """
        """
        self.logger.info(message)

    def debug(self, message):
        """
        """
        self.logger.debug(message)

    def warning(self, message):
        """
        """
        self.logger.warning(message)

    def error(self, message):
        """
        """
        self.logger.error(message)

    def section(self, title):
        """
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f" {title}")
        self.logger.info("=" * 80)

    def subsection(self, title):
        """
        """
        self.logger.info("")
        self.logger.info(f"--- {title} ---")

    def log_config(self, config_dict):
        """
        """
        self.section("实验配置")
        for key, value in config_dict.items():
            self.logger.info(f"  {key:20s}: {value}")

    def log_results(self, results_dict):
        """
        """
        self.section("实验结果")
        for key, value in results_dict.items():
            if isinstance(value, float):
                self.logger.info(f"  {key:20s}: {value:.4f}")
            else:
                self.logger.info(f"  {key:20s}: {value}")

    def log_code_change(self, file_path, change_description):
        """
        """
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
        """
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("实验日志已关闭")
        self.logger.info("=" * 80)

        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


_global_logger = None


def get_logger(log_dir='logs', experiment_name=None, to_file=False):
    """
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = ExperimentLogger(log_dir, experiment_name, to_file=to_file)
    return _global_logger


def close_logger():
    """
    """
    global _global_logger
    if _global_logger is not None:
        _global_logger.close()
        _global_logger = None
