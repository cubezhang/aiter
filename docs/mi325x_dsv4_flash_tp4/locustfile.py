import os
import json
import time
import logging
import glob
import random
import multiprocessing
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import csv
import gevent
import numpy as np
from flask import Blueprint, render_template, jsonify
from locust import HttpUser, task, constant, events, between
from locust.runners import MasterRunner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------
# 扩展 Web UI
# ------------------------
path = os.path.dirname(os.path.abspath(__file__))
extend = Blueprint(
    "extend",
    "extend_web_ui",
    static_folder=f"{path}/static/",
    static_url_path="/extend/static/",
    template_folder=f"{path}/templates/",
)

dataset_files: List[str] = []

# 默认配置值
DATASET_DIR = './llm_test_datasets-prod'
DEFAULT_MODEL_NAME = 'DeepSeek-V3.2'
DEFAULT_API_KEY = 'XXXXXX'
DEFAULT_MAX_TOKENS = 16384
DEFAULT_THINKING_ENABLED = False
DEFAULT_INCLUDE_USAGE = True

# 全局进程序号计数
process_counter = multiprocessing.Value("i", 0)
_locust_environment = None

# 存储测试开始时的参数配置
_test_config: Dict[str, Any] = {
    "datasets_dir": DATASET_DIR,
    "model_name": DEFAULT_MODEL_NAME,
    "api_key": DEFAULT_API_KEY,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "thinking_enabled": DEFAULT_THINKING_ENABLED,
    "include_usage": DEFAULT_INCLUDE_USAGE
}

# SLA历史记录保存目录
SLA_HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sla_history")
os.makedirs(SLA_HISTORY_DIR, exist_ok=True)
_current_sla_dir = None  # 当前测试的日志目录
_current_test_start_time = None  # 当前测试开始时间


class SLAManager:
    """SLA管理类，负责处理SLA相关的统计和检查"""

    def __init__(self, model_name: str = "Unknown", thinking_enabled: bool = False):
        self.sla_config = {
            "0-24k": {"max_tpot": 23, "context_range": (0, 24576)},
            "24k-48k": {"max_tpot": 35, "context_range": (24576, 49152)},
            "48k-128k": {"max_tpot": 50, "context_range": (49152, 131072)}
        }

        self.five_min_stats = {
            "start_time": None,
            "end_time": None,
            "request_count": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "avg_context_length": 0,
            "total_e2e": 0.0,
            "total_ttft": 0.0,
            "total_tpot": 0.0,
            "avg_tpot": 0.0,
            "total_concurrency": 0,
            "sla_violations": 0,
            "total_sla_checks": 0
        }

        # 添加停止标志
        self._stop_checking = False
        self._checker_greenlet = None

        # 存储测试配置
        self.model_name = model_name
        self.thinking_enabled = thinking_enabled
        self.concurrency = 0

        self.five_min_history = []
        self.csv_file = None
        self.csv_writer = None

    def _create_csv_file(self, log_dir: str) -> None:
        """创建CSV文件用于保存SLA历史记录"""
        # 格式化文件名：模型名称-思考状态-并发-时间戳
        thinking_status = "thinking_on" if self.thinking_enabled else "thinking_off"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name}-{thinking_status}-{timestamp}.csv"
        filepath = os.path.join(log_dir, filename)

        self.csv_file = open(filepath, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)

        # 写入CSV表头
        headers = [
            "model_name", "thinking_enabled", "request_count", "avg_context_length",
            "avg_tpot", "context_range", "max_allowed_tpot", "input_tpm",
            "output_tpm", "total_tpm", "ttft_ms", "e2e_s", "concurrency",
            "sla_violation", "sla_violations_count", "total_sla_checks",
            "sla_failure_rate", "start_time", "end_time"
        ]
        self.csv_writer.writerow(headers)

        logger.info(f"SLA历史记录将保存到: {filepath}")

    def reset_period_stats(self) -> None:
        """重置当前周期的统计数据，但保留累计的违规统计"""
        sla_violations = self.five_min_stats.get("sla_violations", 0)
        total_sla_checks = self.five_min_stats.get("total_sla_checks", 0)

        self.five_min_stats = {
            "start_time": time.time(),
            "end_time": None,
            "request_count": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "avg_context_length": 0,
            "total_e2e": 0.0,
            "total_ttft": 0.0,
            "total_tpot": 0.0,
            "avg_tpot": 0.0,
            "total_concurrency": 0,
            "sla_violations": sla_violations,  # 保留累计值
            "total_sla_checks": total_sla_checks,  # 保留累计值
        }

    def reset_all_stats(self) -> None:
        """重置所有统计数据，包括累计的违规统计"""
        self.five_min_stats = {
            "start_time": time.time(),
            "end_time": None,
            "request_count": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "avg_context_length": 0,
            "total_e2e": 0.0,
            "total_ttft": 0.0,
            "total_tpot": 0.0,
            "avg_tpot": 0.0,
            "total_concurrency": 0,
            "sla_violations": 0,  # 重置累计值
            "total_sla_checks": 0,  # 重置累计值
        }
        self.five_min_history.clear()

    def update_request_stats(self, input_tokens: int, output_tokens: int,
                             e2e_time: float, ttft_time: float, tpot_time: float, concurrency: int) -> None:
        """更新单个请求的统计数据"""
        self.five_min_stats["request_count"] += 1
        self.five_min_stats["total_input_tokens"] += input_tokens
        self.five_min_stats["total_output_tokens"] += output_tokens
        self.five_min_stats["total_e2e"] += e2e_time
        self.five_min_stats["total_ttft"] += ttft_time
        self.five_min_stats["total_tpot"] += tpot_time
        self.five_min_stats["total_concurrency"] += concurrency

        # 计算平均值
        request_count = max(1, self.five_min_stats["request_count"])
        self.five_min_stats["avg_context_length"] = round((
            self.five_min_stats["total_input_tokens"] +
            self.five_min_stats["total_output_tokens"]
        ) / request_count, 2)
        self.five_min_stats["avg_tpot"] = round(
            self.five_min_stats["total_tpot"] / request_count, 2
        )
        self.five_min_stats["avg_concurrency"] = round(
            self.five_min_stats["total_concurrency"] / request_count, 2
        )

    def check_sla(self) -> Dict[str, Any]:
        """检查SLA并返回检查结果"""

        self.five_min_stats["end_time"] = time.time()

        avg_context_length = self.five_min_stats.get("avg_context_length", 0)
        avg_tpot = self.five_min_stats.get("avg_tpot", 0)

        # 确定上下文长度范围
        sla_violation = False
        context_range = None
        max_allowed_tpot = 0

        if 0 <= avg_context_length < 24576:
            context_range = "0-24k"
            max_allowed_tpot = self.sla_config["0-24k"]["max_tpot"]
            if avg_tpot > max_allowed_tpot:
                sla_violation = True
        elif 24576 <= avg_context_length < 49152:
            context_range = "24k-48k"
            max_allowed_tpot = self.sla_config["24k-48k"]["max_tpot"]
            if avg_tpot > max_allowed_tpot:
                sla_violation = True
        elif 49152 <= avg_context_length < 131072:
            context_range = "48k-128k"
            max_allowed_tpot = self.sla_config["48k-128k"]["max_tpot"]
            if avg_tpot > max_allowed_tpot:
                sla_violation = True

        # 计算TPM
        total_time = max(1, self.five_min_stats["end_time"] - self.five_min_stats["start_time"])
        input_tps = self.five_min_stats["total_input_tokens"] / total_time
        output_tps = self.five_min_stats["total_output_tokens"] / total_time
        input_tpm = input_tps * 60 / 10000
        output_tpm = output_tps * 60 / 10000

        # 计算延迟指标
        ttft_ms = self.five_min_stats["total_ttft"] / max(1, self.five_min_stats["request_count"])
        e2e_s = self.five_min_stats["total_e2e"] / max(1, self.five_min_stats["request_count"])

        # 计算并发
        avg_concurrency = self.five_min_stats["total_concurrency"] / max(1, self.five_min_stats["request_count"])

        # 更新SLA违规统计
        if sla_violation:
            self.five_min_stats["sla_violations"] += 1

        self.five_min_stats["total_sla_checks"] += 1

        # 计算失败率
        sla_failure_rate = (
            self.five_min_stats["sla_violations"] /
            max(1, self.five_min_stats["total_sla_checks"])
        )

        # 记录当前五分钟统计信息
        current_stats = {
            "start_time": datetime.fromtimestamp(self.five_min_stats["start_time"]).strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": datetime.fromtimestamp(self.five_min_stats["end_time"]).strftime('%Y-%m-%d %H:%M:%S'),
            "request_count": self.five_min_stats["request_count"],
            "avg_context_length": round(avg_context_length, 0),
            "avg_tpot": round(avg_tpot, 2),
            "context_range": context_range,
            "max_allowed_tpot": max_allowed_tpot,
            "input_tpm": round(input_tpm, 2),
            "output_tpm": round(output_tpm, 2),
            "total_tpm": round(input_tpm + output_tpm, 2),
            "ttft_ms": round(ttft_ms, 2),
            "e2e_s": round(e2e_s, 2),
            "concurrency": round(avg_concurrency, 2),
            "sla_violation": sla_violation,
            "sla_violations_count": self.five_min_stats["sla_violations"],
            "total_sla_checks": self.five_min_stats["total_sla_checks"],
            "sla_failure_rate": round(sla_failure_rate, 4),
        }

        # 添加到历史记录
        self.five_min_history.append(current_stats)

        # 保存到CSV文件
        if self.csv_writer:
            csv_row = [
                self.model_name,
                self.thinking_enabled,
                current_stats["request_count"],
                current_stats["avg_context_length"],
                current_stats["avg_tpot"],
                current_stats["context_range"],
                current_stats["max_allowed_tpot"],
                current_stats["input_tpm"],
                current_stats["output_tpm"],
                current_stats["total_tpm"],
                current_stats["ttft_ms"],
                current_stats["e2e_s"],
                current_stats["concurrency"],
                current_stats["sla_violation"],
                current_stats["sla_violations_count"],
                current_stats["total_sla_checks"],
                current_stats["sla_failure_rate"],
                current_stats["start_time"],
                current_stats["end_time"]
            ]
            self.csv_writer.writerow(csv_row)
            self.csv_file.flush()  # 确保数据写入磁盘


        # 记录日志时显示间隔信息
        interval_minutes = self._check_interval / 60
        logger.info("=" * 100)
        logger.info(f"SLA检查 - 间隔: {self._check_interval}秒 ({interval_minutes:.1f}分钟)")
        logger.info(f"时间段: {current_stats['start_time']} 到 {current_stats['end_time']}")
        logger.info(f"请求数量: {current_stats['request_count']}")
        logger.info(f"平均上下文长度: {current_stats['avg_context_length']} tokens ({context_range})")
        logger.info(f"平均TPOT: {current_stats['avg_tpot']} ms")
        logger.info(f"SLA要求: ≤ {max_allowed_tpot} ms")
        logger.info(f"状态: {'不达标' if sla_violation else '达标'}")
        logger.info(f"累计SLA违规次数: {current_stats['sla_violations_count']}")
        logger.info("=" * 100)

        # 重置5分钟统计
        self.reset_period_stats()

        return current_stats

    def get_sla_status(self) -> Dict[str, Any]:
        """获取当前SLA状态"""
        avg_context_length = self.five_min_stats.get("avg_context_length", 0)
        avg_tpot = self.five_min_stats.get("avg_tpot", 0)

        # 确定当前SLA状态
        sla_violation = False
        if 0 <= avg_context_length < 24576:
            if avg_tpot > self.sla_config["0-24k"]["max_tpot"]:
                sla_violation = True
        elif 24576 <= avg_context_length < 49152:
            if avg_tpot > self.sla_config["24k-48k"]["max_tpot"]:
                sla_violation = True
        elif 49152 <= avg_context_length < 131072:
            if avg_tpot > self.sla_config["48k-128k"]["max_tpot"]:
                sla_violation = True

        return {
            "request_count": self.five_min_stats["request_count"],
            "avg_tpot": self.five_min_stats.get("avg_tpot", 0),
            "avg_context_length": self.five_min_stats.get("avg_context_length", 0),
            "total_sla_checks": self.five_min_stats["total_sla_checks"],
            "sla_violations": self.five_min_stats["sla_violations"],
            "sla_failure_rate": self.five_min_stats["sla_violations"] / max(1, self.five_min_stats["total_sla_checks"]),
            "current_sla_status": "pass" if not sla_violation else "fail",
            "history": self.five_min_history,
            "model_name": self.model_name,
            "thinking_enabled": self.thinking_enabled,
            "concurrency": self.five_min_stats.get("avg_concurrency", 0)
        }

    def start_periodic_check(self, interval_seconds: int = 300) -> None:
        """启动定时SLA检查"""
        # 先停止现有的检查（如果有）
        self.stop_periodic_check()

        self._check_interval = interval_seconds
        self.five_min_stats["start_time"] = time.time()
        self._stop_checking = False

        def checker():
            while not self._stop_checking:
                gevent.sleep(self._check_interval)
                if not self._stop_checking:  # 再次检查，避免在sleep期间被停止
                    self.check_sla()

        self._checker_greenlet = gevent.spawn(checker)
        logger.info(f"SLA定时检查已启动，间隔: {self._check_interval}秒")

    def stop_periodic_check(self) -> None:
        """停止定时SLA检查"""
        self._stop_checking = True
        if self._checker_greenlet:
            self._checker_greenlet.kill(block=False)
            self._checker_greenlet = None
            logger.info("SLA定时检查已停止")

    def restart_periodic_check(self) -> None:
        """重新启动定时SLA检查"""
        self.stop_periodic_check()
        self.start_periodic_check(self._check_interval)
        logger.info("SLA定时检查已重新启动")

    def close_csv_file(self) -> None:
        """关闭CSV文件"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None




class IntervalStats:
    """通用统计器，可配置 interval"""

    def __init__(self, name: str, interval: int, scale: int = 1,
                 report_to_ui: bool = True, max_history: int = 10,
                 log_every: Optional[int] = None, thinking_enabled: bool = False):
        self.name = name
        self.scale = scale
        self.report_to_ui = report_to_ui
        self.max_history = max_history  # 最大历史记录条数
        self.base_interval = max(1, interval)
        self._update_interval_value(interval)
        self.log_every = log_every
        self.thinking_enabled = thinking_enabled
        self.history: List[Dict[str, Any]] = []  # 存储历史记录
        self.reset()

    def _update_interval_value(self, interval: int) -> None:
        """更新间隔值"""
        self.base_interval = max(1, interval)
        self.interval = self.base_interval * self.scale

    def _update_thinking(self, thinking_enabled: bool) -> None:
        """更新思考模式设置"""
        self.thinking_enabled = thinking_enabled

    def reset(self) -> None:
        """重置统计计数器"""
        self.req_count = 0
        self.input_sum = 0
        self.output_sum = 0
        self.first_time = None
        self.last_time = None
        self.e2e_sum = 0
        self.ttft_sum = 0
        self.tpot_sum = 0

    def update(self, input_tokens: int, output_tokens: int,
               e2e: float, ttft: float, tpot: float) -> None:
        """更新统计数据"""
        now = time.perf_counter()
        if self.first_time is None:
            self.first_time = now
        self.last_time = now

        self.req_count += 1
        self.input_sum += input_tokens
        self.output_sum += output_tokens
        self.e2e_sum += e2e
        self.ttft_sum += ttft
        self.tpot_sum += tpot

        if self.req_count >= self.interval:
            self.report()
            self.reset()
            self.first_time = now

        if self.log_every and self.req_count % self.log_every == 0:
            self.report()
        elif not self.log_every and self.base_interval > 0 \
            and self.req_count % (self.base_interval * 5) == 0 \
            and self.req_count < self.interval and self.req_count != 0:
            self.report()

    def _calculate_stats(self) -> Dict[str, Any]:
        """计算当前统计指标"""
        total_time = (self.last_time - self.first_time) or 1
        input_tps = self.input_sum / total_time
        output_tps = self.output_sum / total_time
        total_tps = input_tps + output_tps
        input_tokens_mean = self.input_sum / self.req_count
        output_tokens_mean = self.output_sum / self.req_count

        input_tpm = input_tps * 60 / 10000
        output_tpm = output_tps * 60 / 10000
        total_tpm = input_tpm + output_tpm
        ttft_ms = self.ttft_sum / self.req_count if self.ttft_sum > 0 else 0
        tpot_ms = self.tpot_sum / self.req_count if self.tpot_sum > 0 else 0
        e2e_s = self.e2e_sum / self.req_count if self.e2e_sum > 0 else 0

        return {
            "req_count": self.req_count,
            "input_tps": input_tps,
            "output_tps": output_tps,
            "total_tps": total_tps,
            "input_tpm": input_tpm,
            "output_tpm": output_tpm,
            "total_tpm": total_tpm,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "e2e_s": e2e_s,
            "input_tokens_mean": input_tokens_mean,
            "output_tokens_mean": output_tokens_mean,
            "timestamp": time.time()
        }

    def _calculate_history_means(self) -> Optional[Dict[str, Any]]:
        """计算历史记录的均值"""
        if not self.history:
            return None

        means = {
            "total_req_count": sum(item["req_count"] for item in self.history),
            "mean_input_tpm": sum(item["input_tpm"] for item in self.history) / len(self.history),
            "mean_output_tpm": sum(item["output_tpm"] for item in self.history) / len(self.history),
            "mean_total_tpm": sum(item["total_tpm"] for item in self.history) / len(self.history),
            "mean_ttft_ms": sum(item["ttft_ms"] for item in self.history) / len(self.history),
            "mean_tpot_ms": sum(item["tpot_ms"] for item in self.history) / len(self.history),
            "mean_e2e_s": sum(item["e2e_s"] for item in self.history) / len(self.history),
            "mean_input_tokens": sum(item["input_tokens_mean"] for item in self.history) / len(self.history),
            "mean_output_tokens": sum(item["output_tokens_mean"] for item in self.history) / len(self.history),
            "history_count": len(self.history)
        }
        return means

    def report(self) -> None:
        """报告统计数据"""
        current_stats = self._calculate_stats()

        # 添加到历史记录
        self.history.append(current_stats)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        latencies = {
            "ttft": current_stats["ttft_ms"],
            "tpot": current_stats["tpot_ms"],
            "e2e": current_stats["e2e_s"],
        }

        thinking_label = "Thinking=ON" if self.thinking_enabled else "Thinking=OFF"

        logger.info('-' * 80)
        logger.info(f"[{self.name}] 性能汇总 ({self.req_count} req, {thinking_label})".center(80))
        logger.info(
            f"TPS: 总 {current_stats['total_tps']:>8.2f} w, 输入 {current_stats['input_tps']:>8.2f} w, 输出 {current_stats['output_tps']:>8.2f} w | "
            f"TTFT {latencies['ttft']:>6.2f} ms TPOT {latencies['tpot']:>6.2f} ms E2E {latencies['e2e']:>6.2f} s | "
            f"input_tokens_mean {current_stats['input_tokens_mean']:>4.0f} Output_len_usage: {current_stats['output_tokens_mean']:>4.0f}"
        )
        logger.info(
            f"TPM: 总 {current_stats['total_tpm']:>8.4f} w, 输入 {current_stats['input_tpm']:>8.4f} w, 输出 {current_stats['output_tpm']:>8.4f} w | "
        )

        self.show_history_mean()

        if self.report_to_ui:
            # 同步到 Locust UI
            try:
                env = _locust_environment
                env.stats.log_request("POST", f"LLM/TPM_Total_{self.name}", current_stats['total_tpm'], 0)
                env.stats.log_request("POST", f"LLM/TPM_Input_{self.name}", current_stats['input_tpm'], 0)
                env.stats.log_request("POST", f"LLM/TPM_Output_{self.name}", current_stats['output_tpm'], 0)
                env.stats.log_request("POST", f"LLM/TTFT_{self.name}", latencies['ttft'], 0)
                env.stats.log_request("POST", f"LLM/TPOT_{self.name}", latencies['tpot'], 0)
                env.stats.log_request("POST", f"LLM/E2E_{self.name}", latencies['e2e'], 0)
                env.stats.log_request("POST", f"LLM/input_len_usage_{self.name}", current_stats['input_tokens_mean'], 0)
                env.stats.log_request("POST", f"LLM/output_len_usage_{self.name}", current_stats['output_tokens_mean'], 0)
            except Exception as e:
                logger.warning(f"Failed to log to Locust: {e}")

    def show_history_mean(self) -> None:
        """显示历史均值"""
        # 计算历史均值
        history_means = self._calculate_history_means()

        # 显示历史均值
        if history_means:
            logger.info(f"历史均值--{self.name} ({history_means['history_count']} 次统计--统计间隔 {self.base_interval}): ".center(80))
            logger.info(
                f"Total_TPM {history_means['mean_total_tpm']:>8.4f} w | "
                f"Input_TPM {history_means['mean_input_tpm']:>8.4f} w | "
                f"Output_TPM {history_means['mean_output_tpm']:>8.4f} w | "
                f"TTFT {history_means['mean_ttft_ms']:>6.2f} ms | "
                f"TPOT {history_means['mean_tpot_ms']:>6.2f} ms | "
                f"E2E {history_means['mean_e2e_s']:>6.2f} s | "
                f"Input_tokens {history_means['mean_input_tokens']:>6.2f} | "
                f"Output_tokens {history_means['mean_output_tokens']:>6.2f} | "
            )

        logger.info('-' * 80)

    def get_history(self) -> List[Dict[str, Any]]:
        """获取完整的历史记录"""
        return self.history.copy()

    def get_history_means(self) -> Optional[Dict[str, Any]]:
        """获取历史记录的均值"""
        return self._calculate_history_means()

    def clear_history(self) -> None:
        """清空历史记录"""
        self.history.clear()


# 定义多个统计器
stats_intervals: Dict[str, IntervalStats] = {}


@events.init_command_line_parser.add_listener
def add_tps_interval_argument(parser):
    """添加自定义命令行参数"""
    parser.add_argument(
        "--interval",
        type=int,
        default=-1,
        help="Number of requests per TPS/TPM calculation interval (default: -1)"
    )
    parser.add_argument(
        "--sla-interval",
        type=int,
        default=300,
        help="SLA check interval in seconds (default: 300)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help=f"Path to dataset directory"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        choices=["DeepSeek-R1", "DeepSeek-V3", "DeepSeek-V3.1", "QwQ-32B", "DeepSeek-V3.1-Terminus", "DeepSeek-V3.2", "kimi-k3"],
        help=f"Model name to test (default: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        is_secret=True,
        default=None,
        help=f"API key for authentication"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})"
    )
    parser.add_argument(
        "--thinking-enabled",
        type=lambda x: (str(x).lower() in ['true', '1', 'yes', 'y']),
        default=DEFAULT_THINKING_ENABLED,
        help=f"Enable thinking mode (default: {DEFAULT_THINKING_ENABLED})"
    )
    parser.add_argument(
        "--include-usage",
        type=lambda x: (str(x).lower() in ['true', '1', 'yes', 'y']),
        default=DEFAULT_INCLUDE_USAGE,
        help=f"Include usage information in stream (default: {DEFAULT_INCLUDE_USAGE})"
    )


@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    """初始化 master/worker"""
    global _locust_environment, stats_intervals
    global _test_config, _current_sla_dir
    global SLA_HISTORY_DIR, _current_test_start_time

    if not environment.runner:
        return
    _locust_environment = environment

    base_interval = getattr(environment.parsed_options, "interval", -1)
    stats_intervals = {
        "1x": IntervalStats("1x", base_interval, report_to_ui=False, max_history=20),
        "2x": IntervalStats("2x", base_interval, 2, report_to_ui=False, max_history=20),
        "5x": IntervalStats("5x", base_interval, 5, report_to_ui=True, max_history=20),
        "all": IntervalStats("all", base_interval, 9999999, report_to_ui=False, max_history=20, log_every=10, thinking_enabled=_test_config.get("thinking_enabled", False)),  # 无限累计
    }

    def handle_request_update(msg, environment):
        on_request_update(msg, environment)


    # 仅在Master节点初始化SLA管理器
    if isinstance(environment.runner, MasterRunner):

        # 只在初始化时创建基础目录，不创建具体测试目录
        _current_sla_dir = None
        _current_test_start_time = None

        environment.runner.sla_manager = SLAManager(
            model_name=_test_config["model_name"],
            thinking_enabled=_test_config["thinking_enabled"]

        )


    if environment.web_ui:
        @extend.route("/api/sla")
        def sla_api():
            if hasattr(environment.runner, 'sla_manager'):
                data = environment.runner.sla_manager.get_sla_status()
            else:
                data = {"error": "SLA manager not available on this node"}

            all_stats = stats_intervals.get("all")
            if all_stats:
                all_history_mean = all_stats.get_history_means()
                if all_history_mean:
                    # 将x1的历史均值数据添加到返回的data中，加一个前缀避免冲突
                    data["all_mean_total_tpm"] = all_history_mean["mean_total_tpm"]
                    data["all_mean_input_tpm"] = all_history_mean["mean_input_tpm"]
                    data["all_mean_output_tpm"] = all_history_mean["mean_output_tpm"]

            response = jsonify(data)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response

        @extend.route("/sla")
        def sla_dashboard():
            return render_template("sla_dashboard_0916.html")

        # 注册蓝图
        environment.web_ui.app.register_blueprint(extend)

    environment.runner.register_message("request_update", handle_request_update)


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """测试开始时触发"""
    global _test_config, DATASET_DIR, DEFAULT_MODEL_NAME, DEFAULT_API_KEY
    global DEFAULT_MAX_TOKENS, DEFAULT_THINKING_ENABLED, DEFAULT_INCLUDE_USAGE
    global _current_sla_dir, SLA_HISTORY_DIR, _current_test_start_time

    # 从命令行参数获取配置并保存到全局变量
    _test_config = {
        "datasets_dir": getattr(environment.parsed_options, "dataset_dir", DATASET_DIR),
        "model_name": getattr(environment.parsed_options, "model_name", DEFAULT_MODEL_NAME),
        "api_key": getattr(environment.parsed_options, "api_key", DEFAULT_API_KEY),
        "max_tokens": getattr(environment.parsed_options, "max_tokens", DEFAULT_MAX_TOKENS),
        "thinking_enabled": getattr(environment.parsed_options, "thinking_enabled", DEFAULT_THINKING_ENABLED),
        "include_usage": getattr(environment.parsed_options, "include_usage", DEFAULT_INCLUDE_USAGE),
        "sla_interval": getattr(environment.parsed_options, "sla_interval", 300)
    }

    # 如果api_key为空，使用默认值
    if not _test_config["api_key"]:
        _test_config["api_key"] = DEFAULT_API_KEY

    if not _test_config["datasets_dir"]:
        _test_config["datasets_dir"] = DATASET_DIR

    logger.info(
        f"测试开始，参数配置: "
        f"datasets_dir={_test_config['datasets_dir']}, "
        f"model={_test_config['model_name']}, "
        f"api_key={'(set)' if _test_config['api_key'] else '(missing)'}, "
        f"max_tokens={_test_config['max_tokens']}, "
        f"thinking={_test_config['thinking_enabled']}, "
        f"include_usage={_test_config['include_usage']}, "
        f"sla_interval={_test_config['sla_interval']}s"
    )

    # 记录测试开始时间
    _current_test_start_time = datetime.now()

    # 如果SLA管理器已存在，更新其配置
    if hasattr(environment.runner, 'sla_manager'):
        environment.runner.sla_manager.model_name = _test_config["model_name"]
        environment.runner.sla_manager.thinking_enabled = _test_config["thinking_enabled"]

        # 创建当前测试的日志目录（每次Start都创建新的）
        if _current_sla_dir is None:
            timestamp = _current_test_start_time.strftime("%Y%m%d_%H%M%S")
            log_dir_name = f"{_test_config['model_name']}-{timestamp}"
            _current_sla_dir = os.path.join(SLA_HISTORY_DIR, log_dir_name)
            os.makedirs(_current_sla_dir, exist_ok=True)

        # 创建新的CSV文件
        environment.runner.sla_manager._create_csv_file(_current_sla_dir)
        # 使用配置的SLA间隔启动定时检查
        sla_interval = _test_config["sla_interval"]
        environment.runner.sla_manager.start_periodic_check(sla_interval)


    load_dataset_files()
    update_setting(environment, "test_start")


def load_dataset_files() -> None:
    """加载 JSON 数据集文件"""
    global dataset_files, _test_config
    dataset_files = []
    dataset_dir = _test_config["datasets_dir"]

    if not os.path.exists(dataset_dir):
        logger.critical(f"Dataset dir not found: {dataset_dir}")
        os._exit(1)

    dataset_files = glob.glob(os.path.join(dataset_dir, '*.json'))
    if not dataset_files:
        logger.critical(f"No JSON files found in {dataset_dir}")
        os._exit(1)


def update_setting(environment, name: str) -> None:
    """更新统计间隔"""
    if isinstance(environment.runner, MasterRunner):
        global stats_intervals, _test_config

        # 读取命令行参数
        if hasattr(environment.parsed_options, 'interval'):
            base_interval = environment.parsed_options.interval
            user_count = environment.runner.user_count

            if base_interval == -1:
                for s in stats_intervals.values():
                    s._update_interval_value(user_count)
                    s._update_thinking(_test_config.get("thinking_enabled", False))
                logger.info(f"✅ 统计间隔 user_count 为: {user_count}")
            else:
                for s in stats_intervals.values():
                    s._update_interval_value(base_interval)
                    s._update_thinking(_test_config.get("thinking_enabled", False))
                logger.info(f"✅ 统计间隔 base_interval 已设置为: {base_interval}")
        else:
            base_interval = 50  # 默认值
            logger.warning(f"⚠️ 未设置 --interval，使用默认值: {base_interval}")


@events.reset_stats.add_listener
def on_reset_stats(**kwargs) -> None:
    """重置全局统计数据"""
    global stats_intervals
    for s in stats_intervals.values():
        s.reset()
        s.clear_history()

    # 重置SLA统计（仅在Master节点）
    if hasattr(_locust_environment.runner, 'sla_manager'):
        _locust_environment.runner.sla_manager.reset_all_stats()
        # 重新启动定时检查
        _locust_environment.runner.sla_manager.restart_periodic_check()
    logger.info("on_reset_stats: 全局统计数据已重置")


@events.spawning_complete.add_listener
def on_spawning_complete(user_count, **kwargs):
    """用户生成完成时触发"""
    global _locust_environment

    if _locust_environment is None:
        return

    if not isinstance(_locust_environment.runner, MasterRunner):
        return

    update_setting(_locust_environment, "on_spawning_complete")
    logger.info(f"所有用户已启动，共 {user_count} 个用户")


def on_request_update(msg, environment):
    """Master 收到 Worker 上报的请求统计"""
    data = msg.data
    input_tokens = data.get("input_len_usage", 0)
    output_tokens = data.get("output_len_usage", 0)
    e2e_time = data.get("e2e_times", 0)
    ttft_time = data.get("TTFT", 0)
    tpot_time = data.get("TPOT", 0)
    worker_id = data.get("worker_id", "unknown")
    thinking_enabled = data.get("think", False)
    concurrency = environment.runner.user_count if environment.runner else 0

    # 如果都是0或无效，不统计
    if input_tokens <= 0 and output_tokens <= 0 and e2e_time <= 0:
        return

    # 更新SLA统计（仅在Master节点）
    if hasattr(environment.runner, 'sla_manager'):
        environment.runner.sla_manager.update_request_stats(input_tokens, output_tokens, e2e_time, ttft_time, tpot_time, concurrency)

    # 记录每个请求的统计数据
    logger.info(
        f"input_tokens: {input_tokens:>8} | "
        f"output_tokens: {output_tokens:>8} | "
        f"e2e_time: {e2e_time:>6.2f} s | "
        f"ttft_time: {ttft_time:>8.2f} ms | "
        f"tpot_time: {tpot_time:>7.2f} ms | "
        f"worker_id: {worker_id:>10} | "
        f"global_request_count: {stats_intervals['5x'].req_count} | "
        f"thinking: {thinking_enabled}"
    )

    for s in stats_intervals.values():
        s.update(
            input_tokens,
            output_tokens,
            e2e_time,
            ttft_time,
            tpot_time
        )


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """测试停止时触发（包括Web界面停止）"""
    if isinstance(environment.runner, MasterRunner):
        logger.info("测试停止 - Master节点")
        # 最后一次SLA检查
        if hasattr(environment.runner, 'sla_manager'):
            environment.runner.sla_manager.stop_periodic_check()
            # environment.runner.sla_manager.check_sla()
            environment.runner.sla_manager.close_csv_file()
        handle_web_stop(environment)


def handle_web_stop(environment):
    """处理Web界面停止事件"""
    logger.info("Web界面停止按钮被点击")
    global stats_intervals

    for s in stats_intervals.values():
        s.show_history_mean()


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Locust退出时触发"""
    if hasattr(environment.runner, 'sla_manager'):
        environment.runner.sla_manager.stop_periodic_check()
        environment.runner.sla_manager.close_csv_file()



class OpenAIUser(HttpUser):
    """OpenAI API 压力测试用户"""
    wait_time = constant(0)
    #wait_time = between(5, 100)
    def on_start(self):
        # 设置请求头强制关闭连接
        pass
        #self.client.headers.update({'Connection': 'close'})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random = random.Random()

        # 如果 runner 没有 worker_id，则生成一个本地唯一 ID
        self.local_worker_id = getattr(self.environment.runner, "worker_id", None)
        if self.local_worker_id is None:
            with process_counter.get_lock():
                process_counter.value += 1
                self.local_worker_id = f"worker-{process_counter.value}"

        # 使用测试开始时保存的配置
        global _test_config
        self.model_name = _test_config["model_name"]
        self.api_key = _test_config["api_key"]
        self.max_tokens = _test_config["max_tokens"]
        self.thinking_enabled = _test_config["thinking_enabled"]
        self.include_usage = _test_config["include_usage"]

        logger.debug(f"用户初始化，使用配置: model={self.model_name}, max_tokens={self.max_tokens}")

    @task
    def chat_completion(self):
        """执行聊天完成请求"""
        try:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            dataset_file = self.random.choice(dataset_files)
            with open(dataset_file, 'r') as f:
                data = json.load(f)
            messages.extend(data)
        except IndexError:
            logger.error("Empty prompts dataset")
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "max_tokens": self.max_tokens,
        }

        # 只有在启用thinking时才添加chat_template_kwargs
        self.thinking_enabled = random.random() < 0.4
        if self.thinking_enabled:
            payload["chat_template_kwargs"] = {
                "thinking": True
            }

        # 只有在启用include_usage时才添加stream_options
        if self.include_usage:
            payload["stream_options"] = {
                "include_usage": True
            }

        first_token_received = False
        first_token_time = None
        usage_data = None
        input_len_usage = 0
        output_len_usage = 0
        start_time = time.perf_counter()  # 记录请求开始时间

        try:
            with self.client.post(
                "/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
                catch_response=True,
            ) as response:

                if response.status_code != 200:
                    error_msg = f"HTTP {response.status_code} - {response.text}"
                    response.failure(error_msg)
                    logger.error(error_msg)
                    return

                # 使用 iter_content 更底层地读取流式响应
                for chunk in response.iter_lines():
                    if not chunk:
                        continue

                    if not first_token_received:
                        first_token_time = time.perf_counter()
                        first_token_received = True

                    # 解析 chunk 内容
                    try:
                        lines = chunk.decode("utf-8").split("\n")
                        for line in lines:
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                continue

                            data = json.loads(data_str)

                            # 处理usage数据（最后一个chunk）
                            if "usage" in data and data["usage"] is not None:
                                usage_data = data["usage"]
                    except Exception as e:
                        logger.error(f"Chunk解析失败: {e}")

            end_time = time.perf_counter()  # 请求结束时间

            if usage_data is not None:
                input_len_usage = usage_data.get("prompt_tokens", 0)
                output_len_usage = usage_data.get("completion_tokens", 0)

                e2e_s = (end_time - start_time) if start_time else 0

                # TTFT：从请求开始到第一个 token 返回的时间
                ttft_ms = 0  # 默认值
                if first_token_received and start_time:
                    ttft_ms = (first_token_time - start_time) * 1000

                # TPOT：每个 token 的平均间隔时间
                tpot_ms = 0  # 默认值
                if output_len_usage >= 2:
                    generation_duration = end_time - first_token_time
                    tpot_ms = (generation_duration / (output_len_usage - 1)) * 1000

                # 上报 Master
                if self.environment.runner:
                    self.environment.runner.send_message("request_update", {
                        "input_len_usage": input_len_usage,
                        "output_len_usage": output_len_usage,
                        "e2e_times": round(e2e_s, 2) if start_time else 0.0,
                        "TTFT": round(ttft_ms, 2) if start_time else 0.0,
                        "TPOT": round(tpot_ms, 2) if start_time else 0.0,
                        "worker_id": self.local_worker_id,
                        "think": self.thinking_enabled
                    })

        except Exception as e:
            logger.exception(f"请求异常: {str(e)}")
