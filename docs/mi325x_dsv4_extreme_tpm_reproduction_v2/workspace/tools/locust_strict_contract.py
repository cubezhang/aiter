#!/usr/bin/env python3
"""Finite, distributed Locust driver for the strict MI325X TPM contract.

Every manifest entry is attempted exactly once. Each worker owns the entries whose
position modulo worker_count equals its immutable Locust worker index.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gevent
from gevent.lock import Semaphore
from locust import HttpUser, constant, events, task
from locust.exception import StopUser
from locust.runners import LocalRunner, MasterRunner, WorkerRunner


SYSTEM_PROMPT = "You are a helpful assistant."
EXPECTED_COMPLETION_TOKENS = 1024

state_lock = Semaphore()
write_lock = Semaphore()
queue: deque[dict[str, Any]] = deque()
inflight = 0
done_sent = False
raw_file = None
worker_index = -1
worker_count = 1
dataset_dir = Path(".")
run_id = "unset"
master_done_nodes: set[str] = set()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


@events.init_command_line_parser.add_listener
def add_contract_arguments(parser, **_kwargs):
    parser.add_argument("--strict-manifest", required=True)
    parser.add_argument("--strict-dataset-dir", required=True)
    parser.add_argument("--strict-result-dir", required=True)
    parser.add_argument("--strict-worker-count", type=int, default=8)
    parser.add_argument("--strict-run-id", required=True)


def on_shard_done(environment, msg, **_kwargs):
    if not isinstance(environment.runner, MasterRunner):
        return
    master_done_nodes.add(msg.node_id)
    result_dir = Path(environment.parsed_options.strict_result_dir)
    write_json(
        result_dir / "master_progress.json",
        {
            "run_id": environment.parsed_options.strict_run_id,
            "updated_at_utc": utc_now(),
            "done_count": len(master_done_nodes),
            "expected_workers": environment.parsed_options.strict_worker_count,
            "done_nodes": sorted(master_done_nodes),
            "last_message": msg.data,
        },
    )
    if len(master_done_nodes) == environment.parsed_options.strict_worker_count:
        write_json(
            result_dir / "master_complete.json",
            {
                "run_id": environment.parsed_options.strict_run_id,
                "completed_at_utc": utc_now(),
                "done_nodes": sorted(master_done_nodes),
            },
        )
        gevent.spawn_later(0.5, environment.runner.quit)


@events.init.add_listener
def on_init(environment, **_kwargs):
    if environment.runner:
        environment.runner.register_message("strict_shard_done", on_shard_done)


@events.test_start.add_listener
def on_test_start(environment, **_kwargs):
    global queue, inflight, done_sent, raw_file, worker_index, worker_count, dataset_dir, run_id
    if isinstance(environment.runner, MasterRunner):
        master_done_nodes.clear()
        result_dir = Path(environment.parsed_options.strict_result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            result_dir / "master_start.json",
            {
                "run_id": environment.parsed_options.strict_run_id,
                "started_at_utc": utc_now(),
                "expected_workers": environment.parsed_options.strict_worker_count,
                "target_users": environment.runner.target_user_count,
            },
        )
        return

    opts = environment.parsed_options
    result_dir = Path(opts.strict_result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(opts.strict_manifest)
    dataset_dir = Path(opts.strict_dataset_dir)
    worker_count = opts.strict_worker_count
    run_id = opts.strict_run_id
    if isinstance(environment.runner, WorkerRunner):
        worker_index = environment.runner.worker_index
    elif isinstance(environment.runner, LocalRunner):
        worker_index = 0
        worker_count = 1
    else:
        raise RuntimeError(f"unsupported runner: {type(environment.runner).__name__}")
    if worker_index < 0 or worker_index >= worker_count:
        raise RuntimeError(f"invalid worker index {worker_index} for worker_count={worker_count}")

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    shard = [row for row in rows if int(row["position"]) % worker_count == worker_index]
    queue = deque(shard)
    inflight = 0
    done_sent = False
    raw_path = result_dir / f"requests_worker_{worker_index:02d}.jsonl"
    raw_file = raw_path.open("a", encoding="utf-8", buffering=1)
    write_json(
        result_dir / f"worker_{worker_index:02d}_start.json",
        {
            "run_id": run_id,
            "started_at_utc": utc_now(),
            "pid": os.getpid(),
            "worker_index": worker_index,
            "worker_count": worker_count,
            "shard_count": len(shard),
            "first_position": shard[0]["position"] if shard else None,
            "last_position": shard[-1]["position"] if shard else None,
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
        },
    )
    if not shard:
        notify_done(environment)


def notify_done(environment) -> None:
    global done_sent
    with state_lock:
        if done_sent or queue or inflight:
            return
        done_sent = True
    environment.runner.send_message(
        "strict_shard_done",
        {"run_id": run_id, "worker_index": worker_index, "completed_at_utc": utc_now()},
    )


def reserve_entry() -> dict[str, Any] | None:
    global inflight
    with state_lock:
        if not queue:
            return None
        entry = queue.popleft()
        inflight += 1
        return entry


def release_entry(environment) -> None:
    global inflight
    with state_lock:
        inflight -= 1
        if inflight < 0:
            raise RuntimeError("negative inflight count")
    notify_done(environment)


def append_result(record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    with write_lock:
        raw_file.write(line)
        raw_file.flush()


def finite_numbers(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    return True


class StrictContractUser(HttpUser):
    wait_time = constant(0)

    @task
    def one_request(self):
        entry = reserve_entry()
        if entry is None:
            notify_done(self.environment)
            raise StopUser()

        started_wall_ns = time.time_ns()
        started_ns = time.perf_counter_ns()
        first_token_ns = None
        last_token_ns = None
        response_end_ns = None
        usage = None
        done_seen = False
        finish_reasons: list[str] = []
        parse_errors: list[str] = []
        content_chars = 0
        reasoning_chars = 0
        generated_events = 0
        sse_data_events = 0
        response_bytes = 0
        status_code = None
        backend = None
        exception_text = None

        dataset_path = dataset_dir / entry["relative_filename"]
        try:
            if file_sha256(dataset_path) != entry["file_sha256"]:
                raise ValueError("dataset file SHA256 mismatch")
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            dataset_messages = json.loads(dataset_path.read_text(encoding="utf-8"))
            if not isinstance(dataset_messages, list):
                raise ValueError("dataset payload is not a message list")
            messages.extend(dataset_messages)
            payload: dict[str, Any] = {
                "model": "DeepSeek-V4",
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": EXPECTED_COMPLETION_TOKENS,
                "ignore_eos": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "n": 1,
            }
            payload["chat_template_kwargs"] = {
                "thinking": bool(entry["thinking"]),
            }
            headers = {"Authorization": "Bearer XXXXXX", "Content-Type": "application/json"}

            with self.client.post(
                "/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
                catch_response=True,
                name="strict_chat_1024",
                timeout=None,
            ) as response:
                status_code = response.status_code
                backend = response.headers.get("x-mi325-backend") or response.headers.get("x-backend")
                if status_code == 200:
                    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                        line_ns = time.perf_counter_ns()
                        if line is None:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", errors="strict")
                        response_bytes += len(line.encode("utf-8"))
                        if not line or not line.startswith("data:"):
                            continue
                        data_text = line[5:].strip()
                        if data_text == "[DONE]":
                            done_seen = True
                            continue
                        try:
                            event = json.loads(data_text)
                            sse_data_events += 1
                        except Exception as exc:
                            parse_errors.append(f"{type(exc).__name__}: {exc}"[:300])
                            continue
                        if event.get("usage") is not None:
                            usage = event["usage"]
                        for choice in event.get("choices") or []:
                            reason = choice.get("finish_reason")
                            if reason:
                                finish_reasons.append(str(reason))
                            delta = choice.get("delta") or {}
                            content = delta.get("content")
                            reasoning = delta.get("reasoning_content")
                            if reasoning is None:
                                reasoning = delta.get("reasoning")
                            chars = 0
                            if isinstance(content, str) and content:
                                content_chars += len(content)
                                chars += len(content)
                            if isinstance(reasoning, str) and reasoning:
                                reasoning_chars += len(reasoning)
                                chars += len(reasoning)
                            if chars:
                                generated_events += 1
                                if first_token_ns is None:
                                    first_token_ns = line_ns
                                last_token_ns = line_ns
                else:
                    exception_text = f"HTTP {status_code}: {response.text[:500]}"
                response_end_ns = time.perf_counter_ns()

                prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
                total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
                cached_tokens = 0
                if isinstance(usage, dict):
                    details = usage.get("prompt_tokens_details") or {}
                    if isinstance(details, dict):
                        cached_tokens = details.get("cached_tokens") or 0
                valid = all(
                    [
                        status_code == 200,
                        not parse_errors,
                        done_seen,
                        isinstance(usage, dict),
                        isinstance(prompt_tokens, int) and prompt_tokens > 0,
                        completion_tokens == EXPECTED_COMPLETION_TOKENS,
                        isinstance(total_tokens, int) and total_tokens == prompt_tokens + completion_tokens,
                        first_token_ns is not None,
                        last_token_ns is not None,
                        content_chars + reasoning_chars > 0,
                        finite_numbers(usage),
                        cached_tokens == 0,
                    ]
                )
                if valid:
                    response.success()
                else:
                    reasons = []
                    if status_code != 200:
                        reasons.append(f"http_{status_code}")
                    if parse_errors:
                        reasons.append("sse_parse")
                    if not done_seen:
                        reasons.append("missing_done")
                    if not isinstance(usage, dict):
                        reasons.append("missing_usage")
                    if completion_tokens != EXPECTED_COMPLETION_TOKENS:
                        reasons.append(f"completion_tokens_{completion_tokens}")
                    if first_token_ns is None:
                        reasons.append("missing_generated_token")
                    if cached_tokens:
                        reasons.append(f"cached_tokens_{cached_tokens}")
                    exception_text = ",".join(reasons) or "invalid_response"
                    response.failure(exception_text)
        except Exception as exc:
            response_end_ns = time.perf_counter_ns()
            exception_text = f"{type(exc).__name__}: {exc}"[:500]

        if response_end_ns is None:
            response_end_ns = time.perf_counter_ns()
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        cached_tokens = 0
        if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens_details"), dict):
            cached_tokens = usage["prompt_tokens_details"].get("cached_tokens") or 0
        valid = all(
            [
                status_code == 200,
                not parse_errors,
                done_seen,
                isinstance(usage, dict),
                isinstance(prompt_tokens, int) and prompt_tokens > 0,
                completion_tokens == EXPECTED_COMPLETION_TOKENS,
                isinstance(total_tokens, int) and total_tokens == prompt_tokens + completion_tokens,
                first_token_ns is not None,
                last_token_ns is not None,
                content_chars + reasoning_chars > 0,
                finite_numbers(usage),
                cached_tokens == 0,
                exception_text is None,
            ]
        )
        ttft_ms = (first_token_ns - started_ns) / 1e6 if first_token_ns is not None else None
        tpot_ms = (
            (last_token_ns - first_token_ns) / 1e6 / (completion_tokens - 1)
            if first_token_ns is not None and last_token_ns is not None and completion_tokens and completion_tokens >= 2
            else None
        )
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "request_id": entry["request_id"],
            "manifest_position": entry["position"],
            "relative_filename": entry["relative_filename"],
            "file_sha256": entry["file_sha256"],
            "thinking": bool(entry["thinking"]),
            "worker_index": worker_index,
            "started_at_unix_ns": started_wall_ns,
            "started_perf_ns": started_ns,
            "first_token_perf_ns": first_token_ns,
            "last_token_perf_ns": last_token_ns,
            "response_end_perf_ns": response_end_ns,
            "e2e_ms": (response_end_ns - started_ns) / 1e6,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "status_code": status_code,
            "valid_success": bool(valid),
            "error": exception_text,
            "parse_errors": parse_errors,
            "done_seen": done_seen,
            "finish_reasons": finish_reasons,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "content_chars": content_chars,
            "reasoning_chars": reasoning_chars,
            "generated_events": generated_events,
            "sse_data_events": sse_data_events,
            "response_bytes": response_bytes,
            "backend": backend,
        }
        append_result(record)
        release_entry(self.environment)


@events.test_stop.add_listener
def on_test_stop(environment, **_kwargs):
    global raw_file
    if raw_file is not None:
        raw_file.flush()
        raw_file.close()
        raw_file = None
