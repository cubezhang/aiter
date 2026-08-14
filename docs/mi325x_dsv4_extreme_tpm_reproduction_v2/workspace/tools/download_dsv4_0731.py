#!/usr/bin/env python3
"""Resumable, supervised downloader for the pinned DeepSeek-V4-Flash-0731 release."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
API_URL = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true"
EXPECTED_FILES = 74
EXPECTED_SHARDS = 48
# Sum of all 74 repository files. HF's ``usedStorage`` only covers the LFS
# objects (166,886,718,583 bytes) and excludes the small Git-managed files.
EXPECTED_TOTAL_BYTES = 166_898_660_330

DOWNLOAD_DIR = Path("/data/DeepSeek-V4-Flash-0731")
WORK_DIR = Path("/data/download_dsv4_0731")
MANIFEST = WORK_DIR / "manifest.json"
STATE = WORK_DIR / "state.json"
LOG = WORK_DIR / "download.log"
CLI_LOG = WORK_DIR / "hf_cli.log"
DAEMON_LOG = WORK_DIR / "daemon.log"
PID_FILE = WORK_DIR / "supervisor.pid"
LOCK_FILE = WORK_DIR / "supervisor.lock"

HF_BIN = Path("/data/hf_env/bin/hf")
HF_PYTHON = Path("/data/hf_env/bin/python")
RESUMABLE_DOWNLOADER = Path(__file__).with_name("hf_resumable_snapshot.py")
WORKERS = 48
POLL_SECONDS = 30
STALL_SECONDS = 30 * 60
DISK_RESERVE = 20 * 1024**3
LOW_RATE_BPS = 6 * 1024**2
LOW_RATE_WINDOWS = 6  # six 30-second samples = three minutes
LOW_RATE_GRACE_SECONDS = 10 * 60


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def human(n: float) -> str:
    n = float(max(0, n))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    value = int(seconds)
    days, value = divmod(value, 86400)
    hours, value = divmod(value, 3600)
    minutes, _ = divmod(value, 60)
    return f"{days}d {hours:02d}h {minutes:02d}m" if days else f"{hours}h {minutes:02d}m"


def append_log(message: str) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{now()} | {message}\n")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def read_pid() -> int:
    try:
        return int(PID_FILE.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return 0


def fetch_manifest() -> dict:
    headers = {"User-Agent": "dsv4-0731-supervised-downloader/1.0"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(API_URL, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = json.load(response)
    if raw.get("sha") != REVISION:
        raise RuntimeError(f"revision mismatch: {raw.get('sha')} != {REVISION}")
    files = []
    for item in raw.get("siblings", []):
        name = item.get("rfilename")
        size = item.get("size")
        if not name or not isinstance(size, int):
            raise RuntimeError(f"manifest entry lacks exact size: {item}")
        lfs = item.get("lfs") or {}
        files.append({"path": name, "size": size, "sha256": lfs.get("sha256")})
    shards = [f for f in files if f["path"].startswith("model-") and f["path"].endswith(".safetensors")]
    total = sum(f["size"] for f in files)
    if len(files) != EXPECTED_FILES or len(shards) != EXPECTED_SHARDS or total != EXPECTED_TOTAL_BYTES:
        raise RuntimeError(
            f"unexpected repository inventory: files={len(files)}, shards={len(shards)}, bytes={total}"
        )
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "created_at": now(),
        "total_bytes": total,
        "files": files,
    }
    atomic_json(MANIFEST, manifest)
    return manifest


def load_manifest(refresh: bool = False) -> dict:
    cached = read_json(MANIFEST)
    if not refresh and cached.get("revision") == REVISION and len(cached.get("files", [])) == EXPECTED_FILES:
        return cached
    return fetch_manifest()


def scan(manifest: dict) -> dict:
    complete_files = 0
    complete_shards = 0
    final_bytes = 0
    cache_dir = DOWNLOAD_DIR / ".cache" / "huggingface" / "download"
    completed_content_hashes: set[str] = set()
    for item in manifest["files"]:
        path = DOWNLOAD_DIR / item["path"]
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == item["size"]:
            complete_files += 1
            final_bytes += size
            metadata = cache_dir / f"{item['path']}.metadata"
            try:
                metadata_lines = metadata.read_text().splitlines()
                if len(metadata_lines) >= 2 and re.fullmatch(r"[0-9a-f]{64}", metadata_lines[1]):
                    completed_content_hashes.add(metadata_lines[1])
            except OSError:
                pass
            if path.name.startswith("model-") and path.name.endswith(".safetensors"):
                complete_shards += 1

    # New huggingface_hub releases can leave several UUID-suffixed partials
    # for one blob after a process restart. Count only the largest candidate
    # for each content hash so telemetry never double-counts duplicate bytes.
    partial_sizes: dict[str, int] = {}
    if cache_dir.exists():
        for path in cache_dir.rglob("*.incomplete"):
            try:
                match = re.search(r"\.([0-9a-f]{64})(?:\.[0-9a-f]{8})?\.incomplete$", path.name)
                if match and match.group(1) in completed_content_hashes:
                    continue
                key = re.sub(r"\.[0-9a-f]{8}\.incomplete$", ".incomplete", path.name)
                partial_sizes[key] = max(partial_sizes.get(key, 0), path.stat().st_size)
            except OSError:
                pass
    partial_bytes = sum(partial_sizes.values())
    partial_files = len(partial_sizes)
    effective = min(EXPECTED_TOTAL_BYTES, final_bytes + partial_bytes)
    return {
        "effective_bytes": effective,
        "final_bytes": final_bytes,
        "partial_bytes": partial_bytes,
        "partial_files": partial_files,
        "complete_files": complete_files,
        "complete_shards": complete_shards,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=16 * 1024**2) as handle:
        while block := handle.read(16 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def verify(manifest: dict, hash_weights: bool = True) -> tuple[bool, str]:
    missing = []
    wrong_size = []
    for item in manifest["files"]:
        path = DOWNLOAD_DIR / item["path"]
        try:
            size = path.stat().st_size
        except OSError:
            missing.append(item["path"])
            continue
        if size != item["size"]:
            wrong_size.append(f"{item['path']}:{size}!={item['size']}")
    if missing or wrong_size:
        return False, f"missing={len(missing)}, wrong_size={len(wrong_size)}"
    if hash_weights:
        append_log(f"SHA256 verification started for {EXPECTED_SHARDS} weight shards")
        for index, item in enumerate((x for x in manifest["files"] if x.get("sha256")), 1):
            actual = sha256_file(DOWNLOAD_DIR / item["path"])
            if actual != item["sha256"]:
                return False, f"sha256 mismatch: {item['path']}"
            if index % 8 == 0 or index == EXPECTED_SHARDS:
                append_log(f"SHA256 verified {index}/{EXPECTED_SHARDS} shards")
    return True, f"{EXPECTED_FILES} files, {EXPECTED_SHARDS} shards, {EXPECTED_TOTAL_BYTES} bytes"


class Supervisor:
    def __init__(self) -> None:
        self.stop = False
        self.child: subprocess.Popen | None = None
        self.samples: deque[tuple[float, int]] = deque()
        self.started = now()
        self.attempt = 0
        self.restarts = 0

    def signal(self, signum: int, _frame: object) -> None:
        append_log(f"received signal {signum}; preserving resumable state")
        self.stop = True
        if self.child and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def save(self, progress: dict, phase: str, last_error: str | None = None) -> None:
        stamp = time.monotonic()
        self.samples.append((stamp, progress["effective_bytes"]))
        while len(self.samples) > 2 and self.samples[0][0] < stamp - 900:
            self.samples.popleft()
        rate = 0.0
        if len(self.samples) >= 2:
            elapsed = self.samples[-1][0] - self.samples[0][0]
            rate = max(0.0, (self.samples[-1][1] - self.samples[0][1]) / elapsed) if elapsed else 0.0
        eta = (EXPECTED_TOTAL_BYTES - progress["effective_bytes"]) / rate if rate > 0 else None
        state = {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "phase": phase,
            "pid": os.getpid(),
            "child_pid": self.child.pid if self.child and self.child.poll() is None else None,
            "started_at": self.started,
            "updated_at": now(),
            "attempt": self.attempt,
            "restart_count": self.restarts,
            "rate_bps": rate,
            "eta_seconds": eta,
            "last_error": last_error,
            "total_bytes": EXPECTED_TOTAL_BYTES,
            **progress,
        }
        atomic_json(STATE, state)

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self.signal)
        signal.signal(signal.SIGINT, self.signal)
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        manifest = load_manifest(refresh=True)
        progress = scan(manifest)
        remaining = EXPECTED_TOTAL_BYTES - progress["final_bytes"]
        free = shutil.disk_usage(DOWNLOAD_DIR).free
        if free < remaining + DISK_RESERVE:
            raise RuntimeError(f"insufficient disk: free={human(free)}, need={human(remaining + DISK_RESERVE)}")
        append_log(
            f"preflight PASS model={MODEL_ID}@{REVISION} files={EXPECTED_FILES} "
            f"bytes={EXPECTED_TOTAL_BYTES} free={human(free)} workers={WORKERS} xet=disabled"
        )
        last_bytes = progress["effective_bytes"]
        last_growth = time.monotonic()

        env = os.environ.copy()
        env.update({
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "HF_HUB_DOWNLOAD_TIMEOUT": "1200",
            "HF_HUB_ETAG_TIMEOUT": "120",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        })
        command = [
            str(HF_PYTHON), str(RESUMABLE_DOWNLOADER), MODEL_ID,
            "--revision", REVISION,
            "--local-dir", str(DOWNLOAD_DIR),
            "--max-workers", str(WORKERS),
        ]

        while not self.stop:
            progress = scan(manifest)
            if progress["complete_files"] == EXPECTED_FILES:
                self.save(progress, "verifying")
                ok, detail = verify(manifest, hash_weights=True)
                if ok:
                    self.save(scan(manifest), "completed")
                    append_log(f"COMPLETE and verified: {detail}")
                    return 0
                append_log(f"verification failed; resuming download: {detail}")

            self.attempt += 1
            append_log(f"attempt {self.attempt} starting: {' '.join(command)}")
            low_rate_windows = 0
            child_started = time.monotonic()
            low_rate_grace = LOW_RATE_GRACE_SECONDS * min(4, 2 ** self.restarts)
            sample_started = child_started
            sample_bytes = progress["effective_bytes"]
            with CLI_LOG.open("ab", buffering=0) as output:
                output.write(f"\n===== attempt {self.attempt} {now()} =====\n".encode())
                self.child = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                    close_fds=True,
                )
                self.save(scan(manifest), "downloading")
                while self.child.poll() is None and not self.stop:
                    time.sleep(POLL_SECONDS)
                    progress = scan(manifest)
                    sample_now = time.monotonic()
                    sample_elapsed = sample_now - sample_started
                    sample_rate = (
                        max(0, progress["effective_bytes"] - sample_bytes) / sample_elapsed
                        if sample_elapsed > 0
                        else 0.0
                    )
                    sample_started = sample_now
                    sample_bytes = progress["effective_bytes"]
                    if progress["effective_bytes"] > last_bytes:
                        last_bytes = progress["effective_bytes"]
                        last_growth = time.monotonic()
                    self.save(progress, "downloading")
                    if time.monotonic() - last_growth >= STALL_SECONDS:
                        append_log(f"no byte growth for {STALL_SECONDS}s; restarting hf child PID={self.child.pid}")
                        os.killpg(self.child.pid, signal.SIGTERM)
                        last_growth = time.monotonic()
                        break
                    if sample_now - child_started >= low_rate_grace:
                        if sample_rate < LOW_RATE_BPS:
                            low_rate_windows += 1
                        else:
                            low_rate_windows = 0
                        if low_rate_windows >= LOW_RATE_WINDOWS:
                            append_log(
                                f"throughput below {human(LOW_RATE_BPS)}/s for three minutes; "
                                f"last sample={human(sample_rate)}/s; recycling child PID={self.child.pid}"
                            )
                            os.killpg(self.child.pid, signal.SIGTERM)
                            break
                if self.stop and self.child.poll() is None:
                    os.killpg(self.child.pid, signal.SIGTERM)
                try:
                    rc = self.child.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(self.child.pid, signal.SIGKILL)
                    rc = self.child.wait()
            self.child = None
            progress = scan(manifest)
            if self.stop:
                self.save(progress, "stopped")
                return 0
            self.restarts += 1
            error = f"hf exited rc={rc}; retrying"
            self.save(progress, "retry_wait", error)
            append_log(error)
            time.sleep(min(300, 10 * 2 ** min(self.restarts, 5)))
        return 0


def run_locked() -> int:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("w", encoding="ascii") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another supervisor is already running")
            return 0
        PID_FILE.write_text(f"{os.getpid()}\n", encoding="ascii")
        try:
            return Supervisor().run()
        except Exception as exc:
            append_log(f"FATAL: {type(exc).__name__}: {exc}")
            state = read_json(STATE)
            state.update({"phase": "failed", "updated_at": now(), "last_error": str(exc)})
            atomic_json(STATE, state)
            return 1
        finally:
            if read_pid() == os.getpid():
                PID_FILE.unlink(missing_ok=True)


def command_start() -> int:
    pid = read_pid()
    if pid and process_alive(pid):
        print(f"already running PID={pid}")
        return 0
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with DAEMON_LOG.open("ab", buffering=0) as output:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "run"],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            start_new_session=True,
            close_fds=True,
        )
    for _ in range(40):
        time.sleep(0.5)
        pid = read_pid()
        if pid and process_alive(pid):
            print(f"started PID={pid}")
            return 0
        if proc.poll() is not None:
            print(f"start failed rc={proc.returncode}; see {DAEMON_LOG}")
            return proc.returncode or 1
    print(f"start timed out; see {DAEMON_LOG}")
    return 1


def command_status(json_output: bool = False) -> int:
    state = read_json(STATE)
    pid = read_pid()
    state["process_alive"] = bool(pid and process_alive(pid))
    # A supervisor started before a telemetry fix can leave stale partial
    # counters in its final state. Once it has stopped, report the filesystem
    # as the authority without changing the recorded completion result.
    if state and not state["process_alive"]:
        try:
            state.update(scan(load_manifest()))
        except Exception:
            pass
    if json_output:
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    if not state:
        print("no state yet")
        return 1
    done = int(state.get("effective_bytes", 0))
    total = int(state.get("total_bytes", EXPECTED_TOTAL_BYTES))
    rate = float(state.get("rate_bps", 0) or 0)
    pct = done / total * 100 if total else 0
    print(f"process    : {'RUNNING' if state['process_alive'] else 'STOPPED'} PID={pid or '--'}")
    print(f"phase      : {state.get('phase', '--')} attempt={state.get('attempt', 0)} restarts={state.get('restart_count', 0)}")
    print(f"revision   : {state.get('revision', '--')}")
    print(f"progress   : {pct:.2f}% {human(done)} / {human(total)}")
    print(f"files      : {state.get('complete_files', 0)}/{EXPECTED_FILES}; shards {state.get('complete_shards', 0)}/{EXPECTED_SHARDS}")
    print(f"partial    : {state.get('partial_files', 0)} files, {human(state.get('partial_bytes', 0))}")
    print(f"rate / ETA : {human(rate)}/s; {duration(state.get('eta_seconds'))}")
    print(f"updated    : {state.get('updated_at', '--')}")
    print(f"last error : {state.get('last_error') or '--'}")
    return 0


def command_stop() -> int:
    pid = read_pid()
    if not pid or not process_alive(pid):
        print("not running")
        return 0
    os.kill(pid, signal.SIGTERM)
    print(f"SIGTERM sent to PID={pid}; resumable data retained")
    return 0


def command_verify() -> int:
    manifest = load_manifest()
    ok, detail = verify(manifest, hash_weights=True)
    print(("PASS: " if ok else "FAILED: ") + detail)
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("start")
    sub.add_parser("run")
    status = sub.add_parser("status")
    status.add_argument("--json", action="store_true")
    sub.add_parser("stop")
    sub.add_parser("verify")
    args = parser.parse_args()
    if args.command == "start":
        return command_start()
    if args.command == "run":
        return run_locked()
    if args.command == "status":
        return command_status(args.json)
    if args.command == "stop":
        return command_stop()
    return command_verify()


if __name__ == "__main__":
    raise SystemExit(main())
