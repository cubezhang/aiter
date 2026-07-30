# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""General utilities shared across all FlyDSL kernel families."""

import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
from functools import lru_cache
from pathlib import Path

import torch

_FALLBACK_MAX_LDS_BYTES = 65536


def do_bench_collective(fn, warmup=5, rep=25, quantiles=None):
    """Benchmark on all ranks and share the maximum latency."""
    from flydsl.autotune import do_bench

    elapsed = do_bench(fn, warmup=warmup, rep=rep, quantiles=quantiles)
    if not torch.distributed.is_initialized():
        return elapsed
    if quantiles:
        raise ValueError("collective autotune does not support quantile results")
    value = torch.tensor(float(elapsed), dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.reduce(value, dst=0, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.broadcast(value, src=0)
    return float(value.item())


def _save_autotune_cache(tuner):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    path = tuner._cache_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = json.loads(path.read_text()) if path.exists() else {}
        except (OSError, ValueError):
            data = {}
        for key, config in tuner.cache.items():
            data[json.dumps(list(key))] = config.to_dict()
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)


class _CollectiveAutotuner:
    """Compatibility wrapper for FlyDSL 0.2.4."""

    def __init__(self, tuner, *, default=None, artifact_name=None):
        self._tuner = tuner
        self._synced_keys = set()
        self._default = default
        self._artifact_name = artifact_name
        self.last_config = None
        tuner._save_disk_cache = lambda: _save_autotune_cache(tuner)

    def __getattr__(self, name):
        return getattr(self._tuner, name)

    @staticmethod
    def _broadcast(value):
        payload = [value if torch.distributed.get_rank() == 0 else None]
        torch.distributed.broadcast_object_list(
            payload, src=0, device=torch.device("cuda", torch.cuda.current_device())
        )
        return payload[0]

    def _artifact_ref(self, args, kwargs):
        config_dir = os.environ.get("FLYDSL_AUTOTUNE_CONFIG_DIR")
        if not self._artifact_name or not config_dir:
            return None
        source = self._tuner.fn.func if hasattr(self._tuner.fn, "func") else self._tuner.fn
        bound = inspect.signature(source).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        key = {}
        for name in self._tuner.key:
            value = bound.arguments[name]
            if hasattr(value, "shape"):
                value = tuple(value.shape)
            elif hasattr(value, "dtype"):
                value = str(value.dtype)
            key[name] = value
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        from flydsl.runtime.device import get_rocm_arch

        identity = {
            "name": self._artifact_name,
            "key": key,
            "device": {
                "name": str(props.name),
                "arch": str(get_rocm_arch()),
                "compute_units": int(props.multi_processor_count),
            },
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        config_dir = Path(config_dir).expanduser().resolve()
        bundle = config_dir / f"{self._artifact_name}.json"
        legacy = config_dir / f"{self._artifact_name}-{digest}.json"
        return bundle, legacy, digest, identity

    @staticmethod
    def _load_artifact(ref):
        if ref is None:
            return None
        path, legacy, digest, identity = ref
        body = None
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") != 1 or data.get("name") != identity["name"]:
                return None
            entry = data.get("entries", {}).get(digest)
            if entry is not None and entry.get("identity") == identity:
                body = entry.get("config")
        if body is None and legacy.is_file():
            data = json.loads(legacy.read_text(encoding="utf-8"))
            if data.get("version") == 1 and data.get("identity") == identity:
                body = data.get("config")
        if not isinstance(body, dict):
            return None
        from flydsl.autotune import Config

        return Config.from_dict(body)

    @staticmethod
    def _emit_artifact(ref, config):
        if ref is None:
            return
        path, _legacy, digest, identity = ref
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(path.parent, os.O_RDONLY)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    payload.get("version") != 1
                    or payload.get("name") != identity["name"]
                    or not isinstance(payload.get("entries"), dict)
                ):
                    raise ValueError(f"cannot update invalid artifact bundle {path}")
            else:
                payload = {"version": 1, "name": identity["name"], "entries": {}}
            payload["entries"][digest] = {"identity": identity, "config": config.to_dict()}
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def __call__(self, *args, **kwargs):
        key = self._tuner._make_key(args, kwargs)
        distributed = torch.distributed.is_initialized()
        enabled = os.environ.get("FLYDSL_AUTOTUNE", "").strip().lower() in ("1", "true", "yes", "on")
        target = os.environ.get("FLYDSL_AUTOTUNE_TARGET", "").strip()
        force = enabled and (not target or target == self._artifact_name)
        rank0 = not distributed or torch.distributed.get_rank() == 0
        if force:
            self._tuner.cache.pop(key, None)
        if distributed and not force and key in self._synced_keys:
            result = self._tuner(*args, **kwargs)
            self.last_config = self._tuner.cache[key]
            return result
        if distributed and key not in self._synced_keys:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("collective autotune must be warmed up before CUDA Graph capture")
            cached = self._tuner.cache.get(key) if rank0 and not force else None
            cached_dict = self._broadcast(cached.to_dict() if cached is not None else None)
            if cached_dict is None:
                self._tuner.cache.pop(key, None)
            else:
                from flydsl.autotune import Config

                self._tuner.cache[key] = Config.from_dict(cached_dict)
            self._synced_keys.add(key)
        if not force and key not in self._tuner.cache:
            ref = self._artifact_ref(args, kwargs)
            artifact = self._load_artifact(ref) if rank0 else None
            artifact_dict = artifact.to_dict() if artifact is not None else None
            if distributed:
                artifact_dict = self._broadcast(artifact_dict)
            if artifact_dict is not None:
                from flydsl.autotune import Config

                self._tuner.cache[key] = Config.from_dict(artifact_dict)
        if not force and key not in self._tuner.cache and self._default is not None:
            config = self._default(*args, **kwargs) if callable(self._default) and rank0 else self._default
            config_dict = config.to_dict() if rank0 else None
            if distributed:
                config_dict = self._broadcast(config_dict)
            from flydsl.autotune import Config

            self._tuner.cache[key] = Config.from_dict(config_dict)
        result = self._tuner(*args, **kwargs)
        if distributed:
            selected = self._tuner.cache[key].to_dict() if rank0 else None
            selected = self._broadcast(selected)
            from flydsl.autotune import Config

            self._tuner.cache[key] = Config.from_dict(selected)
        if force and rank0:
            self._emit_artifact(self._artifact_ref(args, kwargs), self._tuner.cache[key])
        self.last_config = self._tuner.cache[key]
        return result


def autotune_compat(*args, **kwargs):
    """Use FlyDSL autotune with compatibility for the released package."""
    import importlib

    module = importlib.import_module("flydsl.autotune")
    signature = inspect.signature(module.autotune)
    if kwargs.get("do_bench") is do_bench_collective:
        kwargs["do_bench"] = getattr(module, "do_bench_collective", do_bench_collective)
    default = kwargs.get("default")
    artifact_name = kwargs.get("artifact_name")
    if "default" not in signature.parameters:
        kwargs.pop("default", None)
    if "artifact_name" not in signature.parameters:
        kwargs.pop("artifact_name", None)
    decorate = module.autotune(*args, **kwargs)

    def decorator(fn):
        tuner = decorate(fn)
        if hasattr(tuner, "last_config"):
            return tuner
        return _CollectiveAutotuner(tuner, default=default, artifact_name=artifact_name)

    return decorator


def addressable_lds_bytes_for_gfx(gfx: str) -> int:
    g = (gfx or "").strip().lower().split(":")[0]
    if not g.startswith("gfx"):
        return _FALLBACK_MAX_LDS_BYTES
    if g.startswith("gfx950"):
        return 163840
    if g.startswith("gfx1250"):
        return 327680
    if g.startswith("gfx7") or g.startswith("gfx8"):
        return 32768
    return 65536


@lru_cache(maxsize=1)
def _default_cuda_device_index():
    try:
        return int(torch.cuda.current_device())
    except Exception:
        return None


@lru_cache(maxsize=None)
def _get_shared_memory_per_block_cached(device_index: int, fallback_gfx: str) -> int:
    try:
        props = torch.cuda.get_device_properties(device_index)
        shared_memory_per_block = int(getattr(props, "shared_memory_per_block", 0) or 0)
        if shared_memory_per_block > 0:
            return shared_memory_per_block
        return addressable_lds_bytes_for_gfx(
            getattr(props, "gcnArchName", fallback_gfx)
        )
    except Exception:
        return addressable_lds_bytes_for_gfx(fallback_gfx)


def get_shared_memory_per_block(device=None, fallback_gfx: str = "") -> int:
    """Return per-block shared memory/LDS limit for the active device."""
    if device is None:
        device = _default_cuda_device_index()
    elif isinstance(device, torch.device):
        if device.type != "cuda":
            device = None
        elif device.index is None:
            device = _default_cuda_device_index()
        else:
            device = int(device.index)
    else:
        try:
            device = int(device)
        except Exception:
            device = None

    if device is None:
        return addressable_lds_bytes_for_gfx(fallback_gfx)
    return _get_shared_memory_per_block_cached(device, fallback_gfx)


@lru_cache(maxsize=1)
def is_flydsl_available() -> bool:
    if importlib.util.find_spec("flydsl") is None:
        return False
    # flydsl only ships kernels for the architectures in its SMEM_CAPACITY_MAP.
    # On other archs (e.g. gfx1100 / RDNA3) importing the kernel modules crashes
    # during config registration, so report flydsl as unavailable there instead
    # of failing the import.
    from flydsl.runtime.device import get_rocm_arch
    from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP

    return get_rocm_arch() in SMEM_CAPACITY_MAP
