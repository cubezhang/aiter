    except Exception:
        gfx = "gfx942"
    if gfx in ("gfx950", "gfx1250"):
        return "torch.float8_e4m3fn", "QuantType.per_1x128"
    else:
        return "torch.float8_e4m3fnuz", "QuantType.per_Token"


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _cleanup_stale_lock_files():
    """Remove stale FileBaton lock files left by killed subprocesses."""
    build_dir = os.path.join(AITER_ROOT, "aiter", "jit", "build")
    if not os.path.isdir(build_dir):
        return
    lock_patterns = [
        os.path.join(build_dir, "lock_*"),
        os.path.join(build_dir, "*", "build", "lock"),
        os.path.join(build_dir, "lock_3rdparty_*"),
    ]
    for pattern in lock_patterns:
        for lock_file in glob.glob(pattern):
            try:
                os.remove(lock_file)
                print(f"Cleaned up stale lock file: {lock_file}", flush=True)
            except OSError:
                pass


def _run_tuner(script, untuned, tuned, extra_args=None, timeout=300, mp=1):
    _cleanup_stale_lock_files()
    cmd = [
        sys.executable,
        os.path.join(AITER_ROOT, script),
        "-i",
        untuned,
        "-o",
        tuned,
        "--warmup",
        "2",
        "--iters",
        "5",
    ]
    if mp is not None:
        cmd.extend(["--mp", str(mp)])
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    script_dir = os.path.dirname(os.path.join(AITER_ROOT, script))
    env["PYTHONPATH"] = script_dir + ":" + env.get("PYTHONPATH", "")
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=AITER_ROOT,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        _cleanup_stale_lock_files()
        raise AssertionError(
            f"Tuner timed out after {timeout}s (likely GPU hang or infinite loop)\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout (last 500): {(e.stdout or b'')[-500:]}\n"
            f"  stderr (last 500): {(e.stderr or b'')[-500:]}"
        ) from None


@unittest.skipUnless(_gpu_available(), "No GPU available")
class TestTunePipeline(unittest.TestCase):
    """Smoke test: run each tuner on 1 small shape, verify CSV output."""

    @classmethod
    def setUpClass(cls):
        fp8, qtype = _get_platform_dtypes()
        cls.TUNERS = {
            "a8w8": {
                "script": "csrc/ck_gemm_a8w8/gemm_a8w8_tune.py",
                "header": ["M", "N", "K", "q_dtype_w"],
                "shapes": [
                    (1, 1024, 512, "torch.int8"),
                    (1, 1024, 512, fp8),
                ],
                "shapes_mp1": [
                    (1, 1024, 512, "torch.int8"),
                ],
                "keys": ["cu_num", "M", "N", "K", "q_dtype_w"],
            },
            "a8w8_blockscale": {
                "script": "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py",
                "header": ["M", "N", "K"],
                # Use the B-preshuffle ASM path to avoid expensive CK JIT builds
                # in the smoke pipeline.
                "shapes": [(16, 1536, 7168)],
                "shapes_mp1": [(16, 1536, 7168)],
                "keys": ["cu_num", "M", "N", "K"],
                "extra_args": ["--libtype", "asm", "--preshuffle", "--batch", "1"],
            },
            "a8w8_bpreshuffle": {
                "script": "csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py",
                "header": ["M", "N", "K", "q_dtype_w"],
                "shapes": [
                    (1, 1024, 512, "torch.int8"),
                    (1, 1024, 512, fp8),
                ],
                "shapes_mp1": [
                    (1, 1024, 512, "torch.int8"),
                ],
                "keys": ["cu_num", "M", "N", "K", "q_dtype_w"],
                "timeout": 900,
                "timeout_mp1": 1200,
            },
            "batched_a8w8": {
                "script": "csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py",
                "header": ["B", "M", "N", "K"],
                "shapes": [(2, 1, 512, 256)],
                "shapes_mp1": [(2, 1, 512, 256)],
                "keys": ["cu_num", "B", "M", "N", "K"],
            },
            "batched_bf16": {
                "script": "csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py",
                "header": ["B", "M", "N", "K"],
                "shapes": [(2, 1, 512, 256)],
                "shapes_mp1": [(2, 1, 512, 256)],
