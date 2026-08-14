import csv
import math
import time

import torch
import triton

from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse


def make_inputs(t, h, d, kv_len, seed=20260804):
    torch.manual_seed(seed + t * 10000 + kv_len)
    q = torch.randn((t, h, d), device="cuda", dtype=torch.bfloat16) * 0.5
    kv = torch.randn((t * kv_len, d), device="cuda", dtype=torch.bfloat16) * 0.5
    indices = torch.arange(t * kv_len, device="cuda", dtype=torch.int32)
    indptr = torch.arange(0, (t + 1) * kv_len, kv_len, device="cuda", dtype=torch.int32)
    sink = torch.randn((h,), device="cuda", dtype=torch.float32) * 0.1
    return q, kv, indices, indptr, sink, d**-0.5


def torch_reference(q, kv, indices, indptr, sink, scale):
    # Fixed-length contiguous CSR used by this benchmark.
    t, h, d = q.shape
    kv_len = int((indptr[1] - indptr[0]).item())
    gathered = kv[indices.long()].view(t, kv_len, d).float()
    scores = torch.einsum("thd,tkd->thk", q.float(), gathered) * scale
    combined = torch.cat((scores, sink.view(1, h, 1).expand(t, h, 1)), dim=-1)
    weights = combined.softmax(dim=-1)[..., :-1]
    return torch.einsum("thk,tkd->thd", weights, gathered).to(q.dtype)


def bench(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / 20)
    samples.sort()
    return samples[2], samples[0], samples[-1]


def next_pow2(x):
    return 1 << (max(1, int(x)) - 1).bit_length()


def default_splits(t, kv_len):
    raw = min(max(1, 256 // t), math.ceil(t * kv_len / 16))
    return next_pow2(raw)


def candidate_splits(t, kv_len):
    if kv_len <= 320:
        return 1
    if kv_len <= 512:
        return 4
    fill = next_pow2(math.ceil(768 / t))
    work = next_pow2(math.ceil(kv_len / 64))
    return min(fill, work)


def main():
    writer = csv.DictWriter(
        open("/work/experiments/e128_pa_decode_candidate_heuristic.csv", "w", newline=""),
        fieldnames=[
            "T", "H", "D", "kv_len", "variant", "splits", "allclose", "max_abs",
            "mean_abs", "median_us", "min_us", "max_us",
        ],
    )
    writer.writeheader()
    for t in (1, 2, 4, 8, 16, 24, 32, 48, 64, 72, 96, 128):
        for kv_len in (136, 320, 388, 512, 768, 1024, 2048, 4096, 8192):
            args = make_inputs(t, 16, 512, kv_len)
            ref = torch_reference(*args)
            variants = (
                ("default", default_splits(t, kv_len)),
                ("candidate", candidate_splits(t, kv_len)),
            )
            for variant, splits in variants:
                fn = lambda s=splits: pa_decode_sparse(
                    *args, kv_splits=s, has_invalid=False
                )
                out = fn()
                delta = (out.float() - ref.float()).abs()
                ok = torch.allclose(out, ref, atol=5e-3, rtol=5e-3)
                med, lo, hi = bench(fn)
                row = {
                    "T": t, "H": 16, "D": 512, "kv_len": kv_len,
                    "variant": variant, "splits": splits, "allclose": int(ok),
                    "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                    "median_us": med, "min_us": lo, "max_us": hi,
                }
                writer.writerow(row)
                writer._dict_writer__writer if False else None
                print(row, flush=True)


if __name__ == "__main__":
    main()
