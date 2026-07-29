"""Validate the Python MX-fp8 reference against the trusted FlyDSL kernel.

For E=1 with identity row mapping, ``flydsl_moe_fused_quant_preshuffle``'s
payload rows are exactly per-token fp8, so we can validate ``per_token_mx_fp8``
(payload path) without writing a new kernel. This locks the quant formula used
by Task 2/3.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def test_per_token_fp8_payload_matches_kernel():
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_fused_quant_preshuffle
    from op_tests.flydsl_tests._fusion_ref import per_token_mx_fp8

    torch.manual_seed(0)
    wmma_rep = 4
    T, H = 64, 512  # T multiple of wmma_rep*16=64
    hidden = (torch.randn(T, H, device="cuda") * 3.0).to(torch.bfloat16)

    payload, scale = flydsl_moe_fused_quant_preshuffle(
        hidden.reshape(1, T, H), 1, T, wmma_rep=wmma_rep, quant_mode="fp8",
    )
    ref_payload, ref_e8m0 = per_token_mx_fp8(hidden)

    k_payload = payload.reshape(T, H).to("cuda")
    # Compare decoded fp8 values (exact byte match may differ by <=1 ULP at RNE
    # thresholds between torch's f32->fp8 cast and HW bf16 pk8).
    a = k_payload.view(torch.float8_e4m3fn).to(torch.float32)
    b = ref_payload.reshape(T, H).view(torch.float8_e4m3fn).to(torch.float32)
    mism = (a != b).float().mean().item()
    assert mism < 0.01, f"payload mismatch fraction {mism:.4f} too high"
