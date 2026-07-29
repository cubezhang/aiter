"""Numerical references for the dispatch->gemm1 fusion (Phase 1a).

These mirror the FlyDSL MX-fp8 quant math in
``aiter/ops/flydsl/kernels/quant_utils.py`` (``emit_mx_e8m0_scale`` RoundUp +
``v_cvt_scalef32_pk8_fp8_bf16``) so tests can check dispatch-side per-token
quant without a GPU kernel of their own.
"""

import torch


def mx_e8m0_roundup_fp8(amax_f32: torch.Tensor) -> torch.Tensor:
    """E8M0 biased exponent (uint8) for FP8_E4M3, RoundUp mode.

    Mirrors ``emit_mx_e8m0_scale(mode=RoundUp, dtype=FP8_E4M3)``:
    working = amax * (1/448.0); e8m0 = biased_exp(working) + (mantissa!=0);
    clamped to [0, 255].
    """
    inv_max_pos = torch.tensor(1.0 / 448.0, dtype=torch.float32)
    working = (amax_f32.to(torch.float32) * inv_max_pos).contiguous()
    wi = working.view(torch.int32)
    mant = wi & 0x7FFFFF
    biased = (wi >> 23) & 0xFF
    e8m0 = biased + (mant != 0).to(torch.int32)
    e8m0 = torch.clamp(e8m0, 0, 0xFF)
    return e8m0.to(torch.uint8)


def per_token_mx_fp8(hidden_bf16: torch.Tensor):
    """Per-token per-1x32-block MX-fp8 quant of ``hidden_bf16`` [T, H].

    Returns (payload_uint8 [T, H], e8m0_uint8 [T, H//32]) in arrival order,
    NOT preshuffled — matches the dispatch ``disp_out_q`` / ``disp_out_qscale``
    layout produced in Task 2.
    """
    assert hidden_bf16.dtype == torch.bfloat16
    T, H = hidden_bf16.shape
    assert H % 32 == 0
    x = hidden_bf16.to(torch.float32).reshape(T, H // 32, 32)
    amax = x.abs().amax(dim=-1)  # [T, H//32]
    e8m0 = mx_e8m0_roundup_fp8(amax)  # uint8 [T, H//32]
    scale = torch.exp2((e8m0.to(torch.float32) - 127.0)).unsqueeze(-1)  # 2^(e8m0-127)
    # HW divides the (bf16) input by the block scale, then RNE-packs to fp8 e4m3.
    payload = (x / scale).to(torch.float8_e4m3fn).view(torch.uint8).reshape(T, H)
    return payload.contiguous(), e8m0.reshape(T, H // 32).contiguous()
