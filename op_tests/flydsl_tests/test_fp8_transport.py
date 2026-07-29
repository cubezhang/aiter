"""T-A.1: fp8 transport round-trip gate (1a-v2 / fp8 transport route).

Single-rank (world_size=1) in-process check that a dispatch built with
``data_type=fp8`` + ``scale_dim=hidden//32`` moves the per-token fp8 payload AND
the per-token e8m0 scale to the destination byte-for-byte, in arrival order.

This locks the fp8 transport semantics the 1a-v2 route builds on: quantize once
upstream, send only fp8+e8m0 (no bf16 on the wire), let GEMM1 consume recv_x +
out_scales directly. No quantization happens inside dispatch here -- dispatch is
a pure byte mover.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs gfx1250")


def _build_fp8_op(world, rank, hidden, ct, epr, topk):
    from mori.cco import Communicator
    from aiter.ops.flydsl.dispatch_combine_v2.dispatch_combine_op import (
        EpDispatchCombineConfig,
        EpDispatchCombineOp,
    )

    torch.cuda.set_device(0)
    uid = Communicator.get_unique_id()
    comm = Communicator.init(world, rank, uid)
    cfg = EpDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=hidden,
        max_num_inp_token_per_rank=ct,
        num_experts_per_rank=epr,
        num_experts_per_token=topk,
        # fp8 transport: the wire carries fp8 tokens + per-token e8m0 scales.
        data_type=torch.float8_e4m3fn,
        scale_dim=hidden // 32,
        scale_type_size=1,
    )
    op = EpDispatchCombineOp(cfg, comm)
    comm.barrier()
    return comm, op


def quantize_mxfp8_for_dispatch(x_bf16):
    """T-A.2 helper: per-token MX-fp8 quant of [T,H] bf16 -> (x_fp8[T,H] fp8,
    e8m0[T,H//32] uint8), the exact layout dispatch transports and GEMM1's
    fp8-gather consumes.

    Selection = ``flydsl_moe_fused_quant_preshuffle(..., wmma_rep=1)``: the SAME
    FlyDSL per-1x32 MX-fp8 quant the bf16 a1 path uses (wmma_rep=1 => plain
    per-token scale, no preshuffle). Chosen over ``dynamic_mxfp8_quant`` because
    it is byte-identical to the baseline quant, so the fp8-transport a1 output
    reproduces the bf16-transport baseline exactly (see byte-exact test below);
    dynamic_mxfp8_quant differs by an off-by-1 e8m0 rounding convention on ~1%
    of blocks, which would drift MEGA-CHECK logits off the baseline."""
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_fused_quant_preshuffle

    T, H = x_bf16.shape
    p, s = flydsl_moe_fused_quant_preshuffle(
        x_bf16.reshape(1, T, H), 1, T, wmma_rep=1, quant_mode="fp8",
    )
    p = p.reshape(T, H)
    s = s.reshape(T, H // 32)
    x_fp8 = (p if p.dtype != torch.uint8 else p.view(torch.uint8)).view(
        torch.float8_e4m3fn
    )
    e8m0 = s if s.dtype == torch.uint8 else s.view(torch.uint8)
    return x_fp8, e8m0


def test_dispatch_quant_helper_byte_exact_vs_bf16_a1():
    """T-A a1 equivalence: quantize upstream with the dispatch helper, then
    fp8-gather+preshuffle == the bf16 quant+gather+preshuffle a1, byte-for-byte.

    This is the property that keeps fp8 transport correctness identical to the
    bf16-transport baseline: GEMM1 consumes the same a1 bytes either way."""
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_fused_quant_preshuffle

    torch.manual_seed(0)
    dev = torch.device("cuda", 0)
    wmma_rep, topk = 4, 2
    T, H = 32, 512
    numel = T * topk
    contiguous_m = numel  # multiple of wmma_rep*16 = 64
    hidden = (torch.randn(T, H, device=dev) * 3.0).to(torch.bfloat16)
    # identity route: grouped row == route; source_row = route // topk.
    t2r = torch.arange(numel, dtype=torch.int32, device=dev)

    # baseline: bf16 -> quant + gather + preshuffle (the current a1 path).
    ref_p, ref_s = flydsl_moe_fused_quant_preshuffle(
        hidden.reshape(1, T, H), 1, contiguous_m, wmma_rep=wmma_rep,
        quant_mode="fp8", topids_to_rows=t2r, source_topk=topk,
    )
    # fp8 transport: quantize upstream (helper), transport bytes, fp8-gather.
    x_fp8, e8m0 = quantize_mxfp8_for_dispatch(hidden)
    fp8_p, fp8_s = flydsl_moe_fused_quant_preshuffle(
        None, 1, contiguous_m, wmma_rep=wmma_rep, quant_mode="fp8",
        topids_to_rows=t2r, source_topk=topk,
        in_fp8_payload=x_fp8.view(torch.uint8), in_fp8_scale=e8m0,
    )
    torch.testing.assert_close(fp8_p.cpu(), ref_p.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(fp8_s.cpu(), ref_s.cpu(), atol=0, rtol=0)


def test_fp8_transport_roundtrip_byte_exact():
    """Pre-quantized fp8 payload + e8m0 scale survive dispatch byte-for-byte."""
    torch.manual_seed(0)
    world, rank = 1, 0
    hidden, ct, epr, topk = 512, 64, 4, 2
    E = epr * world

    comm, op = _build_fp8_op(world, rank, hidden, ct, epr, topk)
    try:
        dev = torch.device("cuda", 0)
        x = (torch.randn(ct, hidden, device=dev) * 3.0).to(torch.bfloat16)
        # Quantize ONCE upstream (pre-dispatch) -> fp8 payload + e8m0 scale.
        x_fp8, e8m0 = quantize_mxfp8_for_dispatch(x)  # [ct,H] fp8, [ct,H//32] u8
        payload_u8 = x_fp8.view(torch.uint8)

        ids = torch.stack(
            [torch.randperm(E, device=dev)[:topk] for _ in range(ct)]
        ).to(torch.int32)
        wts = torch.rand(ct, topk, device=dev, dtype=torch.float32)

        recv_x, _rw, recv_scales, _ri, tr, handle = op.dispatch(
            x_fp8, wts, e8m0, ids, return_routing=True
        )
        total_recv = int(tr.reshape(-1)[0].item()) if torch.is_tensor(tr) else int(tr)
        # world=1: every token routed to the single local rank exactly once.
        assert total_recv == ct, f"expected {ct} recv tokens, got {total_recv}"

        # arrival-order reverse map: recv slot -> source token id.
        rev = handle.disp_tok_id_to_src_tok_id_local[:total_recv].long().cpu()

        # fp8 payload: recv_x is fp8 [cap, hidden]; compare raw bytes.
        recv_p = recv_x[:total_recv].view(torch.uint8).cpu()
        exp_p = payload_u8.cpu()[rev]
        torch.testing.assert_close(recv_p, exp_p, atol=0, rtol=0)

        # e8m0 scale: out_scales is packed i32 [cap, ceil(H//32/4)]; view as bytes.
        recv_s = recv_scales[:total_recv].view(torch.uint8)[:, : hidden // 32].cpu()
        exp_s = e8m0.cpu()[rev]
        torch.testing.assert_close(recv_s, exp_s, atol=0, rtol=0)
    finally:
        op.close()
