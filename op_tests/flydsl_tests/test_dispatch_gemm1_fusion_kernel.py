"""Single-rank (world_size=1) numerical gate for Task 2.

Runs the fused dispatch on one GPU (no torchrun) and checks that the per-token
fp8 payload / e8m0 scale written into disp_out_q/disp_out_qscale match a
per-token MX-fp8 quant of the received bf16 tokens (disp_out).
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs gfx1250")


def _build_op(world, rank, hidden, ct, epr, topk):
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
        data_type=torch.bfloat16,
    )
    op = EpDispatchCombineOp(cfg, comm)
    comm.barrier()
    return comm, op


def test_dispatch_per_token_quant_matches_ref(monkeypatch):
    monkeypatch.setenv("AITER_EP_FUSE_DISPATCH_GEMM1", "1")
    from op_tests.flydsl_tests._fusion_ref import per_token_mx_fp8

    torch.manual_seed(0)
    world, rank = 1, 0
    hidden, ct, epr, topk = 512, 64, 4, 2
    E = epr * world

    comm, op = _build_op(world, rank, hidden, ct, epr, topk)
    try:
        dev = torch.device("cuda", 0)
        x = (torch.randn(ct, hidden, device=dev) * 3.0).to(torch.bfloat16)
        # topk distinct expert ids per token in [0, E).
        ids = torch.stack(
            [torch.randperm(E, device=dev)[:topk] for _ in range(ct)]
        ).to(torch.int32)
        wts = torch.rand(ct, topk, device=dev, dtype=torch.float32)

        out = op.dispatch(x, wts, None, ids, return_routing=True)
        tr = out[4]
        total_recv = int(tr.reshape(-1)[0].item()) if torch.is_tensor(tr) else int(tr)

        disp_out = op.recv_tokens()[:total_recv].to(torch.bfloat16)
        payload, scale = op.disp_out_q_view()
        ref_payload, ref_e8m0 = per_token_mx_fp8(disp_out)

        # e8m0 is an integer formula -> must match exactly.
        torch.testing.assert_close(
            scale[:total_recv].cpu(), ref_e8m0.cpu(), atol=0, rtol=0
        )
        # fp8 payload: HW pk8 vs torch cast may differ by <=1 ULP at RNE edges.
        a = payload[:total_recv].view(torch.float8_e4m3fn).to(torch.float32).cpu()
        b = ref_payload.view(torch.float8_e4m3fn).to(torch.float32).cpu()
        mism = (a != b).float().mean().item()
        assert mism < 0.01, f"payload mismatch fraction {mism:.4f}"
    finally:
        op.close()
