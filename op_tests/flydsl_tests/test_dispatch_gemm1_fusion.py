import pytest
import torch

from aiter.ops.flydsl.dispatch_combine_v2.dispatch_combine_op import (
    EpDispatchCombineConfig,
    _fused_q_regions,
)


def _make_cfg(**ov):
    base = dict(
        rank=0,
        world_size=2,
        hidden_dim=512,
        max_num_inp_token_per_rank=128,
        num_experts_per_rank=4,
        num_experts_per_token=2,
        data_type=torch.bfloat16,
        # Pin geometry so __post_init__ skips the tuning_configs lookup.
        dispatch_block_num=64,
        combine_block_num=80,
        dispatch_warp_num_per_block=16,
        combine_warp_num_per_block=4,
    )
    base.update(ov)
    return EpDispatchCombineConfig(**base)


def test_fuse_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("AITER_EP_FUSE_DISPATCH_GEMM1", raising=False)
    assert _make_cfg().fuse_dispatch_gemm1 is False


def test_fuse_flag_on(monkeypatch):
    monkeypatch.setenv("AITER_EP_FUSE_DISPATCH_GEMM1", "1")
    assert _make_cfg().fuse_dispatch_gemm1 is True


def test_q_regions_present_when_on(monkeypatch):
    monkeypatch.setenv("AITER_EP_FUSE_DISPATCH_GEMM1", "1")
    cfg = _make_cfg()
    regs = dict(_fused_q_regions(cfg))
    assert regs["disp_out_q"] == cfg.effective_max_recv * cfg.hidden_dim
    assert regs["disp_out_qscale"] == cfg.effective_max_recv * (cfg.hidden_dim // 32)


def test_q_regions_absent_when_off(monkeypatch):
    monkeypatch.delenv("AITER_EP_FUSE_DISPATCH_GEMM1", raising=False)
    assert _fused_q_regions(_make_cfg()) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs gfx1250")
def test_fp8_gather_preshuffle_matches_bf16():
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

    ref_p, ref_s = flydsl_moe_fused_quant_preshuffle(
        hidden.reshape(1, T, H), 1, contiguous_m, wmma_rep=wmma_rep,
        quant_mode="fp8", topids_to_rows=t2r, source_topk=topk,
    )
    # byte-exact per-token fp8 (wmma_rep=1 => no preshuffle) as the fp8 input.
    p_tok, s_tok = flydsl_moe_fused_quant_preshuffle(
        hidden.reshape(1, T, H), 1, T, wmma_rep=1, quant_mode="fp8",
    )
    fp8_p, fp8_s = flydsl_moe_fused_quant_preshuffle(
        None, 1, contiguous_m, wmma_rep=wmma_rep, quant_mode="fp8",
        topids_to_rows=t2r, source_topk=topk,
        in_fp8_payload=p_tok.reshape(T, H), in_fp8_scale=s_tok.reshape(T, H // 32),
    )
    torch.testing.assert_close(fp8_p.cpu(), ref_p.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(fp8_s.cpu(), ref_s.cpu(), atol=0, rtol=0)
