        _, _, y = aiter.mhc_pre(
            x, hc_fn, hc_scale, hc_base, self.norm_eps, self.hc_eps, sinkhorn_repeat=0
        )
        return y

    def forward(
        self,
        x: torch.Tensor,  # [num_tokens, hc, dim]
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: nn.Module,
    ) -> torch.Tensor:  # [bs, vocab]
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)  # [num_tokens, dim]
        # get_logits handles the per-rank vocab shard + all-gather internally.
        return self.get_logits(norm(x))  # [bs, vocab]


def _run_moe(moe: "MoE", x: torch.Tensor) -> torch.Tensor:
    """Replicate MoE.forward's dual/single dispatch (we are already inside an
    opaque op, so call the underlying methods directly rather than re-entering
    the maybe_dual_stream_forward custom op)."""
    threshold = envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD
    num_tokens = x.shape[0]
    if moe._use_dual_stream and 0 < num_tokens <= threshold:
        return moe.dual_stream_moe_forward(x)
    return moe.single_stream_moe_forward(x)


def moe_pcp_merge_forward(
    hidden_states: torch.Tensor,  # [n_local, dim]
    layer_name: str,
) -> torch.Tensor:  # [n_local, dim]  (shape-preserving)
    moe = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    # Gate is read EAGERLY here (op is opaque to Dynamo), so it is NEVER baked —
    # keeping is_dummy_run in the gate is correct and NECESSARY: it keeps this op
    # consistent with the out-of-graph split gate `_pcp_active()` (also has
    # is_dummy). At warmup (is_dummy=True) neither splits nor merges, so x stays
    # full and the hash-MoE input_ids (also un-split at warmup) match. Removing
    # is_dummy here would merge a never-split warmup batch -> input_ids/hidden
    # length mismatch -> crash. (The in-graph gate needed is_dummy *removed* to
    # bake True into code0; the opaque op does NOT — opacity is the fix.)
    do_prefill = _moe_pcp_merge_active()
    do_decode = _moe_pcp_merge_decode_active()
    ws = get_pcp_world_size()
    x = hidden_states
    if do_prefill:
        from atom.utils.tbo.ubatching import (
            tbo_active as _tbo_active,
            tbo_yield_and_switch_from_compute_to_comm,
            tbo_switch_to_compute_sync,
        )

        _tbo = _tbo_active()
