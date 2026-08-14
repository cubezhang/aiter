class MoE(nn.Module):
    """Mixture-of-Experts: top-k routed experts (FusedMoE) + 1 shared expert.

    PR3b: replaces the per-expert nn.Linear list with `FusedMoE` so 384 routed
    experts shard across TP/EP ranks and load FP4 weights via the existing
    `gemm_a4w4_quant` aiter kernel.

    Routing math (`sqrtsoftplus(scores) + bias` topk) is delegated to
    `FusedMoE.select_experts(scoring_func="sqrtsoftplus", e_score_correction_bias=...)`,
    which we extended in atom/model_ops/moe.py to add the V4 path.

    Hash routing for `layer_id < n_hash_layers` (first 3 V4 layers) is wired
    through FusedMoE via the `custom_routing_function` hook: hash layers load a
    `tid2eid` table (token-id -> expert-id) instead of `gate.bias`, and
    `select_experts` gives `custom_routing_function` precedence over the
    standard sqrtsoftplus path. Expert *selection* comes from `tid2eid[input_ids]`
    while expert *weights* still use sqrtsoftplus(gate_logits). Accuracy verified.
    """

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV4Args,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.prefix = prefix
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.is_hash_layer = layer_id < args.n_hash_layers
        self.routed_scaling_factor = args.route_scale
        self.swiglu_limit = args.swiglu_limit
        self.tp_size = get_tensor_model_parallel_world_size()
        self.alt_stream = alt_stream
        qc = args.quant_config

        self.gate = ReplicatedLinear(
            self.dim,
            self.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        # V4 hash-routed layers (layer_id < n_hash_layers) use tid2eid lookup,
        # not bias-corrected gate-logit routing — checkpoint has no
        # `gate.bias` for those layers. Only allocate the bias for
        # sqrtsoftplus layers to avoid 3 spurious unloaded-param warnings.
        if not self.is_hash_layer:
            self.gate.e_score_correction_bias = atom_parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32)
            )
        else:
            # tid2eid: per-token-id top-k expert lookup table (V4 first 3
            # layers use this in lieu of gate-logit routing).
            self.gate.tid2eid = atom_parameter(
                torch.empty(
                    args.vocab_size, args.n_activated_experts, dtype=torch.int32
                ),
            )
            # input_ids for hash routing is read from forward_context.context
            # (set by ModelRunner). torch.compile silently drops NNModule
            # attribute mutation across the compile boundary, so stashing on
            # `self.foo` from inside forward is a no-op at runtime.
        assert args.n_shared_experts == 1
        self._fuse_shared_into_routed = (
            is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
                qc,
                shared_expert_prefix=f"{prefix}.shared_experts",
                routed_expert_prefix=f"{prefix}.experts",
            )
        )
        moe_cfg = SimpleNamespace(
            routed_scaling_factor=self.routed_scaling_factor,
            n_shared_experts=(
                args.n_shared_experts if self._fuse_shared_into_routed else 0
            ),
        )
        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=self.n_activated_experts,
            hidden_size=self.dim,
            intermediate_size=args.moe_inter_dim,
            layer_id=self.layer_id,
            reduce_results=False,
            renormalize=True,
            quant_config=qc,
            use_grouped_topk=False,
            prefix=f"{prefix}.experts",
            scoring_func=args.score_func,  # "sqrtsoftplus"
            e_score_correction_bias=getattr(self.gate, "e_score_correction_bias", None),
            config=moe_cfg,
            shared_expert_prefix=f"{prefix}.shared_experts",
            # inter=3072/TP8=384 is a 128-multiple; pad to 128 (not the 256
            # default) to avoid padding the MoE intermediate up to 512.
            pad_align=128,
        )
        self.experts.swiglu_limit = args.swiglu_limit

        if not self._fuse_shared_into_routed:
            # self.experts.num_fused_shared_experts = 0
            self.shared_experts = Expert(
                args.dim,
                args.moe_inter_dim,
                swiglu_limit=args.swiglu_limit,
                quant_config=qc,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None
        if self.is_hash_layer:
            # Inject hash routing into FusedMoE.select_experts via the
            # custom_routing_function hook (added in atom/model_ops/moe.py).
            self.experts.custom_routing_function = self._hash_topk

        # Dual-stream: run shared_experts on `alt_stream` in parallel with
        # routed experts on the current stream. Mirrors V2's pattern. Only
        # active when shared_experts exist (not fused into routed) AND the
        # env threshold is positive AND we got an alt_stream from the model.
        # Per-call token count gating happens inside the custom op dispatcher
        # — prefill (large batch) skips dual-stream (overhead > benefit).
        self._use_dual_stream = (
            self.shared_experts is not None
            and self.alt_stream is not None
            and envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD > 0
        )
        # Register self in static_forward_context so the custom op dispatcher
        # can look us up by `layer_name` (= self.prefix). Needed by
        # maybe_dual_stream_forward (dual-stream) AND moe_pcp_merge_forward
        # — the latter requires registration regardless of
        # dual-stream, so register whenever either consumer is active.
        _pcp_merge_on = get_pcp_world_size() > 1 and bool(envs.ATOM_PCP_MOE_MERGE)
        if self._use_dual_stream or _pcp_merge_on:
            get_current_atom_config().compilation_config.static_forward_context[
                prefix
            ] = self

    def _hash_topk(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """V4 hash routing for first 3 layers.

        topk_ids = tid2eid[input_ids]  (no gate-based selection)
        topk_weights = sqrtsoftplus(router_logits) gathered at topk_ids
        Then renormalize so weights sum to 1 per token.
        """
        fwd_input_ids = get_forward_context().context.input_ids
        assert (
            fwd_input_ids is not None
        ), "forward_context.context.input_ids is None — caller must invoke DeepseekV4ForCausalLM.forward, not DeepseekV4Model.forward directly."
        ids = fwd_input_ids.flatten()
        num_tokens = gating_output.shape[0]
        assert (
            ids.shape[0] == num_tokens
        ), f"input_ids length {ids.shape[0]} does not match gating_output num_tokens {num_tokens}"
        tid2eid = self.gate.tid2eid

        # Fused-shared expert: the custom_routing_function path bypasses
        # select_experts' shared-expert append, so the shared expert (slot
        # n_routed_experts) would never be routed and its ~40% contribution
        # dropped. When shared is fused, write the routed result into the first
        # `topk` columns of the global topK buffer (shared cols pre-filled) and
        # return the full [N, topk + n_shared] view.
        num_fused_shared = getattr(self.experts, "num_fused_shared_experts", 0)
        if num_fused_shared > 0:
            import atom.model_ops.topK as _topK_mod

            assert _topK_mod.aiter_topK_meta_data is not None, (
                "AITER topK meta data is not initialized. "
                "init_aiter_topK_meta_data must run before hash-layer routing."
            )
            total_topk_weights, total_topk_ids = _topK_mod.aiter_topK_meta_data
            assert total_topk_weights.shape[0] >= num_tokens
            hash_topk_triton(
                ids,
                gating_output,
                tid2eid,
                renormalize,
                self.routed_scaling_factor,
                total_topk_ids[:num_tokens, :topk],
                total_topk_weights[:num_tokens, :topk],
            )
            return total_topk_weights[:num_tokens], total_topk_ids[:num_tokens]

        topk_ids = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device=gating_output.device
        )
        topk_weights = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device=gating_output.device
        )
        hash_topk_triton(
            ids,
            gating_output,
            tid2eid,
            renormalize,
            self.routed_scaling_factor,
            topk_ids,
            topk_weights,
        )
        return topk_weights, topk_ids

    def routed_expert_forward(
        self, x: torch.Tensor  # [num_tokens, dim]
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Gate + FusedMoE routed-expert pass.

        For hash layers the gate's `tid2eid` lookup needs `input_ids`;
        `DeepseekV4ForCausalLM.forward` stashes it on
        `forward_context.context.input_ids` before each forward, and
        `_hash_topk` (FusedMoE's custom_routing_function) reads it there.
        """
        router_logits = self.gate(x)  # [num_tokens, n_routed_experts]
        return self.experts(hidden_states=x, router_logits=router_logits)

    @staticmethod
    def _gather_ids_for_dp(ids: torch.Tensor, ctx) -> torch.Tensor:
        """All-gather input_ids across DP ranks to match gathered hidden_states."""
        from aiter.dist.parallel_state import get_dp_group

        ids_2d = ids.unsqueeze(-1)
        dp_eager_mode = (
            not ctx.context.dp_uniform_decode
        ) and ctx.dp_metadata is not None
        if dp_eager_mode:
            from atom.model_ops.moe import all_gatherv

            sizes = ctx.dp_metadata.get_sizes_across_dp()
            ids_2d = all_gatherv(ids_2d, sizes, get_dp_group())
        else:
            from atom.model_ops.moe import all_gather_with_padding

            ids_2d, _ = all_gather_with_padding(ids_2d, use_cag=False)
        return ids_2d.flatten()

    @mark_trace
    def combine_outputs(
        self,
        routed: torch.Tensor,  # [num_tokens, dim]
        shared: Optional[torch.Tensor],  # [num_tokens, dim] or None
        prefix: str = "",
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Add shared-expert contribution (when not fused into routed) and
        all-reduce across TP ranks.
        """
        if shared is not None:
            # PCP with ATOM_PCP_MOE_MERGE=1 (non-fused shared only): the shared expert
            # is NOT pcp-sharded (its MergedColumn/RowParallelLinear bind to the
            # 4-card tp group, so every pcp rank holds the same shared weights
            # and computes the same full shared output — pcp-redundant). After
            # this combine the result rides through Block.forward's pcp
            # reduce_scatter, which SUMS the pcp partners. Without correction the
            # shared part would be summed pcp_size times (doubled for pcp=2). So
            # pre-scale shared by 1/pcp_size: the reduce_scatter then RESTORES it
            # to 1x instead of multiplying. routed is genuinely pcp-sharded
            # (partial sum) so it must NOT be scaled — only shared.
            if _moe_pcp_merge_active() or _moe_pcp_merge_decode_active():
                shared = shared * (1.0 / get_pcp_world_size())
            routed = routed + shared
        if self.tp_size > 1:
            routed = tensor_model_parallel_all_reduce(routed)
        return routed

    def single_stream_moe_forward(
        self, x: torch.Tensor  # [num_tokens, dim]
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Sequential: shared_experts → routed_experts → combine."""
        shared = self.shared_experts(x) if self.shared_experts is not None else None
        routed = self.routed_expert_forward(x)
        return self.combine_outputs(
            routed, shared, prefix=f"{self.prefix}.combine_outputs"
        )

    def dual_stream_moe_forward(
        self, x: torch.Tensor  # [num_tokens, dim]
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Run shared_experts on `alt_stream` in parallel with routed_experts
        on the current stream. Mirrors V2's pattern. Both reads of `x` are
        independent; main stream waits on alt_stream's completion before
        combining.
        """
        current_stream = get_forward_context().main_stream
        self.alt_stream.wait_stream(current_stream)
        routed = self.routed_expert_forward(x)
        with torch.cuda.stream(self.alt_stream):
            shared = self.shared_experts.forward(x)
        current_stream.wait_stream(self.alt_stream)
        return self.combine_outputs(
            routed, shared, prefix=f"{self.prefix}.combine_outputs"
        )

    def forward(
        self,
        x: torch.Tensor,  # [num_tokens, dim]  hidden state (post ffn_norm)
    ) -> torch.Tensor:  # [num_tokens, dim]
        # Hash-layer routing reads `input_ids` from forward_context.context
        # inside `_hash_topk` (FusedMoE.custom_routing_function callback);
        # the MoE call itself doesn't need it as a parameter.
        assert (
            x.dim() == 2 and x.shape[-1] == self.dim
        ), f"MoE expects 2D [num_tokens, {self.dim}], got {tuple(x.shape)}"
        if self._use_dual_stream:
            # Shared custom op (also used by V2). Dispatcher reads
            # `_use_dual_stream` + per-call num_tokens vs threshold to pick
            # dual vs single. Custom op = Dynamo barrier so stream context
            # inside `dual_stream_moe_forward` is opaque to torch.compile.
            return torch.ops.aiter.maybe_dual_stream_forward(x, self.prefix)
        return self.single_stream_moe_forward(x)


@dataclass
class HCState:
    residual: torch.Tensor
    post_mix: Optional[torch.Tensor] = None
    comb_mix: Optional[torch.Tensor] = None
    x_prev: Optional[torch.Tensor] = None


class Block(nn.Module):
    """Transformer block with Manifold-Constrained Hyper-Connections (mHC).

    Port of inference/model.py:648-701. ATOM 2D-flat convention: the residual
    stream is widened to `[num_tokens, hc_mult, dim]`. Each sub-layer (attn / ffn):
      1. `hc_pre`: project `[num_tokens, hc_mult, dim]` → `[num_tokens, dim]` via
         Sinkhorn-projected pre-weights (also producing post-weights and combination
         matrix for hc_post).
      2. `attn_norm` + `attn` (or `ffn_norm` + `ffn`): standard sub-layer in
         `[num_tokens, dim]`.
      3. `hc_post`: expand `[num_tokens, dim]` back to `[num_tokens, hc_mult, dim]`
         using the post-weights (gate on the new contribution) + the combination
         matrix applied to the previous residual.
    """

    def __init__(
        self,
