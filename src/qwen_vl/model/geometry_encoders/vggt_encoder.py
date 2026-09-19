"""VGGT geometry encoder implementation."""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .base import BaseGeometryEncoder, GeometryEncoderConfig


def _slice1d(x, start, end):
    return x[:, start:end, ...]


def _slice2d(x, start, end):
    return x[:, :, start:end, ...]


def _slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


_DIM_TO_SLICE = {1: _slice1d, 2: _slice2d, 3: _slice3d}


class StartRecentKVCache:
  def __init__(self, start_size=8, recent_size=48, k_seq_dim=2, v_seq_dim=2):
    self.start_size = start_size
    self.recent_size = recent_size
    self.cache_size = start_size + recent_size
    self.k_slice = _DIM_TO_SLICE[k_seq_dim]
    self.v_slice = _DIM_TO_SLICE[v_seq_dim]

  def __call__(self, past_key_values):
    if past_key_values is None:
      return None
    seq_len = past_key_values[0][0].size(2)
    if seq_len <= self.cache_size:
      return past_key_values
    return [
      [
        torch.cat(
          [self.k_slice(k, 0, self.start_size), self.k_slice(k, seq_len - self.recent_size, seq_len)],
          dim=2,
        ),
        torch.cat(
          [self.v_slice(v, 0, self.start_size), self.v_slice(v, seq_len - self.recent_size, seq_len)],
          dim=2,
        ),
      ]
      for k, v in past_key_values
    ]


class VGGTEncoder(BaseGeometryEncoder):
    """VGGT geometry encoder wrapper."""
    
    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)
        
        # Lazy import to avoid circular dependencies
        from ..vggt.models.vggt import VGGT

        # Initialize VGGT model
        self.vggt = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)
        
        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False

        self.reference_frame = config.reference_frame    
        self.patch_size = 14
        self._vggt_pretrained_path = config.model_path
        self._depth_head_ready = False
        self._adageo_heads_ready = False
        self._eval_streaming = False
        self._streaming_past_key_values = None
        self._streaming_past_key_values_camera = None
        self._streaming_importance_cache = None
        self._streaming_frame_metadata = None
        self._streaming_frame_idx = 0
        self._streaming_patch_hw = None
        self.last_vggt_ms = 0.0
        # Legacy frame-strict fallback buffers raw geometry. The Qwen3.5 language-add
        # eval path instead enables _eval_projected_cache and stores post-MLP deltas.
        self._eval_frame_strict = False
        self._eval_projected_cache = False
        self._eval_window_indices = None
        self._frame_feature_buffer = None
        _adageo_env = os.environ.get("USE_ADAGEO_KV_CACHE")
        self.use_adageo_kv_cache = (
            _adageo_env.lower() in ("1", "true", "yes")
            if _adageo_env is not None
            else bool(config.use_adageo_kv_cache)
        )
        self.adageo_kv_score_mode = os.environ.get("ADAGEO_KV_SCORE_MODE", "importance").strip().lower()
        if self.adageo_kv_score_mode not in ("importance", "vln_segment_transition"):
            raise ValueError(
                "ADAGEO_KV_SCORE_MODE must be 'importance' or 'vln_segment_transition', got "
                f"{self.adageo_kv_score_mode!r}"
            )
        self.use_vln_segment_transition = (
            self.use_adageo_kv_cache and self.adageo_kv_score_mode == "vln_segment_transition"
        )
        self._vln_instruction_state = None
        self._vln_transition_state = None
        self._vln_frame_descriptors = None
        self._vln_metadata_per_layer = None
        self._vln_layer_budgets = None
        self._vln_score_probe = None
        self._vln_expand_index_cache = None
        self.vggt_total_budget = int(
            os.environ.get("VGGT_TOTAL_BUDGET", config.vggt_total_budget)
        )
        if self.vggt_total_budget <= 0:
            raise ValueError(
                "VGGT_TOTAL_BUDGET must be a positive integer, got "
                f"{self.vggt_total_budget}"
            )
        self.vggt_importance_weights_path = config.vggt_importance_weights_path
        self.vggt_budget_proportions_path = os.environ.get(
            "VGGT_BUDGET_PROPORTIONS_PATH", config.vggt_budget_proportions_path
        )
        self.vln_segment_transition_weights_path = os.environ.get(
            "VLN_SEGMENT_TRANSITION_WEIGHTS_PATH",
            config.vln_segment_transition_weights_path,
        )
        self.vggt_importance_weights = None
        self.vln_segment_transition_weights = None
        self._vggt_configs_loaded = False
        print(f"[VGGTEncoder] use_adageo_kv_cache={self.use_adageo_kv_cache}")
        _kv_start = int(os.environ.get("VGGT_KV_START", "8"))
        _kv_recent = int(os.environ.get("VGGT_KV_RECENT", "48"))
        print(f"[VGGTEncoder] eval KV-cache window: start={_kv_start} recent={_kv_recent} (total={_kv_start + _kv_recent} frames)")
        if self.use_adageo_kv_cache:
            print(
                f"[VGGTEncoder] AdaGeoKV KV-cache enabled: total_budget={self.vggt_total_budget} "
                f"score_mode={self.adageo_kv_score_mode}"
            )
        self._kv_cache_trim = StartRecentKVCache(start_size=_kv_start, recent_size=_kv_recent, k_seq_dim=2, v_seq_dim=2)
        # Camera-head KV window, in frames. `trunk_fn` appends one token per refinement
        # iteration per trunk layer every frame, so this cache grew linearly with episode
        # length (~50 MB at 400 steps) and was the last unbounded CUDA component. Frame 0
        # is kept because it defines VGGT's reference camera. Set recent<=0 to disable.
        self._camera_kv_start_frames = int(os.environ.get("VGGT_CAMERA_KV_START", "1"))
        self._camera_kv_recent_frames = int(os.environ.get("VGGT_CAMERA_KV_RECENT", "64"))
        self._camera_kv_tokens_per_frame = None
        if self._camera_kv_recent_frames > 0:
            print(
                f"[VGGTEncoder] camera-head KV window: start={self._camera_kv_start_frames} "
                f"recent={self._camera_kv_recent_frames} frames"
            )

    def set_eval_streaming(self, enabled: bool) -> None:
        self._eval_streaming = bool(enabled)

    def set_eval_frame_strict(self, enabled: bool) -> None:
        """Incremental frame-strict eval: buffer each frame's geometry (encoded with the
        growing KV) and return the requested window per-frame instead of broadcasting."""
        self._eval_frame_strict = bool(enabled)
        if enabled and self._frame_feature_buffer is None:
            self._frame_feature_buffer = []

    def set_eval_projected_cache(self, enabled: bool) -> None:
        """Let the Qwen fusion path cache post-MLP deltas instead of raw VGGT features."""
        self._eval_projected_cache = bool(enabled)
        if enabled:
            self._frame_feature_buffer = None

    def set_eval_window_indices(self, indices) -> None:
        """Trajectory frame indices to gather from the per-frame buffer this step."""
        self._eval_window_indices = list(indices) if indices is not None else None

    def reset_streaming_cache(self) -> None:
        self._streaming_past_key_values = None
        self._streaming_past_key_values_camera = None
        self._streaming_importance_cache = None
        self._streaming_frame_metadata = None
        self._streaming_frame_idx = 0
        self._camera_kv_tokens_per_frame = None
        self._vln_metadata_per_layer = None
        if self._vln_transition_state is not None:
            self._vln_transition_state.reset()
        self._vln_frame_descriptors = []
        self.last_vggt_ms = 0.0
        self._frame_feature_buffer = (
            [] if self._eval_frame_strict and not self._eval_projected_cache else None
        )
        self._eval_window_indices = None
        self._reset_vggt_attention_cache_state()

    def set_vln_instruction_state(self, state) -> None:
        """Install immutable per-episode text metadata for the opt-in VLN scorer."""
        if not self.use_vln_segment_transition:
            return
        from ..vggt.eviction.vln_segment_transition import RecentTransitionState
        from ..vggt.eviction.vln_score_probe import VLNScoreProbe

        self._load_vggt_configs()
        transition_cfg = self.vln_segment_transition_weights["transition"]
        self._vln_instruction_state = state
        self._vln_frame_descriptors = []
        self._vln_transition_state = RecentTransitionState(
            max_frames=int(transition_cfg["recent_window_size"])
        )
        self._vln_metadata_per_layer = None
        # Read-only score instrumentation; a no-op unless VLN_SCORE_PROBE=1.
        if self._vln_score_probe is not None:
            self._vln_score_probe.flush()
        self._vln_score_probe = VLNScoreProbe(
            num_segments=int(state.segment_embeddings.shape[0]),
            instruction=state.raw_instruction,
        )
        self._vln_score_probe.start_episode(getattr(state, "raw_instruction", "")[:64])
        # Resolve fixed offline-profiled budgets once during episode initialization,
        # outside the per-frame scorer (which stays GPU-only and synchronization-free).
        self._vln_layer_budgets = self.vggt.aggregator._calculate_dynamic_budgets(
            self.vggt_total_budget
        ).detach().cpu().tolist()
        print(
            "[VGGTEncoder] initialized vln_segment_transition: "
            f"segments={state.segment_embeddings.shape[0]} "
            f"recent_window={transition_cfg['recent_window_size']} "
            f"layer_budgets=[{min(self._vln_layer_budgets)}, {max(self._vln_layer_budgets)}]"
        )

    @torch.no_grad()
    def finalize_vln_segment_transition(
        self,
        aligned_visual_tokens: torch.Tensor,
        aligned_grid_hw=None,
    ) -> None:
        """Score the just-appended frame and globally prune every VGGT layer cache."""
        if not self.use_vln_segment_transition:
            return
        if self._vln_instruction_state is None or self._vln_transition_state is None:
            raise RuntimeError("VLN instruction state must be initialized before geometry forward")
        if not self._streaming_frame_metadata:
            raise RuntimeError("current-frame geometry metadata is unavailable")

        from ..vggt.eviction.vln_segment_transition import (
            AdaGeoKVTokenMetadata,
            build_frame_descriptor,
            compute_instruction_segment_relevance,
            compute_local_transition_score,
            compute_cache_novelty,
            compute_transition_anchor,
            compute_candidate_final_score,
            concat_metadata,
            gather_metadata,
        )

        visual = aligned_visual_tokens.reshape(-1, aligned_visual_tokens.shape[-1])
        device = visual.device
        meta = self._streaming_frame_metadata[-1]
        depth_conf = meta.get("depth_conf")
        point_conf = meta.get("conf")

        # Resolve the token grids up front: confidence must be pooled onto the *actual*
        # merged grid, not onto a near-square factorisation of the token count.
        tokens_per_frame = int(self.vggt.aggregator.global_blocks[0].attn._tokens_per_frame)
        n_special = int(self.vggt.aggregator.patch_start_idx)
        n_patch = tokens_per_frame - n_special
        patch_hw = self._streaming_patch_hw
        if patch_hw is None or patch_hw[0] * patch_hw[1] != n_patch:
            raise AssertionError(f"invalid streaming patch grid {patch_hw} for {n_patch} patches")
        if aligned_grid_hw is None or aligned_grid_hw[0] * aligned_grid_hw[1] != visual.shape[0]:
            raise AssertionError(
                f"invalid aligned grid {aligned_grid_hw} for {visual.shape[0]} projected tokens"
            )
        patch_h, patch_w = patch_hw
        aligned_h, aligned_w = aligned_grid_hw

        def pool_conf(value):
            if value is None:
                return None
            value = value.to(device).float().squeeze()
            if value.ndim == 2:
                value = F.adaptive_avg_pool2d(
                    value[None, None], (aligned_h, aligned_w)
                ).flatten()
            else:
                value = F.adaptive_avg_pool1d(
                    value.flatten()[None, None], aligned_h * aligned_w
                ).flatten()
            # VGGT emits confidence through `expp1` (head_act.activate_head), i.e.
            # conf = 1 + exp(x) in (1, inf). Clamping to [0,1] would saturate every
            # token to exactly 1.0 and silently kill this term, the descriptor
            # weighting and the transition score. Map monotonically onto [0,1) instead.
            return (1.0 - 1.0 / value.clamp_min(1.0 + 1e-6)).clamp(0, 1)

        depth_c = pool_conf(depth_conf)
        point_c = pool_conf(point_conf)
        if depth_c is None and point_c is None:
            confidence = torch.full((visual.shape[0],), 0.5, device=device)
        elif depth_c is None:
            confidence = point_c
        elif point_c is None:
            confidence = depth_c
        else:
            confidence = torch.minimum(depth_c, point_c)

        transition_cfg = self.vln_segment_transition_weights["transition"]
        relevance, best_segment = compute_instruction_segment_relevance(
            visual,
            self._vln_instruction_state.segment_embeddings.to(device),
            centering=self.vln_segment_transition_weights.get("instruction", {}).get(
                "centering", "none"
            ),
        )
        descriptor = build_frame_descriptor(
            visual, confidence,
            pooling=transition_cfg.get("descriptor_pooling", "confidence"),
        )
        transition = compute_local_transition_score(
            descriptor, self._vln_transition_state.descriptors,
            reduce=transition_cfg.get("novelty_reduce", "mean"),
        )
        # The bank must contain this frame before the cache-wide novelty is computed:
        # the current frame's tokens are scored in the same pass, so its frame_id has to
        # be a valid index. The recent-window deque above is deliberately appended later,
        # since `compute_local_transition_score` must only see *previous* frames.
        if self._vln_frame_descriptors is not None:
            self._vln_frame_descriptors.append(descriptor.detach())
        anchor = compute_transition_anchor(transition, confidence.shape[0])

        current_frame_id = self._streaming_frame_idx - 1
        score_weights = self.vln_segment_transition_weights["score_weights"]
        merged_final = (
            float(score_weights["confidence"]) * confidence
            + float(score_weights["instruction"]) * relevance
            + float(score_weights["transition"]) * anchor
        ).clamp(0, 1)

        # Fusion merges spatial groups; assign each merged score to its source KV patches.
        # The merge is an exact floor-divide by `merge`, so invert it that way: scaling by
        # aligned_h/patch_h is only equivalent when patch_h is even, and silently shifts
        # every other row when the patch grid is odd.
        expand_index = self._vln_expand_index(patch_h, patch_w, aligned_h, aligned_w, device)

        def expand(x):
            # Map each merged language token back to its exact 2-D source patches;
            # trimmed right/bottom border patches inherit their nearest merged token.
            return x.reshape(-1).index_select(0, expand_index)

        # Cache raw components in fp32 because cache-wide z-score normalization happens
        # later, immediately before topk. The realised language-relevance spread is only
        # ~1e-2; fp16 storage would collapse many distinct tokens into ties before the
        # normalization step and the lost ordering cannot be recovered.
        score_dtype = torch.float32
        boost = float((self.vggt_importance_weights or {}).get("special_token_boost", 0.3))
        eps = float((self.vggt_importance_weights or {}).get("special_token_tiebreak_eps", 1e-6))
        patch_final = expand(merged_final)
        zeros_special = torch.zeros(n_special, device=device)
        frame_ids = torch.full((tokens_per_frame,), current_frame_id, device=device, dtype=torch.int32)
        is_special = torch.arange(tokens_per_frame, device=device) < n_special
        # Special-token priority: phi(t, p_sp) = s_frame(t) + boost + eps * r, where r is the
        # INTRA-FRAME rank of the special token. Baking the privilege in here keeps the
        # tiebreak infinitesimal; deriving it in the eviction loop from the token's
        # position in the (growing) cache would scale it with cache size instead.
        special_final = patch_final.max() + boost + eps * torch.arange(
            n_special, device=device, dtype=torch.float32
        )
        # final_score stays fp32: the realised score range is ~1e-2 while the fp16 ulp
        # near 0.6 is ~4.9e-4, which would collapse most of a frame into exact ties and
        # hand the eviction decision to topk's tiebreak.
        new_meta = AdaGeoKVTokenMetadata(
            frame_id=frame_ids,
            confidence_score=torch.cat([zeros_special, expand(confidence)]).to(score_dtype),
            instruction_score=torch.cat([zeros_special, expand(relevance)]).to(score_dtype),
            transition_score=torch.cat([zeros_special, expand(anchor)]).to(score_dtype),
            final_score=torch.cat([special_final, patch_final]).float(),
            is_special=is_special,
            best_segment_id=torch.cat([
                torch.full((n_special,), -1, device=device, dtype=torch.int16), expand(best_segment)
            ]),
        )

        probe = self._vln_score_probe
        if probe is not None and probe.should_log(current_frame_id):
            probe.log_frame_scores(
                current_frame_id,
                confidence=confidence,
                instruction=relevance,
                transition_scalar=transition,
                anchor=anchor,
                final=merged_final,
                best_segment=best_segment,
            )

        # Re-score every seen frame against the same reference population, once per step.
        # Freezing novelty at insertion bakes in a when-did-this-arrive bias that never
        # gets revised; recomputing removes it. Cheap: one [F,F] cosine on 1024-d unit
        # descriptors, F <= episode length.
        cache_novelty = None
        if transition_cfg.get("novelty_ref", "insertion") == "cache":
            cache_novelty = compute_cache_novelty(self._vln_frame_descriptors)

        depth_layers = len(self._streaming_past_key_values)
        if self._vln_metadata_per_layer is None:
            self._vln_metadata_per_layer = [None] * depth_layers
        if self._vln_layer_budgets is None:
            raise RuntimeError("VLN layer budgets were not initialized")
        for layer_idx, kv in enumerate(self._streaming_past_key_values):
            if kv is None:
                continue
            candidate_meta = concat_metadata(self._vln_metadata_per_layer[layer_idx], new_meta)
            key, value = kv
            if key.shape[2] != candidate_meta.final_score.numel():
                raise AssertionError(
                    f"layer {layer_idx} KV/metadata mismatch: {key.shape[2]} vs "
                    f"{candidate_meta.final_score.numel()}"
                )
            layer_budget = int(self._vln_layer_budgets[layer_idx])
            if cache_novelty is not None:
                fid = candidate_meta.frame_id.long()
                if int(fid.max()) >= cache_novelty.numel():
                    raise AssertionError(
                        f"frame_id {int(fid.max())} has no descriptor "
                        f"(bank holds {cache_novelty.numel()})"
                    )
                refreshed = cache_novelty[fid]
                candidate_meta.transition_score = (
                    refreshed * (~candidate_meta.is_special.bool())
                ).to(candidate_meta.transition_score.dtype)
            candidate_meta.final_score, normalized_components = compute_candidate_final_score(
                candidate_meta,
                score_weights,
                special_token_boost=boost,
                normalization=self.vln_segment_transition_weights["normalization"]["candidate_terms"],
            )
            candidate_component_std = candidate_variance_share = None
            candidate_true_share = candidate_component_corr = None
            if (
                probe is not None
                and probe.should_log(current_frame_id)
                and layer_idx in probe.layers
            ):
                component_names = tuple(normalized_components)
                candidate_component_std = torch.stack([
                    normalized_components[name].std(unbiased=False) for name in component_names
                ])
                weighted_variance = torch.stack([
                    (float(score_weights[name]) * candidate_component_std[idx]).square()
                    for idx, name in enumerate(component_names)
                ])
                # The proxy share above assumes the components are independent. They are
                # not, so a component can report healthy standalone variance while being
                # cancelled inside the sum. Var(S) = sum_i Cov(w_i z_i, S) is the
                # decomposition that survives correlation: it sums to 1, and a component
                # that fights the sum goes negative.
                comp_matrix = torch.stack(
                    [normalized_components[name] for name in component_names]
                )
                comp_centered = comp_matrix - comp_matrix.mean(dim=1, keepdim=True)
                comp_sigma = comp_centered.square().mean(dim=1).sqrt().clamp_min(1e-6)
                candidate_component_corr = (
                    comp_centered @ comp_centered.T
                ) / (comp_matrix.shape[1] * comp_sigma[:, None] * comp_sigma[None, :])
                w_vec = torch.tensor(
                    [float(score_weights[name]) for name in component_names],
                    device=comp_matrix.device, dtype=comp_matrix.dtype,
                )
                summed_centered = (w_vec[:, None] * comp_centered).sum(0)
                candidate_true_share = (
                    (w_vec[:, None] * comp_centered) * summed_centered
                ).mean(dim=1) / summed_centered.square().mean().clamp_min(1e-12)
                candidate_variance_share = weighted_variance / weighted_variance.sum().clamp_min(1e-12)
            if key.shape[2] > layer_budget:
                # Only pay for the gather when the layer is actually over budget: a
                # full-cache index_select moves ~2 * 2048 * T bytes per layer and is a
                # pure no-op while the cache is still filling.
                keep = torch.topk(
                    candidate_meta.final_score, k=layer_budget, largest=True, sorted=False
                ).indices.sort().values
                self._streaming_past_key_values[layer_idx] = [
                    key.index_select(2, keep), value.index_select(2, keep)
                ]
                candidate_meta = gather_metadata(candidate_meta, keep)
                if candidate_meta.final_score.numel() > layer_budget:
                    raise AssertionError(f"layer {layer_idx} exceeded its cache budget")
            self._vln_metadata_per_layer[layer_idx] = candidate_meta
            if probe is not None:
                probe.log_layer_retention(
                    current_frame_id,
                    layer_idx,
                    candidate_meta,
                    layer_budget,
                    candidate_component_std=candidate_component_std,
                    candidate_variance_share=candidate_variance_share,
                    candidate_true_share=candidate_true_share,
                    candidate_component_corr=candidate_component_corr,
                )
        # Keep the descriptor fp32 and already L2-normalised: casting it to the visual
        # dtype only to cast it back (and re-normalise) next frame loses precision on a
        # cosine that consecutive VLN frames drive to within ~1e-3 of each other.
        self._vln_transition_state.descriptors.append(descriptor.detach())
        # Only the immediately previous pose is needed next frame. Drop old dense
        # depth/confidence maps so this scorer does not create an unbounded side memory.
        if len(self._streaming_frame_metadata) > 2:
            self._streaming_frame_metadata = self._streaming_frame_metadata[-2:]

    def _vln_expand_index(self, patch_h, patch_w, aligned_h, aligned_w, device):
        """Flat merged-grid index for every source patch, cached across frames.

        The projector merges each `merge x merge` patch block into one language token,
        so the inverse map is a floor-divide; right/bottom patches trimmed before the
        merge fall back to the last merged row/column.
        """
        key = (patch_h, patch_w, aligned_h, aligned_w, str(device))
        cached = self._vln_expand_index_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        merge_h = max(1, patch_h // aligned_h)
        merge_w = max(1, patch_w // aligned_w)
        y = torch.arange(patch_h, device=device).div(merge_h, rounding_mode="floor").clamp_max(aligned_h - 1)
        x = torch.arange(patch_w, device=device).div(merge_w, rounding_mode="floor").clamp_max(aligned_w - 1)
        index = (y[:, None] * aligned_w + x[None, :]).reshape(-1)
        self._vln_expand_index_cache = (key, index)
        return index

    def _reset_vggt_attention_cache_state(self) -> None:
        blocks = list(self.vggt.aggregator.global_blocks)
        # The camera trunk also runs with use_cache=True, so it accumulates the same
        # per-Attention cache state; without this its state leaked across episodes.
        camera_head = getattr(self.vggt, "camera_head", None)
        if camera_head is not None:
            blocks.extend(camera_head.trunk)
        for block in blocks:
            if hasattr(block.attn, "_reset_cache_state"):
                block.attn._reset_cache_state()

    @staticmethod
    def _resolve_config_path(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        if os.path.isabs(path) and os.path.isfile(path):
            return path
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        )
        for candidate in (path, os.path.join(os.getcwd(), path), os.path.join(repo_root, path)):
            candidate = os.path.normpath(candidate)
            if os.path.isfile(candidate):
                return candidate
        return None

    def _load_vggt_configs(self) -> None:
        if self._vggt_configs_loaded:
            return
        weights_path = self._resolve_config_path(self.vggt_importance_weights_path)
        if weights_path is not None:
            with open(weights_path) as f:
                self.vggt_importance_weights = json.load(f)
        proportions_path = self._resolve_config_path(self.vggt_budget_proportions_path)
        if proportions_path is not None:
            with open(proportions_path) as f:
                cfg = json.load(f)
            proportions = cfg["proportions"]
            depth = int(self.vggt.aggregator.depth)
            if len(proportions) != depth:
                raise ValueError(
                    f"{proportions_path}: expected {depth} proportions, "
                    f"got {len(proportions)}"
                )
            if abs(sum(float(x) for x in proportions) - 1.0) > 1e-4:
                raise ValueError(
                    f"{proportions_path}: proportions must sum to 1, "
                    f"got {sum(float(x) for x in proportions):.6f}"
                )
            self.vggt.aggregator.budget_proportions = torch.tensor(
                proportions, dtype=torch.float32
            )
            print(
                f"[VGGTEncoder] layer budget proportions: {proportions_path} "
                f"(min {min(proportions):.4f}, max {max(proportions):.4f})"
            )
        if self.use_vln_segment_transition:
            vln_weights_path = self._resolve_config_path(
                self.vln_segment_transition_weights_path
            )
            if vln_weights_path is None:
                raise FileNotFoundError(
                    "Could not resolve VLN segment-transition weight profile: "
                    f"{self.vln_segment_transition_weights_path!r}"
                )
            with open(vln_weights_path) as f:
                vln_weights = json.load(f)
            self._validate_vln_segment_transition_weights(vln_weights, vln_weights_path)
            self.vln_segment_transition_weights = vln_weights
        self._vggt_configs_loaded = True

    @staticmethod
    def _validate_vln_segment_transition_weights(config, path: str) -> None:
        if config.get("score_mode") != "vln_segment_transition":
            raise ValueError(f"{path}: score_mode must be 'vln_segment_transition'")
        required = {
            "score_weights": ("confidence", "instruction", "transition"),
        }
        optional = {}
        for section, names in required.items():
            values = config.get(section)
            if not isinstance(values, dict):
                raise ValueError(f"{path}: missing object {section!r}")
            names = names + tuple(
                n for n in optional.get(section, ()) if n in values
            )
            unknown = set(values) - set(names)
            if unknown:
                raise ValueError(
                    f"{path}: unknown {section} entries {sorted(unknown)}; "
                    f"a misspelled weight would silently be ignored"
                )
            for name in names:
                value = values.get(name)
                if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                    raise ValueError(f"{path}: {section}.{name} must be in [0,1]")
            if abs(sum(float(values[name]) for name in names) - 1.0) > 1e-6:
                raise ValueError(f"{path}: values in {section} must sum to 1")
        confidence = config.get("confidence", {})
        if confidence.get("merge") != "min":
            raise ValueError(f"{path}: confidence.merge must be 'min'")
        normalization = config.get("normalization", {})
        if normalization.get("candidate_terms") not in ("zscore", "sigmoid"):
            raise ValueError(
                f"{path}: normalization.candidate_terms must be 'zscore' or 'sigmoid'"
            )
        instruction_cfg = config.get("instruction", {})
        if not isinstance(instruction_cfg, dict):
            raise ValueError(f"{path}: instruction must be an object")
        if instruction_cfg.get("centering", "none") not in ("none", "both"):
            raise ValueError(
                f"{path}: instruction.centering must be 'none' or 'both'"
            )
        transition = config.get("transition", {})
        window = transition.get("recent_window_size")
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"{path}: transition.recent_window_size must be a positive integer")
        if transition.get("descriptor_pooling", "confidence") not in ("confidence", "uniform"):
            raise ValueError(
                f"{path}: transition.descriptor_pooling must be 'confidence' or 'uniform'"
            )
        if transition.get("novelty_reduce", "mean") not in ("mean", "max"):
            raise ValueError(
                f"{path}: transition.novelty_reduce must be 'mean' or 'max'"
            )
        novelty_ref = transition.get("novelty_ref", "insertion")
        if novelty_ref not in ("insertion", "cache"):
            raise ValueError(
                f"{path}: transition.novelty_ref must be 'insertion' or 'cache'"
            )
        
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using VGGT and return the default (final) feature set."""
        self.vggt.eval()

        # Apply reference frame transformation
        images = self._apply_reference_frame_transform(images)

        # Determine dtype for mixed precision
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images[None])
                features = aggregated_tokens_list[-2][0, :, patch_start_idx:]

        # Apply inverse reference frame transformation
        features = self._apply_inverse_reference_frame_transform(features)

        return features

    def encode_layers(
        self,
        images: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        spatial_merge_size: int = 1,
        include_camera_token: bool = False,
    ):
        """Encode images and return features from specific aggregator layers."""
        self.vggt.eval()

        # Apply reference frame transformation
        images = self._apply_reference_frame_transform(images)

        # Determine dtype for mixed precision
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images[None])

        n_image, _, height, width = images.shape
        h_patch = height // self.patch_size
        w_patch = width // self.patch_size
        spatial_merge_size = spatial_merge_size if spatial_merge_size and spatial_merge_size > 0 else 2

        tensor_features = []

        if layer_indices is None:
            layer_indices = [-2]

        for idx in layer_indices:
            tokens = aggregated_tokens_list[idx][0]
            tokens = self._apply_inverse_reference_frame_transform(tokens) # flip frames if ture
            patch_tokens = tokens[:, patch_start_idx:]
            camera_token = tokens[:, 0:1] # first token

            # reshape and trim
            patch_grid = patch_tokens.reshape(n_image, h_patch, w_patch, -1)
            trimmed_h = (h_patch // spatial_merge_size) * spatial_merge_size or h_patch
            trimmed_w = (w_patch // spatial_merge_size) * spatial_merge_size or w_patch
            patch_grid = patch_grid[:, :trimmed_h, :trimmed_w, :]
            patch_grid = patch_grid.reshape(n_image, trimmed_h // spatial_merge_size, spatial_merge_size, trimmed_w // spatial_merge_size, spatial_merge_size, -1)
            patch_grid = patch_grid.permute(0, 1, 3, 2, 4, 5)
            patch_tokens = patch_grid.reshape(n_image, trimmed_h * trimmed_w, -1)

            if not include_camera_token:
                geo_feature = patch_tokens
            else:
                geo_feature = torch.cat([camera_token, patch_tokens], dim=1)

            tensor_features.append(geo_feature.to(dtype).contiguous())

        self._maybe_debug_geometry_layers(
            layer_indices=layer_indices,
            tensor_features=tensor_features,
            images=images,
            trimmed_h=trimmed_h,
            trimmed_w=trimmed_w,
            streaming=False,
        )

        return tensor_features

    def _maybe_debug_geometry_layers(
        self,
        *,
        layer_indices: List[int],
        tensor_features: List[torch.Tensor],
        images: torch.Tensor,
        trimmed_h: int,
        trimmed_w: int,
        streaming: bool,
    ) -> None:
        from qwen_vl.debug import vln_debug

        if not vln_debug.is_enabled():
            return
        if vln_debug.should_save_geo_layers():
            vln_debug.save_geometry_encoder_layers(
                layer_indices=layer_indices,
                tensor_features=tensor_features,
                trimmed_h=trimmed_h,
                trimmed_w=trimmed_w,
                input_images=images,
                streaming=streaming,
            )
        if vln_debug.should_save_depth():
            vln_debug.save_vggt_depth_maps(self, images)

    def supports_streaming(self) -> bool:
        import inspect
        params = inspect.signature(self.vggt.aggregator.forward).parameters
        return "use_cache" in params

    def _format_streaming_layer_features(
        self,
        layer_output: torch.Tensor,
        *,
        h_patch: int,
        w_patch: int,
        spatial_merge_size: int,
        include_camera_token: bool,
        dtype: torch.dtype,
    ):
        frame_tokens = layer_output[0, -1:, :, :]
        patch_grid = frame_tokens.reshape(1, h_patch, w_patch, -1)
        trimmed_h = (h_patch // spatial_merge_size) * spatial_merge_size or h_patch
        trimmed_w = (w_patch // spatial_merge_size) * spatial_merge_size or w_patch
        patch_grid = patch_grid[:, :trimmed_h, :trimmed_w, :]
        patch_grid = patch_grid.reshape(
            1,
            trimmed_h // spatial_merge_size,
            spatial_merge_size,
            trimmed_w // spatial_merge_size,
            spatial_merge_size,
            -1,
        )
        patch_grid = patch_grid.permute(0, 1, 3, 2, 4, 5)
        patch_tokens = patch_grid.reshape(1, trimmed_h * trimmed_w, -1)

        if include_camera_token:
            camera_token = layer_output[0, -1:, 0:1, :]
            geo_feature = torch.cat([camera_token, patch_tokens], dim=1)
        else:
            geo_feature = patch_tokens

        return geo_feature.to(dtype).contiguous(), trimmed_h, trimmed_w

    def encode_layers_streaming(
        self,
        images: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        spatial_merge_size: int = 1,
        include_camera_token: bool = False,
        frame_strict: bool = False,
    ):
        if not self.supports_streaming():
            raise RuntimeError(
                "VGGT aggregator does not support streaming (missing KV cache). "
            )

        if self._eval_streaming:
            return self._encode_layers_streaming_eval(
                images,
                layer_indices=layer_indices,
                spatial_merge_size=spatial_merge_size,
                include_camera_token=include_camera_token,
            )

        self.vggt.eval()
        images = self._apply_reference_frame_transform(images)
        n_image, _, height, width = images.shape
        h_patch = height // self.patch_size
        w_patch = width // self.patch_size
        self._streaming_patch_hw = (h_patch, w_patch)
        spatial_merge_size = spatial_merge_size if spatial_merge_size and spatial_merge_size > 0 else 2

        if layer_indices is None:
            layer_indices = [-2]

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        past_key_values = [None] * self.vggt.aggregator.depth
        aggregated_tokens_list = None
        patch_start_idx = 0
        # FUSION_FRAME_STRICT: keep every frame's geometry (each frame fused with its
        # own vision tokens) instead of only the last frame (broadcast to all frames).
        # per_frame_layers stays None for the non-strict path, which is unchanged.
        per_frame_layers = {idx: [] for idx in layer_indices} if frame_strict else None

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                for frame_idx, frame in enumerate(images):
                    frame_input = frame.unsqueeze(0).unsqueeze(0)
                    output = self.vggt.aggregator(
                        frame_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                        past_frame_idx=frame_idx,
                    )
                    aggregated_tokens_list, patch_start_idx, past_key_values = output
                    if frame_strict:
                        for idx in layer_indices:
                            per_frame_layers[idx].append(aggregated_tokens_list[idx])

        tensor_features = []
        for idx in layer_indices:
            if frame_strict:
                # [n_image, n_patch, dim]: concatenate each frame's own current-frame
                # tokens (frame t already attended to <=t via the KV cache).
                frame_tokens = torch.cat(
                    [lo[0, -1:, patch_start_idx:, :] for lo in per_frame_layers[idx]], dim=0
                )
                camera_token = (
                    torch.cat([lo[0, -1:, 0:1, :] for lo in per_frame_layers[idx]], dim=0)
                    if include_camera_token
                    else None
                )
                batch = n_image
            else:
                layer_output = aggregated_tokens_list[idx]
                frame_tokens = layer_output[0, -1:, patch_start_idx:, :]  # [1, n_patch, dim]
                camera_token = layer_output[0, -1:, 0:1, :] if include_camera_token else None
                batch = 1
            # reference_frame flip is applied on the image sequence before the loop

            patch_grid = frame_tokens.reshape(batch, h_patch, w_patch, -1)
            trimmed_h = (h_patch // spatial_merge_size) * spatial_merge_size or h_patch
            trimmed_w = (w_patch // spatial_merge_size) * spatial_merge_size or w_patch
            patch_grid = patch_grid[:, :trimmed_h, :trimmed_w, :]
            patch_grid = patch_grid.reshape(
                batch,
                trimmed_h // spatial_merge_size,
                spatial_merge_size,
                trimmed_w // spatial_merge_size,
                spatial_merge_size,
                -1,
            )
            patch_grid = patch_grid.permute(0, 1, 3, 2, 4, 5)
            patch_tokens = patch_grid.reshape(batch, trimmed_h * trimmed_w, -1)

            if include_camera_token:
                geo_feature = torch.cat([camera_token, patch_tokens], dim=1)
            else:
                geo_feature = patch_tokens

            tensor_features.append(geo_feature.to(dtype).contiguous())

        from qwen_vl.debug import vln_debug

        if vln_debug.is_enabled() and tensor_features:
            vln_debug.log_geometry_streaming(
                n_image=n_image,
                h_patch=h_patch,
                w_patch=w_patch,
                spatial_merge_size=spatial_merge_size,
                patch_tokens_shape=tuple(tensor_features[0].shape),
            )

        trimmed_h = (h_patch // spatial_merge_size) * spatial_merge_size or h_patch
        trimmed_w = (w_patch // spatial_merge_size) * spatial_merge_size or w_patch
        self._maybe_debug_geometry_layers(
            layer_indices=layer_indices,
            tensor_features=tensor_features,
            images=images,
            trimmed_h=trimmed_h,
            trimmed_w=trimmed_w,
            streaming=True,
        )

        return tensor_features

    def _encode_layers_streaming_eval(
        self,
        images: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        spatial_merge_size: int = 1,
        include_camera_token: bool = False,
    ):
        """Habitat eval: encode only the current frame, keep VGGT KV across steps."""
        self.vggt.eval()
        images = self._apply_reference_frame_transform(images)
        frame = images[-1]
        _, height, width = frame.shape
        h_patch = height // self.patch_size
        w_patch = width // self.patch_size
        self._streaming_patch_hw = (h_patch, w_patch)
        spatial_merge_size = spatial_merge_size if spatial_merge_size and spatial_merge_size > 0 else 2

        if layer_indices is None:
            layer_indices = [-2]

        if self._streaming_past_key_values is None:
            self._streaming_past_key_values = [None] * self.vggt.aggregator.depth
            self._streaming_past_key_values_camera = None
            self._streaming_importance_cache = {}
            self._streaming_frame_metadata = []
            self._reset_vggt_attention_cache_state()
        elif self._streaming_importance_cache is None:
            self._streaming_importance_cache = {}
        if self._streaming_frame_metadata is None:
            self._streaming_frame_metadata = []

        if self.use_adageo_kv_cache:
            self._load_vggt_configs()
            self._ensure_adageo_heads()
            if self._streaming_past_key_values_camera is None:
                self._streaming_past_key_values_camera = [None] * self.vggt.camera_head.trunk_depth

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        frame_input = frame.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                if torch.cuda.is_available():
                    vggt_start = torch.cuda.Event(enable_timing=True)
                    vggt_end = torch.cuda.Event(enable_timing=True)
                    vggt_start.record()

                output = self.vggt.aggregator(
                    frame_input,
                    past_key_values=self._streaming_past_key_values,
                    use_cache=True,
                    past_frame_idx=self._streaming_frame_idx,
                    # The segment-transition mode prunes immediately after its existing
                    # language-space projector runs; cosine-importance mode still prunes here.
                    total_budget=(
                        self.vggt_total_budget
                        if self.use_adageo_kv_cache and not self.use_vln_segment_transition else 0
                    ),
                    eviction_mode=(
                        "importance"
                        if self.use_adageo_kv_cache and not self.use_vln_segment_transition else ""
                    ),
                    frame_metadata=(
                        self._streaming_frame_metadata
                        if self.use_adageo_kv_cache and not self.use_vln_segment_transition else None
                    ),
                    importance_cache=(
                        self._streaming_importance_cache
                        if self.use_adageo_kv_cache and not self.use_vln_segment_transition else None
                    ),
                    importance_weights=(
                        self.vggt_importance_weights
                        if self.use_adageo_kv_cache and not self.use_vln_segment_transition else None
                    ),
                )
                aggregated_tokens_list, patch_start_idx, self._streaming_past_key_values = output
                if self.use_adageo_kv_cache:
                    self._append_adageo_frame_metadata(
                        aggregated_tokens_list,
                        frame_input,
                        patch_start_idx,
                    )
                else:
                    self._streaming_past_key_values = self._kv_cache_trim(self._streaming_past_key_values)
                self._streaming_frame_idx += 1

                if torch.cuda.is_available():
                    vggt_end.record()
                    torch.cuda.synchronize()
                    self.last_vggt_ms = vggt_start.elapsed_time(vggt_end)

        tensor_features = []
        trimmed_h = trimmed_w = 0
        for idx in layer_indices:
            layer_output = aggregated_tokens_list[idx][:, :, patch_start_idx:, :]
            geo_feature, trimmed_h, trimmed_w = self._format_streaming_layer_features(
                layer_output,
                h_patch=h_patch,
                w_patch=w_patch,
                spatial_merge_size=spatial_merge_size,
                include_camera_token=include_camera_token,
                dtype=dtype,
            )
            tensor_features.append(geo_feature)

        if self._eval_frame_strict and not self._eval_projected_cache:
            # Buffer this frame's per-layer features on CPU (it was encoded with the
            # growing KV), then return the requested window gathered PER-FRAME. Each
            # buffered frame i == trajectory frame i (one frame encoded per step).
            if self._frame_feature_buffer is None:
                self._frame_feature_buffer = []
            self._frame_feature_buffer.append([t.detach().to("cpu") for t in tensor_features])
            n_buf = len(self._frame_feature_buffer)
            window = self._eval_window_indices
            window = [i for i in window if 0 <= i < n_buf] if window else [n_buf - 1]
            if not window:
                window = [n_buf - 1]
            gathered = []
            for layer_pos in range(len(tensor_features)):
                frames = [self._frame_feature_buffer[i][layer_pos] for i in window]
                gathered.append(torch.cat(frames, dim=0).to(tensor_features[layer_pos].device))
            tensor_features = gathered

        from qwen_vl.debug import vln_debug

        if vln_debug.is_enabled() and tensor_features:
            vln_debug.log_geometry_streaming(
                n_image=1,
                h_patch=h_patch,
                w_patch=w_patch,
                spatial_merge_size=spatial_merge_size,
                patch_tokens_shape=tuple(tensor_features[0].shape),
            )

        self._maybe_debug_geometry_layers(
            layer_indices=layer_indices,
            tensor_features=tensor_features,
            images=frame.unsqueeze(0),
            trimmed_h=trimmed_h,
            trimmed_w=trimmed_w,
            streaming=True,
        )

        return tensor_features

    def encode_layers_with_mode(
        self,
        images: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        spatial_merge_size: int = 1,
        include_camera_token: bool = False,
        streaming: bool = False,
        frame_strict: bool = False,
    ):
        if streaming:
            return self.encode_layers_streaming(
                images,
                layer_indices=layer_indices,
                spatial_merge_size=spatial_merge_size,
                include_camera_token=include_camera_token,
                frame_strict=frame_strict,
            )
        return self.encode_layers(
            images,
            layer_indices=layer_indices,
            spatial_merge_size=spatial_merge_size,
            include_camera_token=include_camera_token,
        )
    
    def get_feature_dim(self) -> int:
        """Get VGGT feature dimension."""
        return 2048  # VGGT feature dimension
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for compatibility."""
        return self.encode(images)

    def _apply_reference_frame_transform(self, images: torch.Tensor) -> torch.Tensor:
        """Apply reference frame transformation if needed."""
        if self.reference_frame != "first":
            return torch.flip(images, dims=(0,))
        return images
    
    def _apply_inverse_reference_frame_transform(self, features: torch.Tensor) -> torch.Tensor:
        """Apply inverse reference frame transformation if needed."""
        if self.reference_frame != "first":
            return torch.flip(features, dims=(0,))
        return features

    
    def load_model(self, model_path: str) -> None:
        """Load pretrained VGGT model."""
        from ..vggt.models.vggt import VGGT
        self._vggt_pretrained_path = model_path
        self.vggt = VGGT.from_pretrained(model_path, enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)
        self._depth_head_ready = False
        self._adageo_heads_ready = False
                
        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False

    def _ensure_adageo_heads(self) -> None:
        if self._adageo_heads_ready:
            return
        if (
            self.vggt.camera_head is not None
            and self.vggt.depth_head is not None
            and self.vggt.point_head is not None
        ):
            self._adageo_heads_ready = True
            self._depth_head_ready = True
            return

        from ..vggt.models.vggt import VGGT

        path = self._vggt_pretrained_path or "facebook/VGGT-1B"
        tmp = VGGT.from_pretrained(
            path,
            enable_camera=True,
            enable_point=True,
            enable_depth=True,
            enable_track=False,
        )
        device = next(self.vggt.parameters()).device
        self.vggt.camera_head = tmp.camera_head.to(device)
        self.vggt.depth_head = tmp.depth_head.to(device)
        self.vggt.point_head = tmp.point_head.to(device)
        for head in (self.vggt.camera_head, self.vggt.depth_head, self.vggt.point_head):
            head.eval()
            for param in head.parameters():
                param.requires_grad = False
        del tmp
        self._adageo_heads_ready = True
        self._depth_head_ready = True

    def _trim_camera_kv(self) -> None:
        """Bound the camera-head KV to a start+recent window of frames.

        `CameraHead.trunk_fn` appends one token per refinement iteration per trunk layer
        on every frame (4 x 4 at VGGT's defaults), and nothing ever pruned it: the cache
        grew linearly with episode length to ~50 MB at 400 steps, the last CUDA component
        still growing. Only the current and previous pose are ever consumed
        (`_streaming_frame_metadata` is trimmed to `[-2:]`), and only through the scalar
        `sigmoid(_pose_change_score(prev, cur).mean())`.

        Token counts are exactly frame-aligned, so slicing cuts on frame boundaries. The
        per-frame token count is read off the first frame rather than assuming
        `num_iterations`, so it stays correct if the head's defaults change.
        """
        cache = self._streaming_past_key_values_camera
        if cache is None or self._camera_kv_recent_frames <= 0:
            return
        if self._camera_kv_tokens_per_frame is None:
            first = next((kv for kv in cache if kv is not None), None)
            if first is None:
                return
            # Called right after the first frame's head pass, so this is one frame's worth.
            self._camera_kv_tokens_per_frame = int(first[0].shape[2])
            return
        per_frame = self._camera_kv_tokens_per_frame
        keep_start = max(0, self._camera_kv_start_frames) * per_frame
        keep_recent = self._camera_kv_recent_frames * per_frame
        for idx, kv in enumerate(cache):
            if kv is None:
                continue
            key, value = kv
            n = key.shape[2]
            if n <= keep_start + keep_recent:
                continue
            if keep_start > 0:
                key = torch.cat([key[:, :, :keep_start, :], key[:, :, n - keep_recent:, :]], dim=2)
                value = torch.cat([value[:, :, :keep_start, :], value[:, :, n - keep_recent:, :]], dim=2)
            else:
                key = key[:, :, n - keep_recent:, :].contiguous()
                value = value[:, :, n - keep_recent:, :].contiguous()
            cache[idx] = [key, value]

    def _append_adageo_frame_metadata(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
    ) -> None:
        pose_enc, self._streaming_past_key_values_camera = self.vggt.camera_head(
            aggregated_tokens_list,
            past_key_values_camera=self._streaming_past_key_values_camera,
            use_cache=True,
        )
        camera_pose = pose_enc[-1][:, 0, :]
        self._trim_camera_kv()

        depth, depth_conf = self.vggt.depth_head(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
        )
        _pts3d, pts3d_conf = self.vggt.point_head(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
        )
        keep_on_device = self.use_vln_segment_transition
        save = (lambda tensor: tensor.detach()) if keep_on_device else (lambda tensor: tensor.detach().cpu())
        self._streaming_frame_metadata.append(
            {
                "camera_pose": save(camera_pose),
                "depth": save(depth[:, 0]),
                "depth_conf": save(depth_conf[:, 0]),
                "conf": save(pts3d_conf[:, 0]),
            }
        )

    def _ensure_depth_head(self) -> None:
        if self._depth_head_ready:
            return
        if self.vggt.depth_head is not None:
            self._depth_head_ready = True
            return
        from ..vggt.models.vggt import VGGT

        path = self._vggt_pretrained_path or "facebook/VGGT-1B"
        tmp = VGGT.from_pretrained(
            path,
            enable_camera=False,
            enable_point=False,
            enable_depth=True,
            enable_track=False,
        )
        device = next(self.vggt.parameters()).device
        self.vggt.depth_head = tmp.depth_head.to(device)
        self.vggt.depth_head.eval()
        for param in self.vggt.depth_head.parameters():
            param.requires_grad = False
        self._depth_head_ready = True

    def predict_depth_maps(self, images: torch.Tensor) -> torch.Tensor:
        """
        Debug helper: VGGT DPT depth for [S,3,H,W] in [0,1].
        Returns [S, H, W] (full-sequence aggregator, not streaming KV).
        """
        self._ensure_depth_head()
        self.vggt.eval()
        images = self._apply_reference_frame_transform(images)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images[None])
                depth, _conf = self.vggt.depth_head(
                    aggregated_tokens_list,
                    images=images[None],
                    patch_start_idx=patch_start_idx,
                )
        return depth[0, :, 0]
