"""Opt-in, read-only instrumentation for the AdaGeoKV navigation scorer.

Enable with ``VLN_SCORE_PROBE=1``. Nothing in this module feeds back into scoring or
eviction; it only reads the tensors the scorer already produced. All reductions run on
the score device and are moved to host **once** per logged step, so the probe adds a
single synchronization instead of one per statistic.

Env knobs
---------
VLN_SCORE_PROBE        1/true to enable (default off -> every entry point is a no-op)
VLN_SCORE_PROBE_DIR    output directory for the JSONL trace (default ``vln_score_probe``)
VLN_SCORE_PROBE_EVERY  log one step out of N (default 10)
VLN_SCORE_PROBE_LAYERS comma-separated KV layer indices to inspect (default ``0,11,23``)
"""

from __future__ import annotations

import atexit
import json
import os
from typing import Dict, List, Optional

import torch

_QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)

# Must match the key order of `components` in compute_candidate_final_score.
_COMPONENT_NAMES = ("confidence", "instruction", "transition")

# The probe for the episode currently being recorded, flushed once at interpreter exit
# so the final episode is not lost (earlier episodes flush when the next one starts).
_ACTIVE_PROBE: "Optional[VLNScoreProbe]" = None


def _flush_active_probe() -> None:
    if _ACTIVE_PROBE is not None:
        try:
            _ACTIVE_PROBE.flush()
        except Exception:  # never let instrumentation break interpreter shutdown
            pass


atexit.register(_flush_active_probe)


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


def is_enabled() -> bool:
    return _flag("VLN_SCORE_PROBE")


class VLNScoreProbe:
    """Accumulates per-step score/retention statistics for one episode."""

    def __init__(self, num_segments: int, instruction: str = "") -> None:
        self.enabled = is_enabled()
        if not self.enabled:
            return
        self.every = max(1, int(os.environ.get("VLN_SCORE_PROBE_EVERY", "10")))
        self.layers = [
            int(x) for x in os.environ.get("VLN_SCORE_PROBE_LAYERS", "0,11,23").split(",") if x.strip()
        ]
        self.out_dir = os.environ.get("VLN_SCORE_PROBE_DIR", "vln_score_probe")
        os.makedirs(self.out_dir, exist_ok=True)
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        self.path = os.path.join(self.out_dir, f"vln_score_probe_rank{rank}.jsonl")
        self.num_segments = int(num_segments)
        self.instruction = instruction
        self.episode = -1
        self._records: List[dict] = []
        # A probe is built per episode; registering atexit here would accumulate one
        # handler (and one live probe) per episode. Track the current one instead and
        # flush it from a single module-level handler.
        global _ACTIVE_PROBE
        _ACTIVE_PROBE = self

    # -- episode lifecycle -------------------------------------------------
    def start_episode(self, episode_id) -> None:
        if not self.enabled:
            return
        self.flush()
        self.episode = episode_id

    def flush(self) -> None:
        if not self.enabled or not self._records:
            return
        with open(self.path, "a") as f:
            for record in self._records:
                f.write(json.dumps(record) + "\n")
        self._records = []

    # -- per-frame hooks ---------------------------------------------------
    def should_log(self, frame_id: int) -> bool:
        return self.enabled and frame_id % self.every == 0

    @torch.no_grad()
    def log_frame_scores(
        self,
        frame_id: int,
        *,
        confidence: torch.Tensor,
        instruction: torch.Tensor,
        transition_scalar: torch.Tensor,
        anchor: torch.Tensor,
        final: torch.Tensor,
        best_segment: torch.Tensor,
    ) -> None:
        """Record the newly scored frame. One host transfer for the whole record."""
        if not self.should_log(frame_id):
            return
        device = final.device
        qs = torch.tensor(_QUANTILES, device=device, dtype=torch.float32)
        stack = torch.stack(
            [confidence.float(), instruction.float(), anchor.float(), final.float()]
        )  # [4, n_tokens]
        quants = torch.quantile(stack, qs, dim=1)  # [n_q, 4]
        scalars = torch.stack(
            [
                transition_scalar.float().reshape(()),
                stack.mean(dim=1)[3],
                stack.std(dim=1)[3],
                stack.std(dim=1)[0],  # confidence std
                stack.std(dim=1)[1],  # instruction std
                stack.std(dim=1)[2],  # anchor std
            ]
        )
        hist = torch.bincount(
            (best_segment.long() + 1).clamp_min(0), minlength=self.num_segments + 1
        ).float()
        # Single device->host transfer for the whole record.
        packed = torch.cat([quants.reshape(-1), scalars, hist]).cpu().tolist()
        n_q = len(_QUANTILES)
        quant_flat = packed[: n_q * 4]
        scalar_flat = packed[n_q * 4 : n_q * 4 + 6]
        hist_flat = packed[n_q * 4 + 6 :]
        names = ("confidence", "instruction", "transition_anchor", "final")
        self._records.append(
            {
                "kind": "frame",
                "episode": self.episode,
                "frame_id": int(frame_id),
                "quantiles": list(_QUANTILES),
                "dist": {
                    name: [quant_flat[i * 4 + j] for i in range(n_q)]
                    for j, name in enumerate(names)
                },
                "transition_scalar": scalar_flat[0],
                "final_mean": scalar_flat[1],
                "final_std": scalar_flat[2],
                "confidence_std": scalar_flat[3],
                "instruction_std": scalar_flat[4],
                "anchor_std": scalar_flat[5],
                "best_segment_hist": {
                    str(i - 1): int(v) for i, v in enumerate(hist_flat) if v > 0
                },
            }
        )

    @torch.no_grad()
    def log_layer_retention(
        self,
        frame_id: int,
        layer_idx: int,
        metadata,
        budget: int,
        candidate_component_std: Optional[torch.Tensor] = None,
        candidate_variance_share: Optional[torch.Tensor] = None,
        candidate_true_share: Optional[torch.Tensor] = None,
        candidate_component_corr: Optional[torch.Tensor] = None,
    ) -> None:
        """Record what survived eviction in ``layer_idx``."""
        if not self.should_log(frame_id) or layer_idx not in self.layers:
            return
        device = metadata.final_score.device
        n = metadata.final_score.numel()
        age = (frame_id - metadata.frame_id.long()).float()
        is_special = metadata.is_special
        seg = metadata.best_segment_id
        seg_hist = torch.bincount(
            (seg.long() + 1).clamp_min(0), minlength=self.num_segments + 1
        ).float()
        qs = torch.tensor(_QUANTILES, device=device, dtype=torch.float32)
        age_q = torch.quantile(age, qs)
        scalars = torch.stack(
            [
                torch.tensor(float(n), device=device),
                is_special.float().sum(),
                metadata.frame_id.unique().numel() * torch.ones((), device=device),
                metadata.final_score.float().mean(),
                metadata.final_score.float().std(),
                metadata.final_score.float().unique().numel() * torch.ones((), device=device),
                metadata.transition_score.float().mean(),
                metadata.instruction_score.float().mean(),
                age.mean(),
            ]
        )
        n_comp_t = candidate_component_std.numel()
        diagnostics = torch.cat([
            candidate_component_std.float(),
            candidate_variance_share.float(),
            candidate_true_share.float()
            if candidate_true_share is not None
            else torch.zeros(n_comp_t, device=device),
            candidate_component_corr.float().reshape(-1)
            if candidate_component_corr is not None
            else torch.zeros(n_comp_t * n_comp_t, device=device),
        ])
        packed = torch.cat([age_q, scalars, diagnostics, seg_hist]).cpu().tolist()
        n_q = len(_QUANTILES)
        age_flat = packed[:n_q]
        scalar_flat = packed[n_q : n_q + 9]
        n_comp = candidate_component_std.numel()
        diagnostic_flat = packed[n_q + 9 : n_q + 9 + 2 * n_comp]
        true_share_flat = packed[n_q + 9 + 2 * n_comp : n_q + 9 + 3 * n_comp]
        corr_end = n_q + 9 + 3 * n_comp + n_comp * n_comp
        corr_flat = packed[n_q + 9 + 3 * n_comp : corr_end]
        seg_flat = packed[corr_end:]
        # Derived from the tensor the scorer handed over, so adding a score component
        # does not silently shift the seg-histogram slice by one.
        component_names = _COMPONENT_NAMES[:n_comp]
        self._records.append(
            {
                "kind": "layer",
                "episode": self.episode,
                "frame_id": int(frame_id),
                "layer": int(layer_idx),
                "budget": int(budget),
                "retained_tokens": int(scalar_flat[0]),
                "retained_special": int(scalar_flat[1]),
                "special_fraction": scalar_flat[1] / max(1.0, scalar_flat[0]),
                "unique_frames": int(scalar_flat[2]),
                "final_mean": scalar_flat[3],
                "final_std": scalar_flat[4],
                "distinct_score_levels": int(scalar_flat[5]),
                "transition_mean": scalar_flat[6],
                "instruction_mean": scalar_flat[7],
                "age_mean": scalar_flat[8],
                "candidate_component_std": {
                    name: diagnostic_flat[idx] for idx, name in enumerate(component_names)
                },
                "candidate_variance_proxy_share": {
                    name: diagnostic_flat[idx + n_comp] for idx, name in enumerate(component_names)
                },
                # Covariance-aware: Var(S) = sum_i Cov(w_i z_i, S). Sums to 1; a component
                # that fights the others goes negative, which the proxy share cannot show.
                "candidate_variance_true_share": {
                    name: true_share_flat[idx] for idx, name in enumerate(component_names)
                },
                "candidate_component_corr": {
                    f"{a}|{b}": corr_flat[i * n_comp + j]
                    for i, a in enumerate(component_names)
                    for j, b in enumerate(component_names)
                    if j > i
                },
                "quantiles": list(_QUANTILES),
                "age_quantiles": age_flat,
                "retained_per_segment": {
                    str(i - 1): int(v) for i, v in enumerate(seg_flat) if v > 0
                },
            }
        )


def summarize(path: str) -> Dict[str, object]:
    """Offline helper: reduce a probe JSONL into per-component summary statistics."""
    frames, layers = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            (frames if rec["kind"] == "frame" else layers).append(rec)

    def _mean(rows, key):
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    out: Dict[str, object] = {"n_frame_records": len(frames), "n_layer_records": len(layers)}
    if frames:
        out["component_means"] = {
            "transition_scalar": _mean(frames, "transition_scalar"),
            "final_mean": _mean(frames, "final_mean"),
        }
        out["within_frame_std"] = {
            "final": _mean(frames, "final_std"),
            "confidence": _mean(frames, "confidence_std"),
            "instruction": _mean(frames, "instruction_std"),
            "transition_anchor": _mean(frames, "anchor_std"),
        }
        seg_total: Dict[str, int] = {}
        for r in frames:
            for k, v in r["best_segment_hist"].items():
                seg_total[k] = seg_total.get(k, 0) + v
        total = sum(seg_total.values()) or 1
        out["best_segment_share"] = {k: v / total for k, v in sorted(seg_total.items())}
    if layers:
        out["retention"] = {
            "special_fraction": _mean(layers, "special_fraction"),
            "unique_frames": _mean(layers, "unique_frames"),
            "age_mean": _mean(layers, "age_mean"),
            "distinct_score_levels": _mean(layers, "distinct_score_levels"),
            "retained_tokens": _mean(layers, "retained_tokens"),
        }
        # Read the component set off the records rather than assuming it: a run with an
        # extra term would otherwise be summarised as if that term did not exist.
        component_names = tuple(layers[0]["candidate_component_std"])
        out["candidate_component_std"] = {
            name: sum(r["candidate_component_std"][name] for r in layers) / len(layers)
            for name in component_names
        }
        out["candidate_variance_proxy_share"] = {
            name: sum(r["candidate_variance_proxy_share"][name] for r in layers) / len(layers)
            for name in component_names
        }
    return out


if __name__ == "__main__":
    import sys

    print(json.dumps(summarize(sys.argv[1]), indent=2))
