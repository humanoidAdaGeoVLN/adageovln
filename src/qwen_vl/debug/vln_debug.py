"""Rank-0 VLN training debug helpers (prints + optional image dumps)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from PIL import Image

_enabled = False
_save_dir: Optional[Path] = None
_max_samples = 5
_save_interval = 100
_save_geo_layers = True
_save_depth = False
_local_rank = 0
_global_step = 0
_saved_this_step = False

_dataset_count = 0
_batch_count = 0


def configure(
    *,
    enabled: bool = False,
    save_dir: str = "",
    max_samples: int = 5,
    max_steps: int = 5,
    save_interval: int = 100,
    save_geo_layers: bool = True,
    save_depth: bool = False,
    local_rank: int = 0,
) -> None:
    global _enabled, _save_dir, _max_samples, _save_interval
    global _save_geo_layers, _save_depth, _local_rank
    _enabled = enabled
    _local_rank = local_rank
    _max_samples = max_samples
    _save_interval = max(1, save_interval)
    _save_geo_layers = save_geo_layers
    _save_depth = save_depth
    if enabled and save_dir:
        _save_dir = Path(save_dir)
        if _is_rank0():
            _save_dir.mkdir(parents=True, exist_ok=True)
    else:
        _save_dir = None


def set_global_step(step: int) -> None:
    global _global_step, _saved_this_step
    _global_step = max(0, int(step))
    _saved_this_step = False


def mark_step_debugged() -> None:
    global _saved_this_step
    _saved_this_step = True


def get_global_step() -> int:
    return _global_step


def should_debug_training_step() -> bool:
    """Print / save only every N training steps (not every micro-batch or epoch)."""
    if not _enabled:
        return False
    if _global_step <= 0 or _saved_this_step:
        return False
    return _global_step % _save_interval == 0


def is_enabled() -> bool:
    return _enabled


def _is_rank0() -> bool:
    return _local_rank == 0


def log(msg: str) -> None:
    if _enabled and _is_rank0():
        print(f"[VLN_DEBUG] {msg}", flush=True)


def _decode_labels(tokenizer, labels: torch.Tensor) -> str:
    mask = labels != -100
    if not mask.any():
        return "<no trainable tokens>"
    ids = labels[mask].tolist()
    return tokenizer.decode(ids, skip_special_tokens=False).strip()


def _save_chw_tensor(tensor: torch.Tensor, path: Path) -> None:
    from torchvision.utils import save_image

    t = tensor.detach().cpu().float()
    if t.dim() == 3:
        save_image(t.clamp(0, 1), str(path))


def should_debug_dataset() -> bool:
    return should_debug_training_step() and _dataset_count < _max_samples


def mark_dataset_logged() -> None:
    global _dataset_count
    _dataset_count += 1


def should_debug_batch() -> bool:
    return should_debug_training_step() and _batch_count < _max_samples


def mark_batch_logged() -> None:
    global _batch_count
    _batch_count += 1


def should_save_depth() -> bool:
    return should_debug_training_step() and _save_depth


def should_save_geo_layers() -> bool:
    return should_debug_training_step() and _save_geo_layers


def log_dataset_sample(
    *,
    sample_idx: int,
    sample_id: Union[str, int],
    raw_images: Sequence[Image.Image],
    grid_thw: Sequence[torch.Tensor],
    geometry_encoder_inputs: Sequence[torch.Tensor],
    merge_size: int,
    labels: torch.Tensor,
    tokenizer,
) -> None:
    if not should_debug_dataset():
        return

    tag = f"step_{_global_step:06d}_sample_{_dataset_count:03d}_idx{sample_idx}"
    n_frames = len(raw_images)
    tokens_per_image = [
        int(thw.prod().item()) // (merge_size * merge_size) for thw in grid_thw
    ]
    geo_shapes = [tuple(g.shape) for g in geometry_encoder_inputs]

    log(
        f"step={_global_step} dataset {tag} id={sample_id} frames={n_frames} "
        f"grid_thw={[tuple(t.tolist()) for t in grid_thw]} "
        f"tokens_per_image={tokens_per_image} total_vision_tokens={sum(tokens_per_image)} "
        f"geo_shapes={geo_shapes} label={_decode_labels(tokenizer, labels)!r}"
    )

    if _save_dir is not None and _is_rank0():
        out = _save_dir / "frames" / tag
        out.mkdir(parents=True, exist_ok=True)
        for fi, pil_img in enumerate(raw_images):
            pil_img.save(out / f"frame_{fi:02d}_raw.png")
        for fi, geo in enumerate(geometry_encoder_inputs):
            _save_chw_tensor(geo, out / f"frame_{fi:02d}_vggt_644.png")
        log(f"saved frame images to {out}")

    mark_dataset_logged()


def log_collator_batch(
    *,
    geometry_encoder_inputs: List[torch.Tensor],
    image_grid_thw: torch.Tensor,
    merge_size: int,
    labels: torch.Tensor,
    tokenizer,
) -> None:
    if not should_debug_batch():
        return

    geo = geometry_encoder_inputs[0]
    tokens_per_image = (
        image_grid_thw.prod(dim=-1) // (merge_size * merge_size)
    ).tolist()
    n_frames = int(geo.shape[0]) if geo is not None and geo.dim() >= 1 else 0
    total_vision = int(sum(tokens_per_image))

    log(
        f"step={_global_step} collator batch#{_batch_count} geometry={tuple(geo.shape)} "
        f"frames={n_frames} image_grid_thw={image_grid_thw.tolist()} "
        f"tokens_per_image={tokens_per_image} total_vision_tokens={total_vision} "
        f"(expected tiling_factor ≈ {n_frames}) "
        f"label={_decode_labels(tokenizer, labels[0])!r}"
    )

    mark_batch_logged()


def save_geometry_encoder_layers(
    *,
    layer_indices: Sequence[int],
    tensor_features: Sequence[torch.Tensor],
    trimmed_h: int,
    trimmed_w: int,
    input_images: Optional[torch.Tensor] = None,
    streaming: bool = False,
) -> None:
    """Save L2-norm heatmaps for geometry_encoder_layers (e.g. 11, 17, 23)."""
    if not should_save_geo_layers() or _save_dir is None or not _is_rank0():
        return

    from qwen_vl.debug.geo_viz import save_layer_heatmaps

    tag = f"step_{_global_step:06d}"
    mode = "streaming_last_frame" if streaming else "batch"
    out = _save_dir / "geometry_layers" / tag
    save_layer_heatmaps(
        out,
        layer_indices=layer_indices,
        tensor_features=tensor_features,
        trimmed_h=trimmed_h,
        trimmed_w=trimmed_w,
        input_images=input_images,
        frame_idx=-1 if streaming else 0,
    )
    log(
        f"step={_global_step} saved geometry_encoder_layers {list(layer_indices)} "
        f"heatmaps ({mode}) -> {out}"
    )
    mark_step_debugged()


def save_vggt_depth_maps(encoder, images: torch.Tensor) -> None:
    """Run VGGT depth head (debug only) and save colormapped depth PNGs."""
    if not should_save_depth() or _save_dir is None or not _is_rank0():
        return

    from qwen_vl.debug.geo_viz import save_depth_maps

    depth = encoder.predict_depth_maps(images)
    tag = f"step_{_global_step:06d}"
    out = _save_dir / "depth" / tag
    save_depth_maps(out, depth_maps=depth, input_images=images)
    log(f"step={_global_step} saved VGGT depth maps (S={images.shape[0]}) -> {out}")
    mark_step_debugged()


def log_geometry_streaming(
    *,
    n_image: int,
    h_patch: int,
    w_patch: int,
    spatial_merge_size: int,
    patch_tokens_shape: tuple,
) -> None:
    if not should_debug_training_step():
        return
    n_geo_merged = patch_tokens_shape[1] if len(patch_tokens_shape) > 1 else 0
    log(
        f"step={_global_step} streaming VGGT S={n_image} h_patch={h_patch} w_patch={w_patch} "
        f"merge={spatial_merge_size} geo_tokens={patch_tokens_shape} "
        f"geo_merged={n_geo_merged}"
    )


def log_fusion(
    *,
    layer_idx: int,
    vision_tokens_shape: tuple,
    geo_shape: tuple,
    tiling_factor: int,
) -> None:
    if not should_debug_training_step():
        return
    log(
        f"step={_global_step} fusion layer={layer_idx} vision_tokens={vision_tokens_shape} "
        f"geo={geo_shape} tiling_factor={tiling_factor}"
    )
