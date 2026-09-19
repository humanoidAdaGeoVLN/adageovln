"""Visualize VGGT geometry_encoder_layers outputs as spatial heatmaps / depth maps."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image


def _normalize01(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _colorize(gray01: np.ndarray) -> np.ndarray:
    """Map [H,W] in [0,1] to RGB uint8 using turbo colormap."""
    gray01 = np.clip(gray01, 0.0, 1.0)
    try:
        import matplotlib.cm as cm

        rgba = cm.get_cmap("turbo")(gray01)
        return (rgba[..., :3] * 255).astype(np.uint8)
    except ImportError:
        g = (gray01 * 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)


def tokens_to_heatmap(
    patch_tokens: torch.Tensor,
    trimmed_h: int,
    trimmed_w: int,
) -> np.ndarray:
    """
    L2-norm activation map from patch tokens [1, trimmed_h*trimmed_w, D] or [seq, D].
    Returns RGB uint8 [trimmed_h, trimmed_w, 3].
    """
    tokens = patch_tokens.detach().float().cpu()
    if tokens.dim() == 3:
        tokens = tokens[-1]  # last frame (streaming / VLN current view)
    norms = tokens.norm(dim=-1).numpy()
    grid = norms.reshape(trimmed_h, trimmed_w)
    return _colorize(_normalize01(grid))


def depth_tensor_to_heatmap(depth: torch.Tensor) -> np.ndarray:
    """VGGT depth [H,W] or [1,H,W] -> RGB uint8 heatmap (closer=warm)."""
    d = depth.detach().float().cpu()
    while d.dim() > 2:
        d = d.squeeze(0)
    valid = d[d > 0]
    if valid.numel() == 0:
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    # invert so nearer objects are brighter (depth-style)
    inv = 1.0 / (d.numpy() + 1e-6)
    inv[d.numpy() <= 0] = 0
    return _colorize(_normalize01(inv))


def _chw_to_pil_rgb(tensor: torch.Tensor) -> Image.Image:
    t = tensor.detach().float().cpu().clamp(0, 1)
    if t.dim() == 3:
        arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise ValueError(f"Expected CHW image tensor, got {tuple(tensor.shape)}")


def save_layer_heatmaps(
    out_dir: Path,
    *,
    layer_indices: Sequence[int],
    tensor_features: Sequence[torch.Tensor],
    trimmed_h: int,
    trimmed_w: int,
    input_images: Optional[torch.Tensor] = None,
    frame_idx: int = -1,
) -> None:
    """
    Save activation heatmaps for each geometry_encoder_layer.

    Files:
      layer_XX_heatmap.png       — L2 norm of patch tokens (pseudo-depth / saliency)
      layer_XX_overlay.png       — heatmap resized over input RGB (if images given)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = None
    if input_images is not None and input_images.numel() > 0:
        fi = frame_idx if frame_idx >= 0 else input_images.shape[0] - 1
        fi = min(fi, input_images.shape[0] - 1)
        rgb = _chw_to_pil_rgb(input_images[fi])

    for layer_idx, feat in zip(layer_indices, tensor_features):
        heat_rgb = tokens_to_heatmap(feat, trimmed_h, trimmed_w)
        heat_img = Image.fromarray(heat_rgb)
        tag = f"layer_{layer_idx:02d}"
        heat_img.save(out_dir / f"{tag}_heatmap.png")

        if rgb is not None:
            heat_up = heat_img.resize(rgb.size, Image.Resampling.BILINEAR)
            overlay = Image.blend(rgb, heat_up, alpha=0.45)
            overlay.save(out_dir / f"{tag}_overlay.png")


def save_depth_maps(
    out_dir: Path,
    *,
    depth_maps: torch.Tensor,
    input_images: Optional[torch.Tensor] = None,
) -> None:
    """
    Save VGGT DPT depth predictions.

    depth_maps: [S, H, W] or [S, 1, H, W]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_maps = depth_maps.detach().float().cpu()
    if depth_maps.dim() == 4:
        depth_maps = depth_maps.squeeze(-1) if depth_maps.shape[-1] == 1 else depth_maps.squeeze(1)

    for fi in range(depth_maps.shape[0]):
        heat = depth_tensor_to_heatmap(depth_maps[fi])
        Image.fromarray(heat).save(out_dir / f"frame_{fi:02d}_depth.png")
        if input_images is not None and fi < input_images.shape[0]:
            rgb = _chw_to_pil_rgb(input_images[fi])
            heat_img = Image.fromarray(heat).resize(rgb.size, Image.Resampling.BILINEAR)
            Image.blend(rgb, heat_img, alpha=0.5).save(out_dir / f"frame_{fi:02d}_depth_overlay.png")
