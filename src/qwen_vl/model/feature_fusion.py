"""Feature fusion modules for combining 2D and 3D features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class FeatureFusionConfig:
    """Configuration for feature fusion."""
    fusion_method: str = "add"  # "add", "concat", "gated", "weighted", "cross_attention"
    hidden_size: int = 3584
    num_heads: int = 8
    dropout: float = 0.1
    num_layers: int = 1

@dataclass
class MultiLayerFeatureFusionConfig:
    """Configuration for multi layer feature fusion."""
    fusion_method: str = "deepstack_vision_add"  # "add"
    vis_hidden_size: int = 1280
    geo_hidden_size: int = 2048
    lang_hidden_size: int = 2048
    geometry_fusion_layers: List[int] = None
    pos_encoding_type: str = "none"  # "none" | "rope2d" | "sincos2d"
    spatial_merge_size: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    include_camera_token: bool = False
    fusion_scale: float = 1.0
    importance_gate: bool = False  # Step 2': per-position sigmoid gate on the geometry delta (env FUSION_IMPORTANCE_GATE overrides)
    learnable_scale: bool = False  # Step 2'': learnable per-layer scale on the geometry delta (replaces fixed fusion_scale; env FUSION_LEARNABLE_SCALE overrides)
    spatial_bias: bool = False  # Step 4: GeoThinker spatial-distance bias on the SGF cross-attention (env FUSION_SPATIAL_BIAS overrides)


class LearnableScale(nn.Module):
    """Learnable scalar multiplier on the geometry delta (Step 2'').

    Replaces the fixed `fusion_scale`. Initialized to `init_value` so training
    starts identical to the fixed-scale baseline, then the model learns a
    per-fusion-layer geometry strength.
    """

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class SGFCrossAttentionBlock(nn.Module):
    """GeoThinker-style Spatial-Grounded Fusion for the language/decoder path.

    Frame-strict cross-attention (Q = vision tokens, K/V = geometry tokens) with:
      - Step 2  : the cross-attention itself (SDPA, so bias terms can be added).
      - Step 3  : optional importance gate  -> log(importance) added to attn logits.
      - Step 4  : optional spatial-distance bias on half the heads (gated).
    Output is scaled by tanh(gate) with gate init 0 -> exact no-op at step 0.
    Returns the geometry DELTA; the caller adds it to the vision tokens.
    """

    def __init__(self, hidden_size, num_heads=8, dropout=0.1,
                 use_importance_gate=False, use_spatial_bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.use_importance_gate = use_importance_gate
        self.use_spatial_bias = use_spatial_bias

        self.q_norm = nn.LayerNorm(hidden_size)
        self.kv_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.0))  # tanh(gate)=0 -> no-op at init

        if use_importance_gate:
            self.importance_net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid(),
            )
        if use_spatial_bias:
            # One sigmoid gate per (half) head, conditioned on the geometry token.
            self.bias_gate = nn.Sequential(
                nn.Linear(hidden_size, max(1, num_heads // 2)),
                nn.Sigmoid(),
            )

    @staticmethod
    def _spatial_bias(h_feat, w_feat, device, dtype):
        # -normalized pairwise euclidean distance between patch grid coords: [S, S]
        y = torch.arange(h_feat, device=device, dtype=dtype)
        x = torch.arange(w_feat, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([gy.flatten(), gx.flatten()], dim=-1)
        dist = torch.cdist(coords, coords, p=2)
        diag = (float(h_feat) ** 2 + float(w_feat) ** 2) ** 0.5 + 1e-6
        return -dist / diag

    def forward(self, q_in, kv_in, grid_hw=None):
        # q_in / kv_in: [B, S, D]  (B = frames, S = merged tokens per frame)
        q_in = self.q_norm(q_in)
        kv_in = self.kv_norm(kv_in)
        B, S, D = q_in.shape
        q = self.q_proj(q_in).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_bias = None
        if self.use_spatial_bias and grid_hw is not None:
            h_feat, w_feat = grid_hw
            if h_feat * w_feat == S:
                sb = self._spatial_bias(h_feat, w_feat, q.device, q.dtype)  # [S, S]
                gate = self.bias_gate(kv_in).transpose(1, 2).unsqueeze(-1)  # [B, H/2, S, 1]
                bias = torch.zeros(B, self.num_heads, S, S, device=q.device, dtype=q.dtype)
                half = max(1, self.num_heads // 2)
                bias[:, :half] = sb * gate
                attn_bias = bias
        if self.use_importance_gate:
            imp = self.importance_net(kv_in)  # [B, S, 1] from geometry (background suppression)
            imp_logit = torch.log(imp + 0.1).view(B, 1, 1, S)  # additive key-side bias
            attn_bias = imp_logit if attn_bias is None else attn_bias + imp_logit

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)
        return torch.tanh(self.gate) * out


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block with position encoding, MLP and residual connections."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Layer norms
        self.norm1_query = nn.LayerNorm(hidden_size)
        self.norm1_key = nn.LayerNorm(hidden_size)
        self.norm1_value = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        features_2d: torch.Tensor,     # [B, Nq, C]
        features_3d: torch.Tensor,     # [B, Nk, C]
        pos_embed_query: Optional[torch.Tensor] = None,  # [B, Nq, C]
        pos_embed_key: Optional[torch.Tensor] = None,    # [B, Nk, C]
    ) -> torch.Tensor:
        # Normalize features (LayerNorm under autocast returns fp32; cast back to original dtype)
        query = self.norm1_query(features_2d)
        key = self.norm1_key(features_3d)
        value = self.norm1_value(features_3d)

        if query.dtype != features_2d.dtype:
            query = query.to(features_2d.dtype)
        if key.dtype != features_3d.dtype:
            key = key.to(features_3d.dtype)
        if value.dtype != features_3d.dtype:
            value = value.to(features_3d.dtype)

        # Add externally-computed positional embeddings if provided
        if pos_embed_query is not None:
            query = query + pos_embed_query.to(dtype=query.dtype, device=query.device)
        if pos_embed_key is not None:
            key = key + pos_embed_key.to(dtype=key.dtype, device=key.device)
            
        # Cross-attention: 2D features as query, 3D features as key/value
        attn_output, _ = self.cross_attention(query, key, value)
        
        # First residual connection
        x = features_2d + attn_output
        
        # MLP with second residual connection
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        
        return x

class FeatureFusionModule(nn.Module):
    """Enhanced feature fusion module with multiple fusion strategies."""
    
    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_method = config.fusion_method
        self.hidden_size = config.hidden_size
        
        self._build_fusion_layers()
    
    def _build_fusion_layers(self):
        """Build fusion layers based on method."""
        if self.config.fusion_method == "concat":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
            
        elif self.config.fusion_method == "cross_attention":
            self.cross_attn_blocks = nn.ModuleList([
                CrossAttentionBlock(
                    self.hidden_size, 
                    self.config.num_heads, 
                    self.config.dropout
                ) 
                for _ in range(self.config.num_layers)
            ])

        elif self.config.fusion_method == "gated":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.gate_projection = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Sigmoid()
            )
            
        elif self.config.fusion_method == "weighted":
            self.weight_2d = nn.Parameter(torch.tensor(0.5))
            self.weight_3d = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> torch.Tensor:
        """
        Fuse 2D and 3D features.
        
        Args:
            features_2d: 2D image features
            features_3d: 3D geometry features
        Returns:
            Fused features
        """

        _, h_grid, w_grid, _ = features_3d.shape
        if self.fusion_method == "add":
            return features_2d + features_3d
            
        elif self.fusion_method == "concat":
            features_2d = self.norm1(features_2d)
            features_3d = self.norm2(features_3d)
            concat_features = torch.cat([features_2d, features_3d], dim=-1)
            return self.projection(concat_features)
            
        elif self.fusion_method == "cross_attention":
            features_2d = features_2d.view(features_2d.size(0), -1, self.hidden_size)  # Flatten spatial dimensions
            features_3d = features_3d.view(features_3d.size(0), -1, self.hidden_size)
            x = features_2d
            for block in self.cross_attn_blocks:
                x = block(x, features_3d, h_grid, w_grid)
            return x
            
        elif self.fusion_method == "gated":
            features_2d = self.norm1(features_2d)
            features_3d = self.norm2(features_3d)
            concat_features = torch.cat([features_2d, features_3d], dim=-1)
            gate = self.gate_projection(concat_features)
            return gate * features_2d + (1 - gate) * features_3d
            
        elif self.fusion_method == "weighted":
            # Normalize weights to sum to 1
            weight_sum = self.weight_2d + self.weight_3d
            norm_weight_2d = self.weight_2d / weight_sum
            norm_weight_3d = self.weight_3d / weight_sum
            return norm_weight_2d * features_2d + norm_weight_3d * features_3d
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


class GeometryFeatureMerger(nn.Module):
    """Unified merger for geometry features from different encoders.
    
    Supports different merger types:
    - "mlp": MLP-based feature transformation with spatial merging
    - "avg": Average pooling across spatial merge dimensions
    - "attention": Attention-based merger (not implemented yet)
    """
    
    def __init__(self, output_dim: int, hidden_dim: int, context_dim: int, 
                 spatial_merge_size: int = 2, merger_type: str = "mlp"):
        super().__init__()
        self.merger_type = merger_type
        self.input_dim = context_dim * (spatial_merge_size ** 2)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.merge_size = spatial_merge_size
        
        if merger_type == "mlp":
            # Import here to avoid circular import
            try:
                from .modeling_qwen2_5_vl import Qwen2RMSNorm
            except ImportError:
                # Fallback to standard LayerNorm if Qwen2RMSNorm not available
                Qwen2RMSNorm = nn.LayerNorm
                
            self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif merger_type == "avg":
            self.mlp = nn.Sequential(
                nn.Linear(context_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif merger_type == "attention":
            # Add attention-based merger for future extensibility
            raise NotImplementedError("Attention merger not implemented yet")
        else:
            raise ValueError(f"Unknown merger type: {merger_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the merger."""

        n_image, h_patch, w_patch, dim = x.shape
        x = x[:, :h_patch // self.merge_size * self.merge_size, :w_patch // self.merge_size*self.merge_size , :]
        x = x.reshape(n_image, h_patch // self.merge_size, self.merge_size, w_patch // self.merge_size, self.merge_size, dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        if self.merger_type == "mlp":
            x = self.mlp(self.ln_q(x).view(-1, self.input_dim))
        elif self.merger_type == "avg":
            # Average pooling across spatial merge dimensions
            x = x.mean(dim=(3, 4))  # Average over the merge_size dimensions
            x = x.view(-1, dim)  # Flatten for projection
            x = self.mlp(x)
        else:
            raise NotImplementedError(f"Merger type {self.merger_type} not implemented")
        x = x.reshape(n_image, h_patch // self.merge_size, w_patch // self.merge_size, -1)
        return x

class MultiLayerFeatureFusionModule(nn.Module):
    """Enhanced multi feature fusion module with multiple fusion strategies."""
    
    def __init__(self, config: MultiLayerFeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_method = config.fusion_method
        self.vis_hidden_size = config.vis_hidden_size
        self.geo_hidden_size = config.geo_hidden_size
        self.lang_hidden_size = config.lang_hidden_size
        self.geometry_fusion_layers = config.geometry_fusion_layers

        import os as _os
        _env_scale = _os.environ.get("GEOMETRY_FUSION_SCALE")
        self.fusion_scale = float(_env_scale) if _env_scale is not None else float(getattr(config, "fusion_scale", 1.0))
        if self.fusion_scale != 1.0:
            print(f"[FeatureFusion] geometry fusion delta scaled by {self.fusion_scale}")

        # Step 2': per-position importance gate on the geometry delta (GeoThinker-style
        # background suppression, applied to the ADD path). Env FUSION_IMPORTANCE_GATE
        # overrides config.importance_gate. Must be resolved before building layers.
        _env_ig = _os.environ.get("FUSION_IMPORTANCE_GATE")
        self.use_importance_gate = (
            _env_ig.lower() in ("1", "true", "yes")
            if _env_ig is not None
            else bool(getattr(config, "importance_gate", False))
        )
        if self.use_importance_gate:
            print("[FeatureFusion] importance gate ENABLED on geometry delta")

        # Step 2'': learnable per-layer scale replacing the fixed fusion_scale.
        # Env FUSION_LEARNABLE_SCALE overrides config.learnable_scale.
        _env_ls = _os.environ.get("FUSION_LEARNABLE_SCALE")
        self.use_learnable_scale = (
            _env_ls.lower() in ("1", "true", "yes")
            if _env_ls is not None
            else bool(getattr(config, "learnable_scale", False))
        )
        if self.use_learnable_scale:
            print(f"[FeatureFusion] learnable geometry scale ENABLED (init {self.fusion_scale})")

        # Step 4: GeoThinker spatial-distance bias (only used by the SGF cross-attn
        # operator). Env FUSION_SPATIAL_BIAS overrides config.spatial_bias.
        _env_sb = _os.environ.get("FUSION_SPATIAL_BIAS")
        self.use_spatial_bias = (
            _env_sb.lower() in ("1", "true", "yes")
            if _env_sb is not None
            else bool(getattr(config, "spatial_bias", False))
        )
        if self.use_spatial_bias:
            print("[FeatureFusion] spatial-distance bias ENABLED on SGF cross-attention")

        # Allow multiple fusion blocks per decoder layer (support duplicate layer indices)
        self.fusion_layers = nn.ModuleDict()
        for layer_num in self.geometry_fusion_layers:
            layer_key = str(layer_num)
            if layer_key not in self.fusion_layers:
                self.fusion_layers[layer_key] = nn.ModuleList()
            self.fusion_layers[layer_key].append(self._build_fusion_layer())
        self.reset_residual_branches_to_noop()
    
    def get_fusion_layer(self, layer_idx: int):
        fusion_layer = self.fusion_layers[str(layer_idx)]
        # Always return a ModuleList for consistent downstream handling
        if isinstance(fusion_layer, nn.ModuleList):
            return fusion_layer
        return nn.ModuleList([fusion_layer])

    def _build_fusion_layer(self):
        """Build fusion layers based on method."""
        fusion_layer = None

        if self.config.fusion_method == "deepstack_vision_add":
            # Import here to avoid circular import
            try:
                from .modeling_qwen2_5_vl import Qwen2RMSNorm
            except ImportError:
                # Fallback to standard LayerNorm if Qwen2RMSNorm not available
                Qwen2RMSNorm = nn.LayerNorm

            geo_norm = Qwen2RMSNorm(self.geo_hidden_size, eps=1e-6)
            geo_mlp = nn.Sequential(
                nn.Linear(self.geo_hidden_size, self.geo_hidden_size * 2),
                nn.GELU(),
                nn.Linear(self.geo_hidden_size * 2, self.vis_hidden_size),
            )
            fusion_layer = nn.Sequential(
                geo_norm,
                geo_mlp
            )
        elif self.config.fusion_method == "deepstack_vision_cross_attn":

            fusion_layer = nn.ModuleDict({
                "geo_proj": nn.Sequential(
                    nn.LayerNorm(self.geo_hidden_size),
                    nn.Linear(self.geo_hidden_size, self.vis_hidden_size)
                ),
                # Unified cross attention with configurable position encoding
                "cross_attn": CrossAttentionBlock(
                    self.vis_hidden_size, 
                    self.config.num_heads, 
                    self.config.dropout),
            })
        elif self.config.fusion_method == "deepstack_language_add":
            # Import here to avoid circular import
            try:
                from .modeling_qwen2_5_vl import Qwen2RMSNorm
            except ImportError:
                # Fallback to standard LayerNorm if Qwen2RMSNorm not available
                Qwen2RMSNorm = nn.LayerNorm

            layer_modules = {
                "geo_ln":Qwen2RMSNorm(self.geo_hidden_size, eps=1e-6),
                "geo_mlp": nn.Sequential(
                    nn.Linear(self.geo_hidden_size * self.config.spatial_merge_size ** 2, 4096),
                    nn.GELU(),
                    nn.Linear(4096, self.lang_hidden_size),
                ),
            }
            if self.use_importance_gate:
                # Per-position salience gate conditioned on the (merged) geometry token.
                # Input matches geo_mlp's input dim; outputs a scalar in (0, 1) per position.
                gate_in = self.geo_hidden_size * self.config.spatial_merge_size ** 2
                layer_modules["geo_gate"] = nn.Sequential(
                    nn.Linear(gate_in, gate_in // 4),
                    nn.ReLU(),
                    nn.Linear(gate_in // 4, 1),
                    nn.Sigmoid(),
                )
            if self.use_learnable_scale:
                layer_modules["geo_scale"] = LearnableScale(self.fusion_scale)
            fusion_layer = nn.ModuleDict(layer_modules)
        elif self.config.fusion_method == "deepstack_language_cross_attn":
            # Import here to avoid circular import
            try:
                from .modeling_qwen2_5_vl import Qwen2RMSNorm
            except ImportError:
                # Fallback to standard LayerNorm if Qwen2RMSNorm not available
                Qwen2RMSNorm = nn.LayerNorm

            fusion_layer = nn.ModuleDict({
                "geo_ln": Qwen2RMSNorm(self.geo_hidden_size, eps=1e-6),
                "geo_mlp": nn.Sequential(
                    nn.Linear(self.geo_hidden_size * self.config.spatial_merge_size ** 2, 4096),
                    nn.GELU(),
                    nn.Linear(4096, self.lang_hidden_size),
                ),
                "cam_proj": nn.Sequential(
                    Qwen2RMSNorm(self.geo_hidden_size, eps=1e-6),
                    nn.Linear(self.geo_hidden_size, 4096),
                    nn.GELU(),
                    nn.Linear(4096, self.lang_hidden_size),
                ),
                "cross_attn": CrossAttentionBlock(
                    self.lang_hidden_size,
                    self.config.num_heads,
                    self.config.dropout,
                ),
            })
        elif self.config.fusion_method == "deepstack_language_sgf":
            # Import here to avoid circular import
            try:
                from .modeling_qwen2_5_vl import Qwen2RMSNorm
            except ImportError:
                Qwen2RMSNorm = nn.LayerNorm

            fusion_layer = nn.ModuleDict({
                "geo_ln": Qwen2RMSNorm(self.geo_hidden_size, eps=1e-6),
                "geo_mlp": nn.Sequential(
                    nn.Linear(self.geo_hidden_size * self.config.spatial_merge_size ** 2, 4096),
                    nn.GELU(),
                    nn.Linear(4096, self.lang_hidden_size),
                ),
                "sgf": SGFCrossAttentionBlock(
                    self.lang_hidden_size,
                    self.config.num_heads,
                    self.config.dropout,
                    use_importance_gate=self.use_importance_gate,
                    use_spatial_bias=self.use_spatial_bias,
                ),
            })
        else:
            raise ValueError(f"Unknown fusion type: {self.config.fusion_method}")

        return fusion_layer

    def project_language_add_geometry(
        self,
        features_3d: torch.Tensor,
        layer_num: int,
        fusion_layer_idx: int = 0,
    ) -> torch.Tensor:
        """Project geometry into a ready-to-add language-space delta.

        In eval mode this projection is independent of the language/vision tokens,
        so frame-strict inference can compute it once per trajectory frame and cache
        the smaller projected tensor instead of repeatedly running the geometry MLP.
        """
        if self.fusion_method != "deepstack_language_add":
            raise ValueError(
                "Preprojected geometry is only valid for deepstack_language_add."
            )
        fusion_layers = self.get_fusion_layer(layer_num)
        if not 0 <= fusion_layer_idx < len(fusion_layers):
            raise IndexError(
                f"fusion_layer_idx {fusion_layer_idx} out of range for layer "
                f"{layer_num} ({len(fusion_layers)} blocks)"
            )
        fusion_layer = fusion_layers[fusion_layer_idx]
        geo_feats = fusion_layer["geo_ln"](features_3d)
        geo_feats = geo_feats.reshape(
            -1,
            self.config.geo_hidden_size * self.config.spatial_merge_size ** 2,
        )
        delta = fusion_layer["geo_mlp"](geo_feats)
        if "geo_gate" in fusion_layer:
            delta = fusion_layer["geo_gate"](geo_feats) * delta
        if "geo_scale" in fusion_layer:
            delta = fusion_layer["geo_scale"](delta)
        else:
            delta = self.fusion_scale * delta
        return delta

    def add_preprojected_language_geometry(
        self,
        features_2d: torch.Tensor,
        projected_deltas,
        layer_num: int,
    ) -> torch.Tensor:
        """Add cached language-space geometry deltas without rerunning the MLP."""
        if self.fusion_method != "deepstack_language_add":
            raise ValueError(
                "Preprojected geometry is only valid for deepstack_language_add."
            )
        deltas = (
            list(projected_deltas)
            if isinstance(projected_deltas, (list, tuple))
            else [projected_deltas]
        )
        fusion_layers = self.get_fusion_layer(layer_num)
        if len(deltas) not in (1, len(fusion_layers)):
            raise ValueError(
                f"Expected 1 or {len(fusion_layers)} projected deltas for layer "
                f"{layer_num}, got {len(deltas)}"
            )
        for delta in deltas:
            delta = delta.reshape(-1, delta.shape[-1])
            if delta.shape != features_2d.shape:
                raise ValueError(
                    "Preprojected geometry shape mismatch: "
                    f"features_2d={tuple(features_2d.shape)}, delta={tuple(delta.shape)}"
                )
            features_2d = features_2d + delta
        return features_2d

    def reset_residual_branches_to_noop(self) -> None:
        if self.config.fusion_method == "deepstack_language_add":
            for fusion_layers in self.fusion_layers.values():
                for fusion_layer in fusion_layers:
                    self._zero_init_last_linear(fusion_layer["geo_mlp"])
                    if "geo_gate" in fusion_layer:
                        self._open_init_gate(fusion_layer["geo_gate"])
                    if "geo_scale" in fusion_layer:
                        # Re-assert the init (pre-load); a trained value overrides on load.
                        with torch.no_grad():
                            fusion_layer["geo_scale"].scale.fill_(self.fusion_scale)
        elif self.config.fusion_method == "deepstack_language_sgf":
            # tanh(gate)=0 makes the SGF cross-attention a no-op at init (pre-load;
            # a trained gate overrides on load).
            for fusion_layers in self.fusion_layers.values():
                for fusion_layer in fusion_layers:
                    with torch.no_grad():
                        fusion_layer["sgf"].gate.zero_()

    @staticmethod
    def _zero_init_last_linear(module: nn.Module) -> None:
        for layer in reversed(list(module.modules())):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                return

    @staticmethod
    def _open_init_gate(module: nn.Module, bias_value: float = 4.0) -> None:
        # Init the sigmoid gate ~open (g≈sigmoid(4)≈0.98): zero weight makes it
        # uniform, positive bias pushes it toward 1 so early training ≈ ungated add
        # and the gate then LEARNS which geometry to suppress. (Step-0 loss parity is
        # already guaranteed by the zero-init geo_mlp, independent of the gate.)
        for layer in reversed(list(module.modules())):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, bias_value)
                return

    def forward(
        self, 
        features_2d: torch.Tensor,
        features_3d: torch.Tensor,
        layer_num: int,
        vis_pos_embed: Optional[torch.Tensor] = None,
        geo_pos_embed: Optional[torch.Tensor] = None,
        fusion_layer_idx: Optional[int] = None,
        grid_hw: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Fuse 2D and 3D features.
        
        Args:
            features_2d: 2D image features
            features_3d: 3D geometry features (single tensor or list/tuple of tensors)
            layer_num: layer_num
            fusion_layer_idx: select which fusion block to use when multiple are attached to the same layer
        Returns:
            Fused features
        """
        fusion_layers = self.get_fusion_layer(layer_num)
        features_3d_list = list(features_3d) if isinstance(features_3d, (list, tuple)) else [features_3d]

        if fusion_layer_idx is not None:
            if fusion_layer_idx >= len(features_3d_list):
                raise ValueError(
                    f"fusion_layer_idx {fusion_layer_idx} out of range for provided features "
                    f"(available: {len(features_3d_list)})"
                )
            features_3d_list = [features_3d_list[fusion_layer_idx]]
            fusion_indices = [fusion_layer_idx]
        else:
            if len(features_3d_list) not in (1, len(fusion_layers)):
                raise ValueError(
                    f"Expected 1 or {len(fusion_layers)} geometry feature tensors for layer {layer_num}, "
                    f"got {len(features_3d_list)}"
                )
            if len(fusion_layers) == 1:
                fusion_indices = [0] * len(features_3d_list)
            else:
                fusion_indices = list(range(len(features_3d_list)))

        for geo_idx, fusion_idx in enumerate(fusion_indices):
            fusion_layer = fusion_layers[fusion_idx]

            if self.fusion_method == "deepstack_vision_add":
                geo_feats = fusion_layer(features_3d_list[geo_idx])
                assert features_2d.shape == geo_feats.shape, (
                    f"Shape mismatch: features_2d={features_2d.shape}, features_3d={geo_feats.shape}"
                )
                features_2d = features_2d + self.fusion_scale * geo_feats

            # cross attention
            elif self.config.fusion_method == "deepstack_vision_cross_attn":
                geo_feats = fusion_layer['geo_proj'](features_3d_list[geo_idx])
                
                # cross attention
                features_2d = fusion_layer['cross_attn'](features_2d, geo_feats, vis_pos_embed, geo_pos_embed)

            elif self.config.fusion_method == "deepstack_language_add":
                geo_feats = fusion_layer['geo_ln'](features_3d_list[geo_idx])
                geo_feats = geo_feats.reshape(-1, self.config.geo_hidden_size * self.config.spatial_merge_size ** 2)
                delta = fusion_layer['geo_mlp'](geo_feats)
                if 'geo_gate' in fusion_layer:
                    # Per-position salience g in (0,1) from the geometry token; suppresses
                    # redundant background geometry (floor/wall/empty) before it is added.
                    delta = fusion_layer['geo_gate'](geo_feats) * delta
                if 'geo_scale' in fusion_layer:
                    # Step 2'': learnable per-layer scale replaces the fixed fusion_scale.
                    features_2d = features_2d + fusion_layer['geo_scale'](delta)
                else:
                    features_2d = features_2d + self.fusion_scale * delta

            elif self.config.fusion_method == "deepstack_language_cross_attn":
                geo_feats = features_3d_list[geo_idx]
                num_imgs, num_geo_tokens = geo_feats.shape[:2]
                num_merged_patch_tokens = features_2d.shape[0] // num_imgs
                # split cam and patchs
                if num_merged_patch_tokens * self.config.spatial_merge_size ** 2 == num_geo_tokens:
                    geo_feats = fusion_layer['geo_ln'](geo_feats)
                    geo_feats = geo_feats.reshape(-1, self.config.geo_hidden_size * self.config.spatial_merge_size ** 2)
                    geo_feats = fusion_layer['geo_mlp'](geo_feats)
                    geo_feats = geo_feats.reshape(num_imgs, num_merged_patch_tokens, -1)
                else:
                    features_3d_cam = fusion_layer['cam_proj'](geo_feats[:, 0:1])
                    features_3d_patchs = fusion_layer['geo_ln'](geo_feats[:, 1:])
                    features_3d_patchs = features_3d_patchs.reshape(-1, self.config.geo_hidden_size * self.config.spatial_merge_size ** 2)
                    features_3d_patchs = fusion_layer['geo_mlp'](features_3d_patchs)
                    features_3d_patchs = features_3d_patchs.reshape(num_imgs, num_merged_patch_tokens, -1)
                    geo_feats = torch.cat([features_3d_cam, features_3d_patchs], dim=1)

                # cross attention
                features_2d = features_2d.reshape(num_imgs, num_merged_patch_tokens, -1)
                features_2d = fusion_layer['cross_attn'](features_2d, geo_feats, vis_pos_embed, geo_pos_embed)
                features_2d = features_2d.reshape(num_imgs * num_merged_patch_tokens, -1)

            elif self.config.fusion_method == "deepstack_language_sgf":
                # Step 2 + (opt) 3/4: GeoThinker SGF on the language path.
                # Assumes patch-only geometry (include_camera_token=False): T == num_merged * m^2.
                geo_feats = features_3d_list[geo_idx]
                num_imgs = geo_feats.shape[0]
                num_merged = features_2d.shape[0] // num_imgs
                geo_feats = fusion_layer['geo_ln'](geo_feats)
                geo_feats = geo_feats.reshape(-1, self.config.geo_hidden_size * self.config.spatial_merge_size ** 2)
                geo_feats = fusion_layer['geo_mlp'](geo_feats)
                geo_feats = geo_feats.reshape(num_imgs, num_merged, -1)         # K/V: geometry, per frame
                vis = features_2d.reshape(num_imgs, num_merged, -1)             # Q: vision, per frame (frame-strict)
                delta = fusion_layer['sgf'](vis, geo_feats, grid_hw=grid_hw)    # tanh(gate)-scaled delta
                features_2d = features_2d + delta.reshape(num_imgs * num_merged, -1)
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return features_2d
