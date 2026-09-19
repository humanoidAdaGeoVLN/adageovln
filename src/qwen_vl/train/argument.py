import transformers
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    # Geometry encoder configuration
    use_geometry_encoder: bool = field(default=False)  # Whether to use 3D geometry encoder
    geometry_encoder_type: str = field(default="vggt")  # Type of geometry encoder ("vggt", "pi3")
    geometry_encoder_path: str = field(default="facebook/VGGT-1B/")  # Path to pre-trained geometry encoder model
    reference_frame: str = field(default="first")  # Reference frame for geometry encoding ("first", "last"), only available for vggt
    feature_fusion_method: str = field(default="add")  # Method to fuse geometry and visual features ("add", "concat", "cross_attention", "gate")
    fusion_num_layers: int = field(default=1)  # Number of layers in the cross-attention module when feature_fusion_method is "cross_attention"
    geometry_merger_type: str = field(default="mlp")  # Type of geometry feature merger ("mlp", "avg")
    geometry_fusion_layers: Optional[List[int]] = field(default=None)  # Vision block indices for layer-wise fusion
    geometry_encoder_layers: Optional[List[int]] = field(default=None)  # Geometry encoder layer indices
    include_camera_token: bool = field(default=False)  # Whether to include camera token
    pos_encoding_type: str = field(default="none")  # Position encoding: "none", "rope2d", or "sincos2d"
    vision_language_fusion_layers: Optional[List[int]] = field(default=None)  # Vision block indices to fuse into decoder
    geometry_encoder_streaming: bool = field(default=False)
    use_adageo_kv_cache: bool = field(default=False)  # Use AdaGeoKV-style VGGT KV eviction instead of start+recent trimming
    vggt_total_budget: int = field(default=1_200_000)  # Token budget used by AdaGeoKV-style VGGT KV eviction
    vggt_importance_weights_path: str = field(default="config/importance_weights_default.json")
    vggt_budget_proportions_path: str = field(default="config/kv_budget_proportions_cosine.json")
    vln_segment_transition_weights_path: str = field(default="config/adageovln.json")
    geometry_fusion_scale: float = field(default=1.0)
    geometry_frame_strict: bool = field(default=False)  # Fuse each frame with its OWN geometry (vs last-frame broadcast). Env FUSION_FRAME_STRICT overrides.
    geometry_importance_gate: bool = field(default=False)  # Step 2': per-position sigmoid gate on the geometry delta (background suppression). Env FUSION_IMPORTANCE_GATE overrides.
    geometry_learnable_scale: bool = field(default=False)  # Step 2'': learnable per-layer scale replacing fixed geometry_fusion_scale. Env FUSION_LEARNABLE_SCALE overrides.
    geometry_spatial_bias: bool = field(default=False)  # Step 4: GeoThinker spatial-distance bias on the SGF cross-attention (deepstack_language_sgf only). Env FUSION_SPATIAL_BIAS overrides.
    stop_loss_weight: float = field(default=1.0)  # Up-weight STOP-action tokens in the LM loss (exposure-bias fix)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    max_samples: int = field(default=-1)
    shuffle: bool = field(default=True)
    debug_vln: bool = field(
        default=False,
        metadata={"help": "Log VLN shapes and optionally save frame images during training."},
    )
    debug_vln_save_dir: str = field(
        default="",
        metadata={"help": "Directory for debug images (default: <output_dir>/debug_vln)."},
    )
    debug_vln_max_samples: int = field(
        default=5,
        metadata={"help": "Max dataset samples / collator batches to log when debug_vln is on."},
    )
    debug_vln_max_steps: int = field(
        default=5,
        metadata={"help": "Deprecated: use debug_vln_save_interval. Max snapshots kept unused when interval is set."},
    )
    debug_vln_save_interval: int = field(
        default=100,
        metadata={"help": "Save debug images / print VLN debug logs every N training steps (rank 0)."},
    )
    debug_vln_save_geo_layers: bool = field(
        default=True,
        metadata={"help": "Save per-layer VGGT activation heatmaps when debug_vln is on."},
    )
    debug_vln_save_depth: bool = field(
        default=False,
        metadata={"help": "Run VGGT depth head and save depth PNGs (extra forward, loads depth_head)."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
