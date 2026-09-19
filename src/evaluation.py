import argparse
import json
import os
import random
import re
import time
import warnings
from collections import OrderedDict
from typing import Any, Optional, Set

import numpy as np
import torch
import tqdm
from PIL import Image
from packaging.version import Version
import transformers
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

import habitat
from habitat import Env
from habitat.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

from habitat_extensions import measures, task  # noqa: F401
from utils.dist import get_rank, get_world_size, init_distributed_mode
from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry

try:
    import torch.distributed as dist
except ImportError:
    dist = None

MIN_QWEN3_5_TRANSFORMERS_VERSION = Version("5.3.0")
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1605632
ACTION_NAMES = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")

VLN_SYSTEM_PROMPT = (
    "You are a visual language navigation model, and your should go to the locations "
    "to complete the given task. Compare the observation and instruction to infer "
    "your current progress, and then select the correct direction from the candidates "
    "to go to the target location and finish the task."
)


def build_vln_context(task: str) -> str:
    return (
        "These images are your historical observations and your current observation.\n"
        f" Your task is to {task} \n"
        " You should take one of the following actions:\n"
        " MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def require_qwen3_5_support():
    current_version = Version(transformers.__version__)
    if current_version < MIN_QWEN3_5_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Qwen3.5 evaluation requires transformers>="
            f"{MIN_QWEN3_5_TRANSFORMERS_VERSION}, but found {transformers.__version__}."
        )


def patch_qwen3_5_flash_attention():
    try:
        import transformers.modeling_flash_attention_utils as flash_attention_utils
    except ImportError:
        return

    if getattr(flash_attention_utils, "_adageovln_qwen3_5_mrope_patch", False):
        return

    original_is_packed_sequence = flash_attention_utils._is_packed_sequence

    def patched_is_packed_sequence(position_ids, batch_size):
        if position_ids is not None and getattr(position_ids, "ndim", None) == 3:
            return False
        return original_is_packed_sequence(position_ids, batch_size)

    flash_attention_utils._is_packed_sequence = patched_is_packed_sequence
    flash_attention_utils._adageovln_qwen3_5_mrope_patch = True


def build_qwen3_5_geometry_inputs(images, image_grid_thw, patch_size: int = 14):
    geometry_tensors = []
    max_height = 0
    max_width = 0

    for image, grid in zip(images, image_grid_thw):
        _, grid_h, grid_w = [int(v) for v in grid.tolist()]
        target_height = grid_h * patch_size
        target_width = grid_w * patch_size
        resized = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float() / 255.0
        geometry_tensors.append(tensor)
        max_height = max(max_height, target_height)
        max_width = max(max_width, target_width)

    padded_tensors = []
    for tensor in geometry_tensors:
        h_padding = max_height - tensor.shape[1]
        w_padding = max_width - tensor.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            tensor = torch.nn.functional.pad(
                tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )
        padded_tensors.append(tensor)

    return padded_tensors


def strip_thinking_content(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def build_vln_messages(observations, task: str):
    if not observations:
        raise ValueError("At least one observation image is required.")
    if not all(isinstance(image, Image.Image) for image in observations):
        raise TypeError("All observations must be PIL images.")

    content = [{"type": "image", "image": image} for image in observations]
    content.append({"type": "text", "text": build_vln_context(task)})
    return [[
        {"role": "system", "content": VLN_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]]


def parse_vln_action(text: str) -> str:
    text = strip_thinking_content(text).strip().upper()
    for action in ACTION_NAMES:
        if text == action:
            return action
    for action in ACTION_NAMES:
        if action in text:
            return action
    return "STOP"


def _cuda_device(device):
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def reset_peak_memory_stats(device):
    if not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats(_cuda_device(device))


def read_peak_memory_stats(device):
    if not torch.cuda.is_available():
        return {"peak_allocated_mb": 0.0, "peak_reserved_mb": 0.0}
    device = _cuda_device(device)
    torch.cuda.synchronize(device)
    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024 ** 2,
    }


def vggt_kv_bytes(past_key_values):
    if past_key_values is None:
        return 0
    total = 0
    for kv in past_key_values:
        if kv is None:
            continue
        k, v = kv
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


def vln_metadata_bytes(metadata_per_layer):
    """Return storage owned by per-token eviction metadata across VGGT layers."""
    if metadata_per_layer is None:
        return 0
    total = 0
    for metadata in metadata_per_layer:
        if metadata is None:
            continue
        for value in vars(metadata).values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


class AdaGeoVLNInference:
    def __init__(self, pretrained: str, device: str = "cuda", geometry_encoder_path: Optional[str] = None):
        require_qwen3_5_support()
        patch_qwen3_5_flash_attention()

        config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        if geometry_encoder_path:
            config.geometry_encoder_path = geometry_encoder_path

        model_kwargs = {
            "pretrained_model_name_or_path": pretrained,
            "config": config,
            "torch_dtype": torch.bfloat16,
            "device_map": {"": device},
            "attn_implementation": "flash_attention_2",
            "trust_remote_code": True,
        }
        geo_path = geometry_encoder_path or getattr(config, "geometry_encoder_path", None)
        if geo_path:
            model_kwargs["geometry_encoder_path"] = geo_path

        self.model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(**model_kwargs).eval()
        self.use_geometry = getattr(config, "use_geometry_encoder", False)
        self.use_geometry_streaming = getattr(config, "geometry_encoder_streaming", False)
        # Frame-strict eval: fuse each frame with its OWN geometry (as trained), instead
        # of broadcasting the current frame. Env FUSION_FRAME_STRICT overrides config,
        # using the SAME resolution as the model's _collect_geometry_layer_features so
        # the eval wrapper and the model agree.
        _env_fs = os.environ.get("FUSION_FRAME_STRICT")
        self.use_frame_strict = (
            _env_fs.lower() in ("1", "true", "yes")
            if _env_fs is not None
            else bool(getattr(config, "geometry_frame_strict", False))
        )
        if self.use_geometry and not self.use_geometry_streaming:
            warnings.warn(
                "Checkpoint has use_geometry_encoder=True but geometry_encoder_streaming=False. "
                "Habitat eval will only encode the current frame without VGGT temporal cache.",
                stacklevel=2,
            )
        if self.use_geometry and self.use_geometry_streaming:
            # Persistent growing VGGT KV either way (accumulated 3D memory over the episode).
            self.model.enable_vln_eval_streaming()
            if self.use_frame_strict:
                # Incremental frame-strict: encode only the new frame each step against the
                # growing KV, project its selected VGGT layers through the fusion MLP once,
                # then gather cached per-frame deltas for the current window.
                self.model.enable_vln_frame_strict_eval()
                warnings.warn(
                    "Frame-strict eval (incremental): encodes 1 frame/step with growing VGGT "
                    "KV and caches post-MLP per-frame geometry deltas on CPU.",
                    stacklevel=2,
                )
        elif not self.use_geometry:
            warnings.warn(
                "Checkpoint has use_geometry_encoder=False; running vision-only VLN eval.",
                stacklevel=2,
            )

        # Intermediate training checkpoints save the model + tokenizer but NOT the
        # image/video processor configs. Allow PROCESSOR_PATH (base model or a full
        # checkpoint) to supply tokenizer/processor while weights come from `pretrained`.
        proc_src = os.environ.get("PROCESSOR_PATH") or pretrained
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left", trust_remote_code=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(proc_src, padding_side="left", trust_remote_code=True)
        try:
            self.processor = AutoProcessor.from_pretrained(
                pretrained, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS,
                padding_side="left", trust_remote_code=True,
            )
        except Exception as e:
            if os.environ.get("PROCESSOR_PATH") is None:
                raise RuntimeError(
                    f"Could not load the processor from '{pretrained}'. Intermediate checkpoints "
                    f"don't save preprocessor_config.json/processor_config.json. Set "
                    f"PROCESSOR_PATH to a dir that has them (base Qwen3.5-4B or a full/final checkpoint)."
                ) from e
            self.processor = AutoProcessor.from_pretrained(
                proc_src, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS,
                padding_side="left", trust_remote_code=True,
            )
        self.device = device
        self.model_device = self.model.device

    def reset_geometry_cache(self):
        self.model.reset_vln_geometry_cache()

    def initialize_segment_transition_instruction(self, instruction: str):
        """Initialize optional per-episode instruction metadata (a no-op otherwise)."""
        self.model.initialize_vln_segment_instruction(instruction, self.tokenizer)

    def _geometry_kv_cache(self):
        encoder = getattr(self.model.model, "geometry_encoder", None)
        if encoder is None:
            return None
        return getattr(encoder, "_streaming_past_key_values", None)

    def _geometry_cache_memory_mb(self):
        encoder = getattr(self.model.model, "geometry_encoder", None)
        if encoder is None:
            return {"kv": 0.0, "camera_kv": 0.0, "metadata": 0.0, "total": 0.0}
        scale = 1024 ** 2
        kv = vggt_kv_bytes(getattr(encoder, "_streaming_past_key_values", None)) / scale
        camera_kv = vggt_kv_bytes(
            getattr(encoder, "_streaming_past_key_values_camera", None)
        ) / scale
        metadata = vln_metadata_bytes(
            getattr(encoder, "_vln_metadata_per_layer", None)
        ) / scale
        return {
            "kv": kv,
            "camera_kv": camera_kv,
            "metadata": metadata,
            "total": kv + camera_kv + metadata,
        }

    def _last_vggt_ms(self):
        encoder = getattr(self.model.model, "geometry_encoder", None)
        if encoder is None:
            return 0.0
        return float(getattr(encoder, "last_vggt_ms", 0.0))

    def _build_model_inputs(self, observations, task: str) -> dict:
        messages = build_vln_messages(observations, task)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.processor(
            text=text,
            images=list(observations),
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        if self.use_geometry:
            # Both modes encode only the CURRENT frame each step (growing KV). Broadcast
            # tiles it to all frames; frame-strict buffers it and gathers each window
            # frame's own geometry (window indices set on the model before generate).
            current_geometry = build_qwen3_5_geometry_inputs(
                [observations[-1]],
                model_inputs["image_grid_thw"][-1:],
            )
            model_inputs["geometry_encoder_inputs"] = [torch.stack(current_geometry)]
            if (not getattr(self, "_geom_debug_logged", False)
                    and get_rank() == 0 and len(observations) > 1):   # skip step-0 single-frame (uninformative)
                self._geom_debug_logged = True
                print(
                    f"[eval-geom] frame_strict={self.use_frame_strict} (incremental) "
                    f"vision_frames={len(observations)} geom_encoded_this_step=1 "
                    f"(frame-strict gathers {len(observations)} frames from the buffer)"
                )
        return model_inputs

    def call_model(self, observations, task: str, step_id: int, gen_kwargs: Optional[dict] = None,
                   frame_indices=None):
        del step_id
        gen_kwargs = gen_kwargs or {}

        # Incremental frame-strict: tell the encoder which trajectory frames the LLM
        # window contains, so it gathers each one's own buffered geometry this step.
        if self.use_geometry and self.use_frame_strict and frame_indices is not None:
            self.model.set_vln_eval_window_indices(frame_indices)

        model_inputs = self._build_model_inputs(observations, task)

        device = self.model_device
        if self.use_geometry:
            model_inputs["geometry_encoder_inputs"] = [
                feat.to(device) for feat in model_inputs["geometry_encoder_inputs"]
            ]
        model_inputs = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in model_inputs.items()
        }

        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 24
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        with torch.inference_mode():
            output_ids = self.model.generate(
                **model_inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=gen_kwargs["temperature"] > 0,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=True,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs["input_ids"], output_ids)
        ]
        answers = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        parsed = [parse_vln_action(ans) for ans in answers]
        if os.environ.get("VLN_EVAL_DEBUG"):
            print(
                f"[EVAL_DEBUG] n_imgs={len(observations)} raw={answers!r} -> {parsed}",
                flush=True,
            )
        return parsed


class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
        scene_filter: Optional[Set[str]] = None,
    ):
        self.args = args
        self.device = getattr(model, "model_device", getattr(model, "device", "cuda:0"))
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors
        self.save_video_ratio = args.save_video_ratio
        self.scene_filter = scene_filter

        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = self.split
            if torch.cuda.is_available():
                gpu_id = getattr(args, "local_rank", getattr(args, "gpu", 0))
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id
            measurements = {"collisions": CollisionsMeasurementConfig()}
            if self.save_video:
                measurements["top_down_map"] = TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
                )
            self.config.habitat.task.measurements.update(measurements)

        if self.scene_filter and get_rank() == 0:
            print(f"[eval] skipping scenes: {sorted(self.scene_filter)}", flush=True)

        self.model = model
        self.actions2idx = OrderedDict({
            "STOP": [0],
            "MOVE_FORWARD": [1],
            "TURN_LEFT": [2],
            "TURN_RIGHT": [3],
        })
        self.num_history = args.num_history

        # Oracle-stop diagnostic: force STOP once within success_distance of the goal.
        # Isolates navigation quality from the (separately trained) stop policy.
        self.oracle_stop = os.environ.get("VLN_ORACLE_STOP", "").lower() in ("1", "true", "yes")
        try:
            self.success_distance = float(self.config.habitat.task.measurements.success.success_distance)
        except Exception:
            self.success_distance = 3.0
        if self.oracle_stop and get_rank() == 0:
            print(f"[eval] VLN_ORACLE_STOP: auto-STOP within {self.success_distance}m of goal", flush=True)

        # Teacher-forced diagnostic: roll out along the GT expert path (execute the
        # expert action each step) and score the model's predicted action. Isolates
        # per-step action accuracy (what CE loss measures) from rollout/exposure bias.
        self.teacher_forced = os.environ.get("VLN_TEACHER_FORCED", "").lower() in ("1", "true", "yes")
        self.gt_actions_map = None
        if self.teacher_forced:
            import gzip as _gzip
            roles = list(
                getattr(self.config.habitat.dataset, "roles", ["guide"])
            )
            dp = self.config.habitat.dataset.data_path.format(
                split=self.split,
                role=roles[0] if roles else "guide",
            )
            gt_path = (dp[: -len(".json.gz")] if dp.endswith(".json.gz") else dp) + "_gt.json.gz"
            with _gzip.open(gt_path, "rt") as f:
                self.gt_actions_map = json.load(f)
            if get_rank() == 0:
                print(f"[eval] VLN_TEACHER_FORCED: expert rollout scoring | gt={gt_path} ({len(self.gt_actions_map)} eps)", flush=True)

    def config_env(self) -> Env:
        from habitat.datasets import make_dataset

        dataset = make_dataset(
            id_dataset=self.config.habitat.dataset.type,
            config=self.config.habitat.dataset,
        )
        if len(dataset.episodes) == 0:
            raise ValueError(
                f"No episodes found for split={self.split!r}. "
                "Check the dataset path and configured content_scenes."
            )
        return Env(config=self.config)

    def eval_action(self, idx):
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            scene_episode_dict.setdefault(episode.scene_id, []).append(episode)

        sucs, spls, oss, ones, ndtws = [], [], [], [], []
        done_res = []
        result_path = os.path.join(self.output_path, "result.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                for line in f.readlines():
                    if not line.strip():
                        continue
                    res = json.loads(line)
                    if "sucs_all" in res:
                        continue
                    if self.scene_filter is not None and res["scene_id"] in self.scene_filter:
                        continue
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:
                        sucs.append(res["success"])
                        spls.append(res["spl"])
                        oss.append(res["os"])
                        ones.append(res["ne"])
                        ndtws.append(res.get("ndtw", -1.0))

        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split("/")[-2]
            if self.scene_filter is not None and scene_id in self.scene_filter:
                continue
            print(f"scene_id = {scene_id}")

            scene_episodes = episodes[idx::self.env_num]
            _max_eps = int(os.environ.get("MAX_EPISODES", "0"))
            if _max_eps > 0:
                scene_episodes = scene_episodes[:_max_eps]
                print(f"[eval] MAX_EPISODES={_max_eps}: limiting scene {scene_id} to {len(scene_episodes)} episodes")
            process_bar = tqdm.tqdm(range(len(scene_episodes)), desc=f"scene {scene_id}")
            for episode in scene_episodes:
                episode_instruction = (
                    episode.instruction.instruction_text
                    if "objectnav" not in self.config_path
                    else episode.object_category
                )
                episode_id = episode.episode_id
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue

                env.current_episode = episode
                observations = env.reset()

                vis_frames = []
                step_id = 0
                should_save_video = self.save_video and (random.random() < self.save_video_ratio)
                if should_save_video:
                    os.makedirs(os.path.join(self.output_path, f"vis_{self.epoch}"), exist_ok=True)

                rgb_list = []
                self.model.reset_geometry_cache()
                self.model.initialize_segment_transition_instruction(episode_instruction)
                reset_peak_memory_stats(self.device)
                ep_peak_alloc = 0.0
                ep_peak_reserved = 0.0
                ep_peak_vggt_kv = 0.0
                ep_peak_vggt_camera_kv = 0.0
                ep_peak_vggt_metadata = 0.0
                ep_peak_vggt_cache_total = 0.0
                ep_t0 = time.perf_counter()
                step_times_ms = []
                vggt_times_ms = []

                gt_actions = None
                tf_correct = tf_total = 0
                tf_stop_total = tf_stop_correct = 0
                if self.teacher_forced:
                    gt_entry = self.gt_actions_map.get(str(episode_id))
                    if gt_entry is None:
                        print(f"[teacher_forced] no GT actions for episode {episode_id}; skipping")
                        process_bar.update(1)
                        continue
                    gt_actions = gt_entry["actions"]

                ep_actions = []
                ep_positions = []
                while not env.episode_over:
                    if self.teacher_forced and step_id >= len(gt_actions):
                        break
                    rgb = observations["rgb"]
                    image = Image.fromarray(rgb).convert("RGB")
                    rgb_list.append(image)

                    info = env.get_metrics()
                    history_len = len(rgb_list) - 1
                    if history_len <= self.num_history:
                        frame_indices = list(range(len(rgb_list)))          # all frames so far
                    else:
                        frame_indices = np.linspace(0, history_len, self.num_history + 1, dtype=int).tolist()
                    images = [rgb_list[i] for i in frame_indices]

                    step_t0 = time.perf_counter()
                    action = self.model.call_model(
                        images, episode_instruction, step_id, frame_indices=frame_indices
                    )[0]
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.device)
                    step_times_ms.append((time.perf_counter() - step_t0) * 1000)
                    vggt_times_ms.append(self.model._last_vggt_ms())

                    mem_stats = read_peak_memory_stats(self.device)
                    ep_peak_alloc = max(ep_peak_alloc, mem_stats["peak_allocated_mb"])
                    ep_peak_reserved = max(ep_peak_reserved, mem_stats["peak_reserved_mb"])
                    cache_memory = self.model._geometry_cache_memory_mb()
                    ep_peak_vggt_kv = max(ep_peak_vggt_kv, cache_memory["kv"])
                    ep_peak_vggt_camera_kv = max(
                        ep_peak_vggt_camera_kv, cache_memory["camera_kv"]
                    )
                    ep_peak_vggt_metadata = max(
                        ep_peak_vggt_metadata, cache_memory["metadata"]
                    )
                    ep_peak_vggt_cache_total = max(
                        ep_peak_vggt_cache_total, cache_memory["total"]
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if self.save_video and info.get("top_down_map") is not None and should_save_video:
                        vis_frames.append(observations_to_image({"rgb": observations["rgb"]}, info))

                    pred_action = self.actions2idx.get(action, [0])[0]

                    if self.teacher_forced:
                        gt_action = gt_actions[step_id]
                        tf_total += 1
                        if pred_action == gt_action:
                            tf_correct += 1
                        if gt_action == 0:
                            tf_stop_total += 1
                            if pred_action == 0:
                                tf_stop_correct += 1
                        action = gt_action  # follow the expert path
                    else:
                        action = pred_action
                        if self.oracle_stop and info.get("distance_to_goal", float("inf")) <= self.success_distance:
                            action = 0
                        # step_id is zero-based and this action is counted immediately
                        # below, so force STOP on the max_steps-th action, not one later.
                        if step_id >= self.args.max_steps - 1:
                            action = 0

                    ep_actions.append(int(action))
                    ep_positions.append(_agent_xyz(env))
                    observations = env.step(action)
                    step_id += 1

                    # NOTE: do NOT prune rgb_list. Keep the full trajectory so the
                    # history subsampling above (linspace(0, current, num_history+1))
                    # spreads across the whole path, matching training. Destructively
                    # pruning here froze history to the first 8 frames + current.

                ep_positions.append(_agent_xyz(env))
                process_bar.update(1)
                metrics = env.get_metrics()
                if should_save_video:
                    images_to_video(
                        vis_frames,
                        os.path.join(self.output_path, f"vis_{self.epoch}"),
                        f"{scene_id}_{episode_id}",
                        fps=6,
                        quality=9,
                    )
                vis_frames.clear()
                ep_ndtw = float(metrics.get("ndtw", -1.0))
                sucs.append(metrics["success"])
                spls.append(metrics["spl"])
                oss.append(metrics["oracle_success"])
                ones.append(metrics["distance_to_goal"])
                ndtws.append(ep_ndtw)
                print(
                    f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, "
                    f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}"
                    + (f", ndtw: {ep_ndtw}" if ep_ndtw >= 0 else "")
                )
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics["oracle_success"],
                    "ne": metrics["distance_to_goal"],
                    "ndtw": ep_ndtw,
                    "steps": step_id,
                    "episode_instruction": episode_instruction,
                    "peak_vggt_kv_mb": ep_peak_vggt_kv,
                    "peak_vggt_camera_kv_mb": ep_peak_vggt_camera_kv,
                    "peak_vggt_metadata_mb": ep_peak_vggt_metadata,
                    "peak_vggt_cache_total_mb": ep_peak_vggt_cache_total,
                    "peak_alloc_mb": ep_peak_alloc,
                    "peak_reserved_mb": ep_peak_reserved,
                    "mean_step_ms": float(np.mean(step_times_ms)) if step_times_ms else 0.0,
                    "mean_vggt_ms": float(np.mean(vggt_times_ms)) if vggt_times_ms else 0.0,
                    "episode_time_s": time.perf_counter() - ep_t0,
                }
                # The per-step action and position traces dominate the result file on long
                # episodes (74% of a 4 MB RxR run) and nothing downstream reads them: nDTW
                # comes from the habitat metrics, and teacher-forcing reads its ground truth
                # from a separate file. Opt in when a trajectory dump is actually needed.
                if os.environ.get("VLN_SAVE_TRAJECTORY", "0") == "1":
                    result["actions"] = ep_actions
                    result["positions"] = ep_positions
                if self.teacher_forced:
                    result["tf_action_acc"] = tf_correct / tf_total if tf_total else 0.0
                    result["tf_steps"] = tf_total
                    result["tf_stop_recall"] = (tf_stop_correct / tf_stop_total) if tf_stop_total else -1.0
                    print(
                        f"[teacher_forced] {scene_id}_{episode_id} "
                        f"action_acc={result['tf_action_acc']:.3f} ({tf_correct}/{tf_total}) "
                        f"stop_recall={result['tf_stop_recall']:.2f}"
                    )
                with open(result_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

        env.close()
        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(ones).to(self.device),
            torch.tensor(ndtws).to(self.device),
            torch.tensor(len(sucs)).to(self.device),
        )


def _agent_xyz(env):
    """Agent position, rounded to 0.1mm — far finer than the 3m success radius."""
    return [round(float(v), 4) for v in env.sim.get_agent_state().position]


def summarize_ndtw(ndtws):
    """Average nDTW over episodes that actually have it.

    -1 marks "not measured": R2R does not configure the measure, and episodes
    resumed from a result.json written before nDTW was recorded have no value.
    Those are excluded rather than averaged in as zeros.
    """
    valid = ndtws[ndtws >= 0]
    if valid.numel() == 0:
        return {}
    return {
        "ndtw_all": (valid.sum() / valid.numel()).item(),
        "ndtw_length": valid.numel(),
    }


def evaluate(model, args, scene_filter=None):
    world_size = get_world_size()
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        epoch=0,
        args=args,
        scene_filter=scene_filter,
    )
    sucs, spls, oss, ones, ndtws, ep_num = evaluator.eval_action(get_rank())

    if world_size == 1 or dist is None or not dist.is_initialized():
        result_all = {
            "sucs_all": (sum(sucs) / len(sucs)).item() if len(sucs) else 0.0,
            "spls_all": (sum(spls) / len(spls)).item() if len(spls) else 0.0,
            "oss_all": (sum(oss) / len(oss)).item() if len(oss) else 0.0,
            "ones_all": (sum(ones) / len(ones)).item() if len(ones) else 0.0,
            "length": len(sucs),
        }
        result_all.update(summarize_ndtw(ndtws))
        print(result_all)
        if get_rank() == 0:
            with open(os.path.join(args.output_path, "result.json"), "a") as f:
                f.write(json.dumps(result_all) + "\n")
        return

    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)
    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
    ndtws_all = [torch.zeros(ep_num_all[i], dtype=ndtws.dtype).to(ndtws.device) for i in range(world_size)]
    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(ones_all, ones)
    dist.all_gather(ndtws_all, ndtws)
    dist.barrier()
    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    ones_all = torch.cat(ones_all, dim=0)
    ndtws_all = torch.cat(ndtws_all, dim=0)
    result_all = {
        "sucs_all": (sum(sucs_all) / len(sucs_all)).item(),
        "spls_all": (sum(spls_all) / len(spls_all)).item(),
        "oss_all": (sum(oss_all) / len(oss_all)).item(),
        "ones_all": (sum(ones_all) / len(ones_all)).item(),
        "length": len(sucs_all),
    }
    result_all.update(summarize_ndtw(ndtws_all))
    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, "result.json"), "a") as f:
            f.write(json.dumps(result_all) + "\n")


def eval_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--geometry_encoder_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default="config/vln_r2r.yaml")
    parser.add_argument("--eval_split", type=str, default="val_unseen")
    parser.add_argument("--output_path", type=str, default="./evaluation/adageovln")
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--save_video_ratio", type=float, default=0.2, help="0~1")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--port", default="1111")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_steps", default=400, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scene_ids",
        type=str,
        required=True,
        help="Comma-separated MP3D scene ids to skip (e.g. EU6Fwq7SyZv).",
    )
    args = parser.parse_args()

    scene_filter = {
        scene_id.strip()
        for scene_id in args.scene_ids.split(",")
        if scene_id.strip()
    }
    if not scene_filter:
        raise ValueError("--scene_ids must contain at least one scene id to skip")

    set_seed(args.seed)
    init_distributed_mode(args)

    geometry_encoder_path = args.geometry_encoder_path or os.environ.get("GEOMETRY_ENCODER_PATH")
    model = AdaGeoVLNInference(
        args.model_path,
        device=f"cuda:{args.local_rank}",
        geometry_encoder_path=geometry_encoder_path or None,
    )
    evaluate(model, args, scene_filter=scene_filter)


if __name__ == "__main__":
    eval_main()
