import argparse

from evaluation import AdaGeoVLNInference, evaluate, init_distributed_mode, set_seed


def parse_scene_ids(scene_ids: str):
    scene_ids = scene_ids.strip()
    if scene_ids.lower() in {"none", "all"}:
        return None
    return {scene_id.strip() for scene_id in scene_ids.split(",") if scene_id.strip()}


def eval_scene():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--geometry_encoder_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default="config/vln_r2r.yaml")
    parser.add_argument("--eval_split", type=str, default="val_unseen")
    parser.add_argument(
        "--scene_ids",
        type=str,
        required=True,
        help="Comma-separated MP3D scene ids to skip, or 'none'/'all' to skip none.",
    )
    parser.add_argument("--output_path", type=str, default="./evaluation/scene")
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--save_video_ratio", type=float, default=1.0, help="0~1")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--port", default="1111")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_steps", default=400, type=int)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    scene_filter = parse_scene_ids(args.scene_ids)
    if scene_filter == set():
        raise ValueError("--scene_ids must contain at least one scene id to skip")

    set_seed(args.seed)
    init_distributed_mode(args)

    import os

    geometry_encoder_path = args.geometry_encoder_path or os.environ.get("GEOMETRY_ENCODER_PATH")
    model = AdaGeoVLNInference(
        args.model_path,
        device=f"cuda:{args.local_rank}",
        geometry_encoder_path=geometry_encoder_path or None,
    )
    evaluate(model, args, scene_filter=scene_filter)


if __name__ == "__main__":
    eval_scene()
