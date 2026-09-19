"""Shard-based multi-process launcher target for VLN evaluation.

Each invocation evaluates exactly one deterministic dataset shard and writes to
its own directory.  The companion shell script launches several invocations in
parallel and then calls this module once more with ``--merge-only``.

The model and rollout implementation are intentionally imported from
``evaluation.py`` so the single-process and shard-based evaluators cannot drift.
"""

import argparse
import json
import os
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-num", type=int, required=True, help="Total number of dataset shards")
    parser.add_argument("--split-id", type=int, help="Zero-based shard handled by this process")
    parser.add_argument("--merge-only", action="store_true", help="Merge completed shard results")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--geometry_encoder_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default="config/vln_r2r.yaml")
    parser.add_argument("--eval_split", type=str, default="val_unseen")
    parser.add_argument("--output_path", type=str, default="./evaluation/adageovln_multiprocess")
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--save_video_ratio", type=float, default=0.5, help="0~1")
    parser.add_argument("--max_steps", default=400, type=int)
    parser.add_argument(
        "--max-episodes-total",
        default=0,
        type=int,
        help="Evaluate only the first N episodes before sharding; 0 evaluates all episodes",
    )
    parser.add_argument("--seed", type=int, default=42)

    # VLNEvaluator uses these fields when configuring the local Habitat process.
    # CUDA_VISIBLE_DEVICES makes the assigned physical GPU appear as cuda:0.
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--device", default="cuda")
    return parser


def shard_dir(output_path: str, split_id: int) -> Path:
    return Path(output_path) / "shards" / f"shard_{split_id:04d}"


def aggregate(records):
    length = len(records)
    return {
        "sucs_all": sum(float(record["success"]) for record in records) / length if length else 0.0,
        "spls_all": sum(float(record["spl"]) for record in records) / length if length else 0.0,
        "oss_all": sum(float(record["os"]) for record in records) / length if length else 0.0,
        "ones_all": sum(float(record["ne"]) for record in records) / length if length else 0.0,
        "length": length,
    }


def read_episode_records(path: Path):
    if not path.exists():
        return []
    records = []
    with path.open("r") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if "sucs_all" not in record:
                records.append(record)
    return records


def completed_episode_count(output_path: Path) -> int:
    """Return the number of resumable episode records in a shard output."""
    return len(read_episode_records(output_path / "result.json"))


def merge_shards(output_path: str, split_num: int):
    all_records = {}
    missing = []
    for split_id in range(split_num):
        result_path = shard_dir(output_path, split_id) / "result.json"
        if not result_path.exists():
            missing.append(str(result_path))
            continue
        for record in read_episode_records(result_path):
            key = (
                str(record["scene_id"]),
                str(record["episode_id"]),
                record.get("episode_instruction", ""),
            )
            if key in all_records:
                raise RuntimeError(f"Episode appears in multiple shards: {key}")
            all_records[key] = record

    if missing:
        raise FileNotFoundError("Missing shard result(s):\n  " + "\n  ".join(missing))

    records = [all_records[key] for key in sorted(all_records)]
    summary = aggregate(records)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    temporary_path = output_dir / "result.json.tmp"
    with temporary_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        handle.write(json.dumps(summary) + "\n")
    os.replace(temporary_path, result_path)
    print(f"Merged {split_num} shards into {result_path}")
    print(summary)


def evaluate_shard(args):
    # Keep merge/help usable on machines without the evaluation environment.
    # Model dependencies are only imported by actual worker processes.
    from habitat import Env
    from habitat.datasets import make_dataset
    from evaluation import AdaGeoVLNInference, VLNEvaluator, set_seed

    class DatasetShardVLNEvaluator(VLNEvaluator):
        """Apply a global episode limit, then take one disjoint strided shard."""

        def config_env(self):
            dataset = make_dataset(
                id_dataset=self.config.habitat.dataset.type,
                config=self.config.habitat.dataset,
            )
            if args.max_episodes_total > 0:
                dataset.episodes = dataset.episodes[:args.max_episodes_total]
            dataset.episodes = dataset.episodes[args.split_id::args.split_num]
            if not dataset.episodes:
                raise ValueError(
                    f"Shard {args.split_id}/{args.split_num} has no episodes. "
                    "Reduce --split-num or increase --max-episodes-total."
                )
            print(
                f"Shard {args.split_id}/{args.split_num}: "
                f"evaluating {len(dataset.episodes)} episodes"
            )
            return Env(config=self.config, dataset=dataset)

    if args.split_id is None:
        raise ValueError("--split-id is required unless --merge-only is used")
    if not 0 <= args.split_id < args.split_num:
        raise ValueError(f"--split-id must be in [0, {args.split_num}), got {args.split_id}")
    if not args.model_path:
        raise ValueError("--model_path is required for shard evaluation")

    set_seed(args.seed)
    current_shard_dir = shard_dir(args.output_path, args.split_id)
    current_shard_dir.mkdir(parents=True, exist_ok=True)
    completed = completed_episode_count(current_shard_dir)
    if completed:
        print(
            f"[shard {args.split_id}] Resume enabled: found {completed} completed "
            "episodes; they will be skipped.",
            flush=True,
        )
    else:
        print(f"[shard {args.split_id}] No completed episodes found; starting fresh.", flush=True)
    geometry_encoder_path = args.geometry_encoder_path or os.environ.get("GEOMETRY_ENCODER_PATH")
    model = AdaGeoVLNInference(
        args.model_path,
        device="cuda:0",
        geometry_encoder_path=geometry_encoder_path or None,
    )
    evaluator = DatasetShardVLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        # DatasetShardVLNEvaluator already performed global sharding.  Disable
        # VLNEvaluator's additional per-scene striding.
        env_num=1,
        output_path=str(current_shard_dir),
        model=model,
        epoch=0,
        args=args,
    )
    sucs, spls, oss, ones, _ = evaluator.eval_action(0)
    summary = {
        "sucs_all": sucs.float().mean().item() if len(sucs) else 0.0,
        "spls_all": spls.float().mean().item() if len(spls) else 0.0,
        "oss_all": oss.float().mean().item() if len(oss) else 0.0,
        "ones_all": ones.float().mean().item() if len(ones) else 0.0,
        "length": len(sucs),
        "split_id": args.split_id,
        "split_num": args.split_num,
    }
    with (current_shard_dir / "result.json").open("a") as handle:
        handle.write(json.dumps(summary) + "\n")
    print(f"Completed shard {args.split_id}/{args.split_num}: {summary}")


def main():
    args = build_parser().parse_args()
    if args.split_num < 1:
        raise ValueError("--split-num must be at least 1")
    if args.merge_only:
        merge_shards(args.output_path, args.split_num)
    else:
        evaluate_shard(args)


if __name__ == "__main__":
    main()
