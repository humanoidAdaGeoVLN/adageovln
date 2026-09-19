<div align="center">

<a href="https://humanoid-research.github.io/adageovln/">
  <img src="https://humanoid-research.github.io/adageovln/assets/branding/adageovln-wordmark.png" alt="AdaGeoVLN" width="320">
</a>

<h2>AdaGeoVLN: Selective Geometry Across Representation Depth<br>and Navigation Time for Vision-Language Navigation</h2>

<p><strong>Anonymous Authors</strong></p>

<p>
  <a href="https://huggingface.co/adageovln/adageovln_base"><img src="https://img.shields.io/badge/Checkpoint-AdaGeoVLN_Base-ffcc4d?style=flat-square&amp;logo=huggingface&amp;logoColor=white" alt="Hugging Face checkpoint"></a>
  <a href="https://humanoid-research.github.io/adageovln/"><img src="https://img.shields.io/badge/Website-AdaGeoVLN-2563eb?style=flat-square&amp;logo=googlechrome&amp;logoColor=white" alt="Website"></a>
  <a href="#setup"><img src="https://img.shields.io/badge/Quick_Start-Setup-16a34a?style=flat-square" alt="Quick start"></a>
</p>

</div>

<p align="center"><em>Geometry across depth. Memory across time. Navigation from a single RGB stream.</em></p>

---

## Overview

**AdaGeoVLN** is a streaming vision-language navigation framework that selects geometry
across two complementary axes: **representation depth** and **navigation time**. It pairs a
Qwen3.5-4B navigation policy with a frozen VGGT geometry encoder, using a single RGB stream
and no additional navigation training samples beyond R2R/RxR.

- **Hierarchical geometry fusion:** couple intermediate VGGT representations to successive
  policy stages, exposing multiple geometric depths to navigation reasoning.
- **Navigation-aware memory:** retain historical VGGT global-attention KV states according
  to instruction relevance, geometric confidence, and transition novelty under a bounded budget.

## TODO

- ⬜ Release training code.
- ✅ Release inference and evaluation code.
- ✅ Release the AdaGeoVLN model checkpoint.

### Architecture

<p align="center">
  <img src="https://humanoid-research.github.io/adageovln/assets/figures/geometry-fusion.jpg" alt="Hierarchical geometry fusion across VGGT and VLM stages" width="45%">
  <img src="https://humanoid-research.github.io/adageovln/assets/figures/geometry-memory.jpg" alt="Navigation-aware geometry memory with three retention signals" width="53%">
</p>


### Evaluation scripts

Evaluation launch scripts live in `scripts/evaluation/`. Run them from the repository
root after activating the runtime environment. Each script adds `src/` to `PYTHONPATH`.

| Script | Purpose |
| :--- | :--- |
| `inference.sh` | Model inference with one process per GPU and a scene skip list |
| `inference_scene.sh` | Model inference with a scene skip list |
| `inference_parallel.sh` | Model inference with multiple processes per GPU |
| `token_eviction.sh` | AdaGeoVLN navigation-aware geometric memory |
| `benchmark_inference.sh` | Compare sequential and parallel inference throughput |
| `eval_benchmarks.sh` | Evaluate general vision-language benchmarks with lmms_eval |

### Inference and retention

As an episode grows, VGGT's global-attention KV cache grows with it, and keeping the full
history becomes the memory bottleneck.

This repository provides **two evaluation methods**:

| Method | What it does | Entry point |
|---|---|---|
| **1. Model inference** | Runs the policy with the full streaming cache, optionally trimmed by a fixed start-and-recent window | `scripts/evaluation/inference*.sh` |
| **2. Navigation-Aware Geometric Memory** | Runs the policy with navigation-aware KV retention under a fixed token budget | `scripts/evaluation/token_eviction.sh` |

Navigation-aware memory retention is training-free. Every cached patch token is scored by three signals,
standardised per layer and combined additively:

| Signal | Meaning | Computed |
|---|---|---|
| Instruction relevance | cosine between a projected geometry token and its closest instruction segment | once, when the frame enters the cache |
| Geometric confidence | `min` of VGGT's depth- and point-map confidence, calibrated to `[0,1)` | once, when the frame enters the cache |
| Transition novelty | how distinct a frame's viewpoint is from every other observed frame | recomputed at every step |

Selection touches only the VGGT global-attention cache. The Qwen cache and the CPU buffer of
frame-aligned geometry features are left untouched.

### Where the code lives

| Path | Contents |
|---|---|
| `src/qwen_vl/model/vggt/eviction/vln_segment_transition.py` | the three scoring signals and their combination |
| `src/qwen_vl/model/geometry_encoders/vggt_encoder.py` | streaming KV cache, per-layer budgets, top-k selection |
| `src/evaluation.py`, `src/evaluation_scene.py`, `src/evaluation_multiprocess.py` | evaluation entry points |

---

## Setup

> This public branch documents Qwen3.5 only.
> The Habitat setup follows [NAVIDA](https://github.com/waynechu1021/NAVIDA) with
> **Python 3.10**, **Habitat-Sim 0.2.4**, and **Habitat-Lab 0.2.4**.
> The model stack uses **PyTorch 2.8.0+cu128** and **flash_attn 2.8.3**.
> Run all commands from the repository root.

### 1. Create conda environment

```bash
conda create -n adageovln-qwen35 python=3.10 -y
conda activate adageovln-qwen35
```

<details>
<summary>Install Miniconda first (if needed)</summary>

```bash
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/bin/activate"
```
</details>

### 2. Install PyTorch for CUDA 12.8

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

> Ensure CUDA 12.8 is available before installing `flash_attn`.

### 3. Install Habitat-Sim and Habitat-Lab v0.2.4

Following the source-installation procedure from
[NAVIDA](https://github.com/waynechu1021/NAVIDA):

```bash
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install --headless
cd ..

git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
cd ..
```

### 4. Install flash_attn with uv

```bash
pip install uv
uv pip install flash-attn==2.8.3 --no-build-isolation
```

### 5. Install Qwen3.5 and runtime dependencies

```bash
pip install transformers==5.3.0 accelerate
pip install qwen_vl_utils==0.0.14 decord
pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation
pip install -U git+https://github.com/fla-org/flash-linear-attention
pip install gym fastdtw dtw

# Pin these versions to avoid the NumPy and opencv-python conflict.
pip install numpy==1.26.4 opencv-python==4.11.0.86
```

### 6. Make the source packages available

For direct Python commands, add the repository's `src/` directory to the current shell:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

The evaluation launch scripts set this automatically.

### 7. Verify installation

```bash
python - <<'PY'
import habitat, habitat_sim
import torch, transformers, qwen_vl_utils, causal_conv1d, fla, decord
print("habitat-sim", habitat_sim.__version__)
print("habitat-lab", getattr(habitat, "__version__", "0.2.4"))
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("qwen_vl_utils", qwen_vl_utils.__version__)
print("causal_conv1d", causal_conv1d.__version__)
print("fla", fla.__version__)
print("decord", decord.__version__)
PY
```

<details>
<summary>Additional packages for lmms_eval (optional)</summary>

```bash
pip install datasets pyarrow evaluate pytablewriter pandas \
  loguru jsonlines sqlitedict sacrebleu terminaltables zss tenacity==8.3.0 \
  wandb openai tiktoken scipy openpyxl numexpr sympy nltk sentencepiece ftfy \
  timm opencv-python-headless av tqdm-multiprocess transformers-stream-generator \
  hf_transfer
```
</details>

---

## Data and checkpoints

All paths in `config/*.yaml` and in the scripts are relative to the repository root:

```
AdaGeoVLN/
  checkpoints/
    adageovln/      # navigation policy
    VGGT-1B/                     # geometry encoder
  data/
    scene_datasets/              # Matterport3D scenes
    datasets/
      r2r/{split}/{split}.json.gz
      rxr/{split}/{split}_{role}.json.gz
```

Override them with environment variables instead of editing files:

```bash
export CHECKPOINT=/path/to/policy
export GEOMETRY_ENCODER_PATH=/path/to/VGGT-1B
```

---

## Method 1 — Model inference

Runs the policy without navigation-aware eviction. The streaming cache is either kept in
full or trimmed by a fixed **start-and-recent** window, which also serves as the temporal
baseline in the paper.

**Habitat config:** `config/vln_r2r.yaml` (R2R) or `config/vln_rxr.yaml` (RxR)

```bash
# single node, one process per visible GPU
bash scripts/evaluation/inference.sh

# skip specific scenes
SCENE_IDS=EU6Fwq7SyZv bash scripts/evaluation/inference_scene.sh

# several processes per GPU (throughput)
PROCESSES_PER_GPU=2 bash scripts/evaluation/inference_parallel.sh

# RxR instead of R2R
CONFIG=config/vln_rxr.yaml bash scripts/evaluation/inference.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `CONFIG` | `config/vln_r2r.yaml` | Habitat task config |
| `CHECKPOINT` | `checkpoints/adageovln` | policy weights |
| `GEOMETRY_ENCODER_PATH` | `checkpoints/VGGT-1B` | geometry encoder weights |
| `EVAL_SPLIT` | `val_unseen` | dataset split |
| `OUTPUT_PATH` | `evaluation/vln` | where `result.json` is written |
| `SCENE_IDS` | `EU6Fwq7SyZv` | comma-separated scenes to **skip** in `inference.sh` and `inference_scene.sh`; the parallel launcher evaluates all scenes |
| `VGGT_KV_START` / `VGGT_KV_RECENT` | `8` / `48` (`8` / `56` for `inference_scene.sh`) | start-and-recent window over the geometry cache |
| `VLN_PROJECTED_GEOMETRY_CACHE` | `1` | reuse projected geometry across steps; `0` recomputes the fusion projection |
| `SAVE_VIDEO` | `1` | dump episode videos |
| `NPROC_PER_NODE` | all visible GPUs | processes for `torchrun` |

To evaluate all scenes with the scene launcher, use
`SCENE_IDS=none bash scripts/evaluation/inference_scene.sh`.
Its default output directory is `evaluation/scene/${SCENE_IDS}`; the parallel launcher
uses `evaluation/vln_multiprocess`.

To compare one process against several processes on the same GPU, run:

```bash
GPU_ID=0 BENCHMARK_EPISODES=10 PARALLEL_PROCESSES=2 \
  bash scripts/evaluation/benchmark_inference.sh
```

This evaluates the same episode subset in both modes and prints runtime, episodes per
second, and speedup. Logs and results are saved under `evaluation_benchmark/`.

---

## Method 2 — Navigation-Aware Geometric Memory

Runs the policy with navigation-aware KV retention under a fixed token budget.

**Habitat config:** `config/vln_r2r.yaml` &nbsp;&nbsp; **Method config:** `config/adageovln.json`

```bash
bash scripts/evaluation/token_eviction.sh
```

```bash
# a different budget
VGGT_TOTAL_BUDGET=750000 bash scripts/evaluation/token_eviction.sh
```

To reproduce a leave-one-signal-out ablation, copy `config/adageovln.json`, zero one entry
in `score_weights`, and renormalise the remaining two to sum to 1.

| Variable | Default | Meaning |
|---|---|---|
| `VLN_SEGMENT_TRANSITION_WEIGHTS_PATH` | `config/adageovln.json` | signal weights and retention options |
| `VGGT_TOTAL_BUDGET` | `900000` | token capacity summed over all 24 global-attention layers |
| `VGGT_BUDGET_PROPORTIONS_PATH` | `config/kv_budget_proportions_cosine.json` | how that budget is split per layer |
| `USE_ADAGEO_KV_CACHE` | `1` | enables the budgeted cache; `0` falls back to Method 1 |
| `ADAGEO_KV_SCORE_MODE` | `vln_segment_transition` | scoring rule; `importance` selects the cosine-importance baseline |
| `CONFIG` | `config/vln_r2r.yaml` | Habitat task config |
| `MAX_STEPS` | `400` | step cap per episode |
| `CHECKPOINT` | `checkpoints/adageovln` | policy weights |
| `SCENE_IDS` | `none` | comma-separated scenes to **skip**; `none` evaluates all |
| `OUTPUT_PATH` | `evaluation/adageovln` | where `result.json` is written |

The budget is an aggregate: `900000` is `B_total` over all layers, not a count of unique scene
tokens. Layer `g` receives `floor(rho_g * B_total)` tokens.

`VGGT_KV_START` and `VGGT_KV_RECENT` have **no effect** in this mode — start-and-recent
trimming only runs when `USE_ADAGEO_KV_CACHE=0`.

---

## Configuration files

### Retention configs (Method 2)

| File | Configuration |
|---|---|
| `config/adageovln.json` | the method: equal weights on the three signals, per-frame centering, novelty recomputed each step |
| `config/kv_budget_proportions_cosine.json` | per-layer budget proportions derived from cosine similarity |

Fields of a retention config:

```jsonc
{
  "score_weights":  { "confidence": 0.33, "instruction": 0.33, "transition": 0.33 },
  "confidence":     { "merge": "min" },          // depth and point confidence combined by min
  "instruction":    { "centering": "both" },     // centre visual tokens and segment embeddings
  "normalization":  { "candidate_terms": "zscore" },
  "transition":     { "descriptor_pooling": "uniform",
                      "novelty_reduce": "max",
                      "novelty_ref": "cache" }   // "cache" recomputes novelty every step
}
```

### Habitat configs

| File | Task |
|---|---|
| `config/vln_r2r.yaml` | R2R, used by both methods |
| `config/vln_rxr.yaml` | RxR (`languages: ["en-US", "en-IN"]`) |

---

## 🙏 Acknowledgements

Our work builds on [SpatialStack](https://spatial-stack.github.io/), [JanusVLN](https://github.com/MIV-XJTU/JanusVLN), [VGGT](https://github.com/facebookresearch/vggt), [GHOST](https://arxiv.org/pdf/2605.15852), Qwen3.5, and [Habitat](https://github.com/facebookresearch/habitat-lab). We sincerely thank the authors for their work.
