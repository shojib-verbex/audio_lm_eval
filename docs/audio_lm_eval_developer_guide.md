# Audio-LM Eval: Unified Evaluation Pipeline for Verbex.ai

**Developer Guide — March 2026**

---

## Why This Matters for Verbex.ai

**Problem:** We evaluate STT, LLM, and TTS models independently using ad-hoc scripts, making it hard to:
- Compare models consistently across datasets
- Reproduce results across team members
- Track improvements over time
- Evaluate the full voice pipeline end-to-end

**Solution:** A single, extensible evaluation framework built on **lmms-eval** (3,700+ GitHub stars, MIT license) that handles all three modalities with standardized metrics, normalization, and reporting.

---

## What It Does

```
                    ┌──────────────────────────────────┐
  Audio Dataset ──> │        audio-lm-eval              │ ──> CER, WER, BLEU,
  (NeMo JSONL,      │                                    │     UTMOS, latency,
   HuggingFace,     │  STT eval  |  LLM eval  | TTS eval│     comparison tables,
   local files)     │  E2E audio-LLM eval                │     per-sample logs
                    │  Cascade: STT → LLM → TTS          │
                    └──────────────────────────────────┘
```

**Three evaluation modes:**
1. **Component isolation** — benchmark STT, LLM, or TTS alone
2. **End-to-end** — evaluate audio-LLMs (Qwen3-Omni, Ultravox) directly
3. **Cascade** — chain STT → LLM → TTS and score each component

---

## Supported Model Backends

| Backend | When to Use | Example |
|---------|-------------|---------|
| **vLLM / vLLM-omni** | Default for supported models. Fastest. | `--model openai --model_args model=Qwen/Qwen3-Omni-30B,base_url=http://localhost:7000/v1` |
| **HuggingFace Transformers** | Models not in vLLM | `--model qwen2_audio --model_args pretrained=Qwen/Qwen2-Audio-7B-Instruct` |
| **OpenAI-compatible API** | Pre-deployed models, commercial APIs | `--model openai --model_args model=gpt-4o` |

**30+ chat models** already integrated: Qwen3-Omni, Qwen2.5-VL, InternVL, LLaVA, Phi-4, GPT-4o, etc.

---

## How We Use It Today — Japanese STT Evaluation

**Current setup:** Qwen3-Omni-30B served via vLLM-omni, evaluated on 7 Japanese call center datasets.

```bash
# Run evaluation
OPENAI_API_KEY=dummy python -m lmms_eval \
  --model openai \
  --model_args model=Qwen/Qwen3-Omni-30B,base_url=http://localhost:7000/v1 \
  --tasks ccr_jp \
  --output_path ./logs/ \
  --log_samples
```

**Tasks in `ccr_jp` group:**
- `ccr_jp_customer` / `ccr_jp_agent` / `ccr_jp_mixed` — call center data (customer side, agent side, mixed)
- `dlight_jp` — 9,101 samples, 5 hrs
- `oshimaya_50hr_jp` / `oshimaya_80hr_jp` — production data
- `rutilea_jp` — 7,610 samples, 5 hrs

---

## Japanese Text Normalization Pipeline

Fair CER/WER comparison requires normalizing both reference and hypothesis:

```
Input text: "５０００円のサービスです。"
     │
     ▼ NFKC normalization (full-width → half-width)
"5000円のサービスです。"
     │
     ▼ Remove punctuation + spaces
"5000円のサービスです"
     │
     ▼ Lowercase
"5000円のサービスです"
     │
     ▼ Digits → Japanese words
"ごせん円のサービスです"
     │
     ▼ Hiragana conversion (fugashi + pykakasi fallback)
"ごせんえんのさーびすです"
     │
     ▼ Remove filler words (えーと, あのー, etc.)
"ごせんえんのさーびすです"
```

**CER/WER calculation:** `editdistance` library (C extension) with corpus-level micro-average aggregation (standard ASR metric: `total_edits / total_ref_chars`).

---

## Evaluating Existing Predictions (No Re-inference)

Already have prediction files from another model (e.g., GPT-4o-transcribe)? Evaluate them with our normalization:

```bash
# Single file
python evaluate_predictions.py \
  --pred_manifest /path/to/model_pred.jsonl

# All prediction files in a directory
python evaluate_predictions.py \
  --pred_dir /path/to/eval_manifests/
```

**Input format** (NeMo manifest with `pred_text`):
```json
{"audio_filepath": "/path/to/audio.wav", "duration": 1.98, "text": "reference text", "pred_text": "model output"}
```

Uses the exact same normalization and metrics as the full pipeline — results are directly comparable.

---

## How to Add a New Dataset

**Step 1:** Prepare a JSONL file with two required fields — `audio_filepath` and `text`:
```json
{"audio_filepath": "/data/audio/001.wav", "text": "こんにちは"}
{"audio_filepath": "/data/audio/002.wav", "text": "ありがとう"}
```
> Optional fields: `duration` (for reporting), `speaker`, or any other metadata. Only `audio_filepath` and `text` are required.

**Step 2:** Create a YAML config in `lmms_eval/tasks/ccr_jp/`:
```yaml
# my_new_dataset.yaml
task: my_new_dataset
include: _default_template_yaml
dataset_kwargs:
  data_files:
    test: /absolute/path/to/manifest.jsonl
```

**Step 3:** Add to the task group in `ccr_jp.yaml`:
```yaml
group: ccr_jp
task:
  - ccr_jp_customer
  - my_new_dataset   # ← add here
```

**Step 4:** Run it:
```bash
python -m lmms_eval --model openai \
  --model_args model=Qwen/Qwen3-Omni-30B,base_url=http://localhost:7000/v1 \
  --tasks my_new_dataset --log_samples
```

That's it. No Python code needed — the `_default_template_yaml` handles audio loading, prompting, normalization, and metrics.

> **Note:** If your task needs custom Python dependencies, add a clear `ImportError` message in your `utils.py` listing the required packages. The user will see it when they run the task without the deps installed.

---

## How to Add a New Model

### Path A: OpenAI-compatible API (zero code)
Any model behind an OpenAI-compatible endpoint (vLLM, vLLM-omni, TGI, etc.):
```bash
python -m lmms_eval \
  --model openai \
  --model_args model=my-model-name,base_url=http://my-server:8000/v1 \
  --tasks ccr_jp
```

### Path B: vLLM-supported model (zero code)
```bash
python -m lmms_eval \
  --model vllm \
  --model_args pretrained=fixie-ai/ultravox-v0_5-llama-3_3-70b,tensor_parallel_size=4 \
  --tasks ccr_jp
```

### Path C: Custom HuggingFace model (~50-100 lines)
Create a model adapter class in `lmms_eval/models/chat/`:
```python
@register_model("my_model")
class MyModel(lmms):
    def generate_until(self, requests):
        # Load audio, call model, return text
```

---

## Model-Specific Prompts

Different models need different prompts. Handled in task YAML — no code changes:

```yaml
# In _default_template_yaml
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: ""
  qwen3_omni:
    pre_prompt: "この日本語の音声を正確に書き起こしてください。日本語のみで出力してください。"
  whisper:
    pre_prompt: ""
    post_prompt: ""
```

The framework auto-selects the right prompt based on the model being evaluated.

---

## Key CLI Options

```bash
python -m lmms_eval \
  --model openai \
  --model_args model=MODEL,base_url=URL \
  --tasks TASK_OR_GROUP \           # Single task, comma-separated, or group name
  --limit 200 \                     # Evaluate only first N samples (for quick testing)
  --output_path ./logs/ \           # Save results
  --log_samples \                   # Save per-sample predictions + metrics
  --batch_size 1 \                  # Requests per batch
  --seed 42                         # Reproducible results
```

**Useful subcommands:**
```bash
lmms-eval tasks          # List all available tasks
lmms-eval models         # List all available models
lmms-eval ui             # Launch web UI for interactive config
lmms-eval serve          # Start HTTP evaluation server
```

---

## Understanding the Output

**Console output:**
```
|   Tasks   |  CER  |  WER  |
|-----------|-------|-------|
| dlight_jp | 0.125 | 0.234 |
| rutilea_jp| 0.098 | 0.187 |
```

**Per-sample logs** (`--log_samples`):
```json
{
  "doc": {"audio_filepath": "...", "text": "reference"},
  "resps": ["model output text"],
  "cer": {"edits": 3, "ref_len": 25},
  "wer": {"edits": 2, "ref_len": 12}
}
```

**Results JSON** (in `--output_path`):
- Aggregate metrics per task
- Model config snapshot
- Timestamp, run metadata

---

## Extending Beyond STT

### LLM Evaluation
280+ built-in benchmarks: MMMU, MMBench, ARC, MathVista, etc.
```bash
python -m lmms_eval --model vllm \
  --model_args pretrained=meta-llama/Llama-3.3-70B-Instruct \
  --tasks mmmu_val,mmbench_en
```

### TTS Evaluation (Planned)
Speech quality metrics via `pip install audio-lm-eval[speech-quality]`:
- **UTMOS** — speech naturalness (MOS prediction)
- **DNSMOS** — signal quality, background noise, overall quality
- **Speaker similarity** — voice cloning fidelity

### Cascade Pipeline (Planned)
Evaluate the full Verbex.ai voice pipeline:
```
Audio in → STT (Whisper) → LLM (Llama) → TTS (XTTS) → Audio out
              ↓ CER/WER       ↓ accuracy      ↓ UTMOS
         per-component scoring at each stage
```

---

## Practical Workflows for the Team

### Quick Model Comparison
```bash
# Evaluate Model A
python -m lmms_eval --model openai \
  --model_args model=ModelA,base_url=http://server1:7000/v1 \
  --tasks ccr_jp --output_path ./logs/ --log_samples

# Evaluate Model B (same datasets, same normalization)
python -m lmms_eval --model openai \
  --model_args model=ModelB,base_url=http://server2:7000/v1 \
  --tasks ccr_jp --output_path ./logs/ --log_samples

# Results are directly comparable — same metrics, same normalization
```

### Test on Subset First
```bash
# Quick sanity check on 50 samples
python -m lmms_eval --tasks dlight_jp --limit 50 --log_samples ...

# Full evaluation when satisfied
python -m lmms_eval --tasks ccr_jp --log_samples ...
```

### Add New Production Data
1. Export call recordings + transcripts as NeMo JSONL
2. Create one YAML file (3 lines)
3. Run eval — immediately get CER/WER with same normalization as all other datasets

### Evaluate External Model Predictions
```bash
# Got predictions from another team/vendor? Score them with our pipeline:
python evaluate_predictions.py --pred_dir /path/to/predictions/
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / Web UI                          │
├─────────────────────────────────────────────────────────────┤
│                     Evaluation Engine                         │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  Tasks    │  │   Models     │  │   Metrics              │ │
│  │ (YAML +  │  │ (30+ chat,   │  │ CER, WER, BLEU,       │ │
│  │  utils)  │  │  90+ simple) │  │ accuracy, LLM-judge,   │ │
│  │          │  │              │  │ UTMOS, latency         │ │
│  └──────────┘  └──────────────┘  └────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Model Backends                            │
│  ┌─────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │  vLLM   │   │ HuggingFace  │   │ OpenAI-compatible    │ │
│  │ offline  │   │ Transformers │   │ API (any endpoint)   │ │
│  └─────────┘   └──────────────┘   └──────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Datasets: HuggingFace Hub | Local JSONL | NeMo Manifests   │
└─────────────────────────────────────────────────────────────┘
```

---

## Resolving Dependencies

### Core install (STT eval with Japanese normalization)
```bash
pip install -e .
pip install editdistance fugashi unidic-lite pykakasi
```

### Missing dependencies
If a task needs packages you don't have, you'll get a clear error:
```
ImportError: ccr_jp task requires additional dependencies:
  pip install editdistance fugashi unidic-lite pykakasi
```
Just run the command it tells you.

### When models have conflicting dependencies
Some models need specific library versions that clash with each other (e.g., SALMONN needs fairseq which pins an old transformers). Solutions:

**Option 1: Separate virtual environments (recommended)**
```bash
# Env A: Qwen3-Omni evaluation
conda create -n eval-qwen python=3.10
pip install -e . && pip install vllm

# Env B: SALMONN evaluation (conflicting fairseq)
conda create -n eval-salmonn python=3.10
pip install -e . && pip install fairseq salmonn-deps

# Both produce the same output format — compare results directly
```

**Option 2: Serve conflicting model as API, evaluate from main env**
```bash
# Terminal 1: serve the conflicting model in its own env
conda activate eval-salmonn
python -m salmonn.server --port 8003

# Terminal 2: evaluate from main env via API backend
conda activate eval-qwen
python -m lmms_eval --model openai \
  --model_args model=salmonn,base_url=http://localhost:8003/v1 \
  --tasks ccr_jp
```

**Option 3: Use pip extras for optional features**
```bash
pip install -e ".[audio]"           # Audio processing (librosa, soundfile)
pip install -e ".[speech-quality]"  # TTS metrics (UTMOS, DNSMOS)
pip install -e ".[metrics]"         # Extended metrics (sacrebleu, pycocoevalcap)
pip install -e ".[server]"          # HTTP eval server (fastapi, uvicorn)
```

### Dependency cheat sheet

| Use case | What to install |
|----------|----------------|
| Japanese STT eval (current) | `editdistance fugashi unidic-lite pykakasi` |
| vLLM-omni backend | `vllm` (in same env) |
| OpenAI API models (GPT-4o) | `openai` (already in core) |
| TTS quality metrics | `pip install -e ".[speech-quality]"` |
| W&B experiment tracking | `wandb` |
| Web UI | `pip install -e ".[server]"` |

### Key principle
All evaluation runs produce the **same output format** regardless of which env or backend you used. So you can evaluate Model A in env-1, Model B in env-2, and directly compare their result JSONs.

---

## What's Done vs What's Planned

| Feature | Status |
|---------|--------|
| STT evaluation (Japanese) | **Done** — 7 datasets, CER/WER, normalization |
| OpenAI-compatible API backend | **Done** — vLLM-omni, GPT-4o, any endpoint |
| Japanese text normalization (hiragana, digits, fillers) | **Done** |
| Corpus-level CER/WER (editdistance-based) | **Done** |
| Per-sample logging | **Done** |
| Prediction file evaluation (no re-inference) | **Done** |
| Adding new datasets (YAML-only) | **Done** |
| LLM benchmarks (280+ tasks) | **Available** — built into framework |
| TTS quality metrics (UTMOS, DNSMOS) | **Planned** |
| Cascade pipeline (STT → LLM → TTS) | **Planned** |
| Multi-turn conversation eval | **Planned** |
| Robustness testing (noise, reverb) | **Planned** |
| W&B integration | **Available** — built into framework |

---

## Getting Started

```bash
# 1. Clone and install
cd /raid/ASR/Shojib/audio_lm_eval
pip install -e .
pip install editdistance fugashi unidic-lite pykakasi

# 2. Start your model server (example: vLLM-omni)
docker run -p 7000:7000 vllm-omni --model Qwen/Qwen3-Omni-30B

# 3. Run evaluation
OPENAI_API_KEY=dummy python -m lmms_eval \
  --model openai \
  --model_args model=Qwen/Qwen3-Omni-30B,base_url=http://localhost:7000/v1 \
  --tasks ccr_jp \
  --output_path ./logs/ \
  --log_samples

# 4. Add your own dataset: create one YAML file, done.
```

**Key files to know:**
- `lmms_eval/tasks/ccr_jp/utils.py` — normalization + metrics logic
- `lmms_eval/tasks/ccr_jp/*.yaml` — task configs
- `lmms_eval/protocol.py` — audio encoding for API transport
- `evaluate_predictions.py` — score existing prediction files
