# Qwen3-Omni Japanese Baseline Evaluation Plan

## Objective

Establish a comprehensive, reproducible baseline for **Qwen3-Omni-30B-A3B-Instruct** across Japanese language understanding, speech recognition, and dialogue capabilities. This baseline serves as the reference point for tracking fine-tuning progress and comparing against GPT-4o.

## Scope

- **Models under evaluation**: Qwen3-Omni-30B-A3B-Instruct, GPT-4o (API)
- **Evaluation dimensions**: Text-only LLM, ASR (CER/WER), dialogue/conversation, audio-conditioned LLM
- **Compute**: Up to 4x H100 GPUs (`device_map=auto`)
- **Framework**: `lmms-eval` (this repo), with new task configs added per phase

## Why This Order

Phases are ordered by **value-per-effort**:
1. Public text LLM benchmarks give immediately comparable numbers (vs Nejumi/Swallow leaderboards) with minimal implementation effort
2. Public ASR benchmarks fill the gap identified in research — no published Qwen3-Omni scores on Japanese-specific ASR test sets
3. In-house dialogue tasks test telephony-specific capabilities using local data
4. Audio-conditioned LLM tasks quantify the text-vs-audio accuracy gap, answering whether the audio encoder or LLM is the bottleneck

---

## Phase 1: Japanese Text LLM Benchmarks

**Goal**: Measure Qwen3-Omni's Japanese language understanding as a text LLM, comparable to Nejumi 4 leaderboard scores.

**Effort estimate**: ~1 week

### Tasks

| Task | HF Dataset | Test Size | Format | Metric | Leaderboard Reference |
|------|-----------|-----------|--------|--------|-----------------------|
| JMMLU | `nlp-waseda/JMMLU` | ~5,500 (53 subjects) | 4-way MCQ | Accuracy | Nejumi 4, Swallow v2 |
| JCommonsenseQA | `leemeng/jcommonsenseqa` | 1,119 | 5-way MCQ | Accuracy | Nejumi 4 |
| JNLI | `shunk031/JGLUE` (JNLI subset) | 2,263 | 3-way classification | Accuracy | Nejumi 4 |
| MGSM-JA | `juletxara/mgsm` (ja subset) | 250 | Open-ended math | Exact match accuracy | Nejumi 4 |

### File Structure

```
lmms_eval/tasks/
├── jmmlu/
│   ├── _default_template_yaml       # Shared config: dataset_path, output_type, metrics
│   ├── jmmlu.yaml                   # Group definition aggregating all subjects
│   ├── jmmlu_{subject}.yaml         # Per-subject configs (53 files, auto-generated)
│   ├── _generate_configs.py         # Script to generate per-subject YAMLs
│   └── utils.py                     # doc_to_text (JP MCQ format), process_results
├── jcommonsenseqa/
│   ├── jcommonsenseqa.yaml
│   └── utils.py
├── jnli/
│   ├── jnli.yaml
│   └── utils.py
└── mgsm_ja/
    ├── mgsm_ja.yaml
    └── utils.py
```

### Implementation Notes

**JMMLU** — Adapt from existing `lmms_eval/tasks/mmlu/` structure:
- Use `output_type: multiple_choice` with `doc_to_choice: ["A", "B", "C", "D"]`
- Japanese prompt format: `"{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\n答え:"`
- Group subjects into STEM / Humanities / Social Sciences / Other for aggregate scores
- Auto-generate per-subject YAMLs with `_generate_configs.py` (same pattern as MMLU)

**JCommonsenseQA** — 5-choice commonsense reasoning:
- `output_type: multiple_choice` with 5 options
- Prompt: `"質問: {{question}}\n選択肢: {{choices}}\n答え:"`
- Straightforward accuracy computation

**JNLI** — Natural language inference (entailment / contradiction / neutral):
- `output_type: generate_until`
- Prompt: `"前提: {{premise}}\n仮説: {{hypothesis}}\nこの前提から仮説は「含意」「矛盾」「中立」のいずれですか？"`
- Map model output to 3 labels, compute accuracy

**MGSM-JA** — Math word problems in Japanese:
- `output_type: generate_until`
- Extract numeric answer from model output (regex: final number)
- Compare against ground truth with exact match

### Run Commands

```bash
# Full Phase 1 evaluation — Qwen3-Omni (via vLLM server at localhost:7000)
OPENAI_API_KEY=dummy python -m lmms_eval \
  --model openai \
  --model_args model_version=Qwen/Qwen3-Omni-30B-A3B-Instruct,base_url=http://localhost:7000/v1,num_concurrent=32 \
  --tasks japanese_llm_eval \
  --batch_size 32 \
  --log_samples \
  --output_path ./logs/qwen3_omni_llm

# Full Phase 1 evaluation — GPT-4o
python -m lmms_eval \
  --model openai \
  --model_args model_version=gpt-4o \
  --tasks japanese_llm_eval \
  --batch_size 32 \
  --log_samples \
  --output_path ./logs/gpt4o_llm
```

### Results (Qwen3-Omni-30B-A3B-Instruct — Pretrained, No Fine-tuning)

| Task | Metric | Qwen3-Omni | Qwen2.5-72B | Llama 3.1-70B | Notes |
|------|--------|-----------|-------------|---------------|-------|
| JMMLU | exact_match ↑ | **0.7850** | 0.7899 | 0.7323 | Near 72B-class with only 3B active params |
| JCommonsenseQA | exact_match ↑ | **0.9258** | 0.9696 | 0.9482 | Human baseline: 0.986 |
| JNLI | acc ↑ | **0.7634** | — | — | Few published LLM baselines |
| MGSM-JA | exact_match ↑ | **0.8480** | 0.8360 | 0.7440 | Beats Qwen2.5-72B |

**Status**: COMPLETE

**Key findings**:
- JMMLU 78.5% nearly matches Qwen2.5-72B (79.0%) despite only 3B active parameters
- MGSM-JA 84.8% actually exceeds Qwen2.5-72B (83.6%) on Japanese math
- The omni-model backbone does not sacrifice Japanese text understanding

---

## Phase 2: Japanese ASR — Public Benchmarks

**Goal**: Produce CER numbers on public Japanese ASR test sets, directly comparable to published results for ReazonSpeech-k2-v2, Whisper large-v3, and Kotoba-Whisper.

**Effort estimate**: ~3-4 days

### Tasks

| Task | Dataset | Test Size | Metric | Published Baselines |
|------|---------|-----------|--------|-------------------|
| CCR-JP (existing) | In-house (dlight, oshimaya, rutilea) | 4 variants | CER | Internal only |
| CommonVoice JA | `mozilla-foundation/common_voice_17_0` (ja) | ~3,000 | CER | ReazonSpeech: 7.85, Whisper-v3: 8.18 |
| ReazonSpeech test | `reazon-research/reazonspeech` (test split) | ~1,000 | CER | ReazonSpeech-k2-v2: 9.09, Whisper-v3: 14.9 |

### File Structure

```
lmms_eval/tasks/
├── common_voice_ja/
│   ├── _default_template_yaml
│   ├── common_voice_ja.yaml
│   └── utils.py                     # Reuse ccr_jp Japanese CER normalization
├── reazonspeech/
│   ├── _default_template_yaml
│   ├── reazonspeech.yaml
│   └── utils.py                     # Reuse ccr_jp Japanese CER normalization
├── japanese_asr_eval.yaml           # Group: ccr_jp + common_voice_ja + reazonspeech
└── ccr_jp/                          # Already exists — no changes needed
```

### Implementation Notes

- **Japanese CER normalization** pipeline already exists in `lmms_eval/tasks/ccr_jp/utils.py` — reuse directly:
  - NFKC Unicode normalization
  - Japanese punctuation removal
  - Kanji/katakana to hiragana conversion (fugashi + pykakasi fallback)
  - Digit to hiragana word conversion
  - Character-level edit distance via `editdistance`
- **CommonVoice JA**: Extend existing `common_voice_15/` pattern (currently supports en, fr, zh-CN) — add `ja` config
- **ReazonSpeech**: HF dataset `reazon-research/reazonspeech` — new task with same normalization pipeline
- **Audio resampling**: Qwen3-Omni expects 16kHz; CommonVoice is 48kHz, ReazonSpeech is 16kHz — model adapter already handles resampling

### Run Commands

```bash
# Full Phase 2 ASR evaluation — Qwen3-Omni (all tasks via group)
python -m lmms_eval \
  --model qwen3_omni \
  --model_args pretrained=Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --tasks japanese_asr_eval \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/qwen3_omni_asr

# Individual public ASR benchmarks only
python -m lmms_eval \
  --model qwen3_omni \
  --model_args pretrained=Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --tasks common_voice_ja,reazonspeech \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/qwen3_omni_asr

# ASR evaluation — GPT-4o (audio via API)
python -m lmms_eval \
  --model openai \
  --model_args model=gpt-4o-audio-preview \
  --tasks common_voice_ja,reazonspeech \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/gpt4o_asr
```

### CCR-JP Results (Qwen3-Omni-30B-A3B-Instruct — Pretrained, No Fine-tuning)

| Task | CER ↓ | WER ↓ |
|------|-------|-------|
| ccr_jp_agent | **0.0692** | 0.1142 |
| ccr_jp_customer | **0.1840** | 0.2354 |
| ccr_jp_mixed | **0.1148** | 0.1618 |
| dlight_jp | **0.1023** | 0.1621 |
| oshimaya_50hr_jp | **0.2048** | 0.2730 |
| oshimaya_80hr_jp | **0.1807** | 0.2560 |
| rutilea_jp | **0.1064** | 0.1860 |

**Key findings**:
- Agent speech (clearer audio) achieves best CER at 6.9%
- Customer speech is harder (18.4% CER) — noisier, more informal
- Oshimaya datasets show highest error rates (~18-20% CER) — likely domain/audio quality differences

### Expected Outcome

First published Qwen3-Omni CER numbers on Japanese-specific ASR test sets. The Qwen3-Omni technical report only publishes FLEURS 19-language aggregate (5.33 WER) — these results fill that gap.

---

## Phase 3: In-house Dialogue LLM Tasks

**Goal**: Test conversation-specific capabilities using local Japanese dialogue data (JDD, BSD, CCR). These measure telephony-relevant skills: topic classification, response generation, summarization, information extraction.

**Effort estimate**: ~1 week

**Prerequisite**: Preprocessing scripts to convert raw data into evaluation manifests.

### Tasks

| Task | Datasets | Size | Metric | Priority |
|------|----------|------|--------|----------|
| `jp_topic_classify` | JDD (5,200) + BSD test (69) | ~5,269 | Accuracy + Macro F1 | P0 |
| `jp_dialogue_response` | JDD (5,200) + BSD test (69) | ~5,269 | LLM-as-judge (GPT-4o, 1-5) | P0 |
| `jp_dialogue_summary` | BSD test (69) + CCR full (200) | ~269 | LLM-as-judge (GPT-4o, 1-10) | P0 |
| `jp_info_extract` | CCR full (200) + Oshimaya 80hr-full (256) | ~456 | LLM-as-judge (GPT-4o, 1-10) | P1 |
| `jp_dialogue_translate` | BSD test bilingual (69) | 69 | BLEU-4 | P2 |

### Data Sources

| Dataset | Location | Format | Notes |
|---------|----------|--------|-------|
| JDD | `/raid/ASR/ibrahim/japanese_lm_data/japanese_daily_dialogue/data/topic{1..5}.json` | JSON, 5 topic files | Clean kanji text, 5,261 dialogues |
| BSD | `/raid/ASR/ibrahim/japanese_lm_data/bsd/{train,dev,test}.json` | JSON, bilingual JA+EN | 69 test dialogues, business domain |
| CCR Full | `/raid/ASR/asr-ja-eval-datasets/ccr_202306_disc1_phase1_200_calls/full_audios_manifest.jsonl` | JSONL, NeMo format | 200 full call recordings with transcripts |
| Oshimaya 80hr-full | `/raid/ASR/ibrahim/japanese_eval_data_inhouse/oshimaya-verbex-prod-data-80-hr-full-audios_eval_manifest.json` | JSONL, hiragana only | 256 full conversations |

### Preprocessing Pipeline

```
tools/jp_llm_benchmark_prep/
├── prepare_jdd.py       # JDD → jp_dialogue_response, jp_topic_classify manifests
├── prepare_bsd.py       # BSD → jp_dialogue_response, jp_dialogue_summary, jp_topic_classify, jp_dialogue_translate manifests
├── prepare_ccr.py       # CCR → jp_dialogue_summary, jp_info_extract manifests
└── prepare_oshimaya.py  # Oshimaya → jp_info_extract manifests
```

Output: JSONL manifests in `/raid/ASR/ibrahim/japanese_lm_data/manifests/`

### File Structure

```
lmms_eval/tasks/
├── jp_topic_classify/
│   ├── _default_template_yaml
│   ├── jp_topic_classify.yaml
│   ├── jp_topic_classify_jdd.yaml
│   ├── jp_topic_classify_bsd.yaml
│   └── utils.py
├── jp_dialogue_response/
│   ├── _default_template_yaml
│   ├── jp_dialogue_response.yaml
│   ├── jp_dialogue_response_jdd.yaml
│   ├── jp_dialogue_response_bsd.yaml
│   └── utils.py
├── jp_dialogue_summary/
│   ├── _default_template_yaml
│   ├── jp_dialogue_summary.yaml
│   ├── jp_dialogue_summary_bsd.yaml
│   ├── jp_dialogue_summary_ccr.yaml
│   └── utils.py
└── jp_info_extract/
    ├── _default_template_yaml
    ├── jp_info_extract.yaml
    ├── jp_info_extract_ccr.yaml
    ├── jp_info_extract_oshimaya.yaml
    └── utils.py
```

### LLM-as-Judge Details

For `jp_dialogue_response`, `jp_dialogue_summary`, and `jp_info_extract`, scoring uses GPT-4o as judge via the pattern established in `lmms_eval/tasks/ami/utils.py`.

**Judge prompt for dialogue response** (1-5 scale):
```
あなたは日本語の会話の品質を評価するアシスタントです。
以下の会話の履歴と、モデルが生成した次の発話を評価してください。

[会話履歴]:
{history}

[参照応答]:
{reference}

[生成応答]:
{prediction}

以下の基準で1-5のスコアで評価してください:
1: 会話の流れに全く合わない、不自然な応答
2: 一部関連性があるが、大きな問題がある
3: 概ね適切だが、自然さや関連性に改善の余地がある
4: 自然で適切な応答、わずかな改善点がある
5: 非常に自然で、文脈に完全に適合した優れた応答

スコアのみを出力してください。
```

**Judge prompt for summarization** (1-10 scale):
```
あなたは会話要約の品質を評価するアシスタントです。
以下の会話とその要約を評価してください。

[会話]:
{conversation}

[要約]:
{summary}

以下の4つの観点で、それぞれ1-10のスコアで評価してください:
- 完全性: 重要な情報がすべて含まれているか
- 正確性: 事実に誤りがないか
- 簡潔性: 不必要な情報が含まれていないか
- 流暢性: 自然で読みやすい日本語か

JSON形式で出力: {"completeness": N, "accuracy": N, "conciseness": N, "fluency": N}
```

### Design Decisions

1. **No hiragana normalization for LLM tasks** — unlike ASR, LLM outputs should produce natural Japanese with kanji/katakana
2. **Reference-free LLM judge for summarization** — creating ground-truth summaries is expensive; the judge evaluates against the source conversation directly
3. **Text-only first** — JDD and BSD are text-only; add audio variants (CCR, Oshimaya) in Phase 4
4. **BSD test set is small (69 dialogues)** — JDD provides statistical significance; BSD adds domain diversity

### Run Commands

```bash
# Dialogue tasks — Qwen3-Omni (text-only)
python -m lmms_eval \
  --model qwen3_omni \
  --model_args pretrained=Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --tasks jp_topic_classify,jp_dialogue_response,jp_dialogue_summary \
  --batch_size 1 \
  --output_path results/qwen3_omni_phase3

# Dialogue tasks — GPT-4o
python -m lmms_eval \
  --model openai \
  --model_args model=gpt-4o \
  --tasks jp_topic_classify,jp_dialogue_response,jp_dialogue_summary \
  --batch_size 1 \
  --output_path results/gpt4o_phase3
```

---

## Phase 4: Audio-Conditioned LLM Tasks

**Goal**: Quantify the accuracy degradation when LLM tasks receive spoken input instead of text. This is the key metric for omni-model evaluation — it answers "is the audio encoder or the LLM the bottleneck?"

**Effort estimate**: ~1-2 weeks

### Tasks

| Task | Approach | Metric | Comparison |
|------|----------|--------|------------|
| VoiceBench | Already exists in framework — run as-is | Accuracy | Published scores (Qwen3-Omni: 88.8, GPT-4o: 86.8) |
| Spoken JMMLU (subset) | TTS-synthesize ~500 JMMLU questions, evaluate with audio input | Accuracy | Phase 1 text JMMLU scores |
| Audio dialogue tasks | Run Phase 3 tasks with CCR/Oshimaya audio input | Same as Phase 3 | Phase 3 text-only scores |

### Spoken JMMLU Implementation

1. **Select 500 JMMLU questions** — stratified sample across subject categories
2. **TTS synthesis** — use a high-quality Japanese TTS:
   - Option A: Qwen3-Omni's own TTS (to test self-consistency)
   - Option B: Google Cloud TTS (ja-JP, WaveNet voices) for neutral evaluation
   - Option C: VOICEVOX (open-source, high-quality Japanese TTS)
3. **Create audio dataset** — JSONL manifest with `{audio_filepath, question_text, choices, answer}`
4. **Run evaluation** — same MCQ accuracy metric as Phase 1

### Audio Dialogue Tasks

For CCR and Oshimaya data that already has audio:
- Reuse Phase 3 task configs but switch `doc_to_visual` to load audio
- The model receives audio of the conversation instead of text transcript
- Compare scores against Phase 3 text-only results

### File Structure

```
lmms_eval/tasks/
├── spoken_jmmlu/
│   ├── spoken_jmmlu.yaml
│   └── utils.py                     # Audio loading + MCQ extraction
├── jp_dialogue_summary_audio/
│   ├── jp_dialogue_summary_ccr_audio.yaml
│   └── utils.py
└── jp_info_extract_audio/
    ├── jp_info_extract_ccr_audio.yaml
    └── utils.py
```

### Run Commands

```bash
# VoiceBench — already exists
python -m lmms_eval \
  --model qwen3_omni \
  --model_args pretrained=Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --tasks voicebench \
  --batch_size 1 \
  --output_path results/qwen3_omni_voicebench

# Spoken JMMLU
python -m lmms_eval \
  --model qwen3_omni \
  --model_args pretrained=Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --tasks spoken_jmmlu \
  --batch_size 1 \
  --output_path results/qwen3_omni_spoken_jmmlu
```

### Key Analysis

The primary deliverable from Phase 4 is the **text-vs-audio accuracy gap table**:

```
| Benchmark      | Text Input | Audio Input | Gap    |
|----------------|-----------|-------------|--------|
| JMMLU          | XX.X%     | YY.Y%       | -Z.Z%  |
| Topic Classify | XX.X%     | YY.Y%       | -Z.Z%  |
| Summarization  | X.X/10    | Y.Y/10      | -Z.Z   |
```

If the gap is large (>10%), the audio encoder is the bottleneck and fine-tuning should focus on speech understanding. If the gap is small (<5%), the LLM backbone is the limiting factor.

---

## Results Tracking

### Directory Structure

```
results/
├── qwen3_omni_phase1/           # Text LLM benchmarks
├── qwen3_omni_phase2/           # ASR benchmarks
├── qwen3_omni_phase3/           # Dialogue tasks (text)
├── qwen3_omni_phase4/           # Audio-conditioned tasks
├── gpt4o_phase1/                # GPT-4o comparison
├── gpt4o_phase2/
├── gpt4o_phase3/
└── baseline_summary.md          # Consolidated baseline table
```

### Baseline Summary Table

```
| Benchmark            | Qwen3-Omni (text) | Qwen3-Omni (audio) | GPT-4o | Published SOTA    |
|----------------------|-------------------|--------------------:|-------:|------------------:|
| JMMLU (acc)          | 0.7850            | —                   |        | 0.7899 (Qwen2.5-72B) |
| JCommonsenseQA       | 0.9258            | —                   |        | 0.9696 (Qwen2.5-72B) |
| JNLI                 | 0.7634            | —                   |        |                   |
| MGSM-JA              | 0.8480            | —                   |        | 0.8360 (Qwen2.5-72B) |
| CCR-JP Agent CER     | —                 | 0.0692              |        | Internal          |
| CCR-JP Customer CER  | —                 | 0.1840              |        | Internal          |
| CCR-JP Mixed CER     | —                 | 0.1148              |        | Internal          |
| Dlight JP CER        | —                 | 0.1023              |        | Internal          |
| Oshimaya 50hr CER    | —                 | 0.2048              |        | Internal          |
| Oshimaya 80hr CER    | —                 | 0.1807              |        | Internal          |
| Rutilea JP CER       | —                 | 0.1064              |        | Internal          |
| CommonVoice JA CER   | —                 |                     |        | 7.85 (Reazon)     |
| ReazonSpeech CER     | —                 |                     |        | 9.09 (Reazon)     |
| Topic Classify       |                   |                     |        |                   |
| Dialogue Response    |                   |                     |        |                   |
| Summarization        |                   |                     |        |                   |
| VoiceBench           | —                 |                     |        | 88.8 (Qwen3)      |
```

---

## Dependencies & Setup

### Python Packages (beyond base lmms-eval)

```bash
# Phase 1 — text benchmarks (no extra deps beyond base)
pip install -e ".[audio]"

# Phase 2 — ASR (already installed for ccr_jp)
# fugashi, pykakasi, editdistance — already in requirements

# Phase 3 — LLM-as-judge
# Uses OpenAI API for GPT-4o judge — needs OPENAI_API_KEY env var

# Phase 4 — TTS synthesis (for spoken JMMLU)
pip install voicevox-core  # or google-cloud-texttospeech
```

### HuggingFace Datasets (auto-downloaded on first run)

| Dataset | HF Path | Approx Size |
|---------|---------|-------------|
| JMMLU | `nlp-waseda/JMMLU` | ~50MB |
| JCommonsenseQA | `leemeng/jcommonsenseqa` | ~5MB |
| JGLUE (JNLI) | `shunk031/JGLUE` | ~20MB |
| MGSM | `juletxara/mgsm` | ~2MB |
| CommonVoice JA | `mozilla-foundation/common_voice_17_0` | ~5GB (ja subset) |
| ReazonSpeech | `reazon-research/reazonspeech` | ~10GB (test split) |

### GPU Requirements

| Model | VRAM | GPUs | Notes |
|-------|------|------|-------|
| Qwen3-Omni-30B-A3B | ~60GB | 2-4x H100 | MoE, 3B active params but full model is 30B |
| GPT-4o | 0 (API) | — | Requires `OPENAI_API_KEY` |

---

## Risk & Gaps

| Risk | Mitigation |
|------|------------|
| JMMLU HF dataset format may differ from MMLU | Inspect dataset schema before writing configs; adapt `doc_to_text` accordingly |
| Qwen3-Omni text-only mode may behave differently than backbone Qwen3-30B | Document any discrepancies; consider adding Qwen3-30B as a third comparison model |
| LLM-as-judge (GPT-4o) introduces scoring variance | Run judge 2-3 times per sample, take mean; report inter-rater agreement |
| CommonVoice JA test set quality varies (crowd-sourced) | Filter by `up_votes >= 2` if supported in dataset |
| No public keigo correctness benchmark exists | Out of scope for this plan; note as future work |
| No Japanese function-calling benchmark exists | Out of scope for this plan; note as future work |

## Future Work (Beyond This Plan)

- **Keigo correctness evaluation** — build in-house using annotated customer-agent transcripts
- **Japanese function calling** — adapt BFCL/tau-bench with Japanese prompts
- **Streaming/latency evaluation** — measure real-time factor (RTF) for telephony deployment
- **Fine-tuned model comparison** — re-run all phases after fine-tuning to measure improvement
- **Additional models** — Ultravox, Whisper + LLM cascade, ReazonSpeech-k2-v2 + LLM cascade
