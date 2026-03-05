#!/bin/bash
# =============================================================================
# Qwen3-Omni Evaluation Example
# =============================================================================
#
# Qwen3-Omni is a multimodal model supporting audio, image, and video inputs.
# Model: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
#
# Installation:
#   pip install -e ".[qwen3-omni]"
#   # This installs: librosa, soundfile, qwen-omni-utils, moviepy
#
# For flash attention support (recommended):
#   pip install flash-attn --no-build-isolation
#
# Available model_args:
#   pretrained          - HuggingFace model ID or local path
#                         (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)
#   device_map          - Device placement strategy (default: auto)
#   attn_implementation - Attention backend: flash_attention_2, sdpa, eager
#                         (default: flash_attention_2)
#   max_num_frames      - Max video frames to extract (default: 128)
#   batch_size          - Batch size for inference (default: 1)
#
# Example audio tasks:
#   librispeech_test_clean    - Speech recognition (ASR)
#   air_bench_chat_speech     - Speech understanding
#   voicebench                - Voice chat benchmark
#   alpaca_audio              - Audio instruction following
#   common_voice_15_en        - CommonVoice English ASR
#   vocalsound_test           - Sound classification
#
# =============================================================================

TASK=${1:-"librispeech_test_clean"}
MODEL_PATH=${2:-"Qwen/Qwen3-Omni-30B-A3B-Instruct"}

echo "Task: $TASK"
TASK_SUFFIX="${TASK//,/_}"

python -m lmms_eval \
    --model qwen3_omni \
    --model_args pretrained=$MODEL_PATH,attn_implementation=flash_attention_2 \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/
