"""Ultravox audio-language model adapter via vLLM.

Ultravox is natively supported by vLLM's offline LLM class, so this adapter
is a thin wrapper that handles audio input formatting and delegates inference
to vLLM's ``LLM.chat()`` method.

Usage::

    lmms-eval --model ultravox \\
        --model_args model=fixie-ai/ultravox-v0_5-llama-3_2-1b,tensor_parallel_size=1 \\
        --tasks librispeech_test_clean \\
        --batch_size 1

Requires: ``pip install lmms_eval[ultravox]``
"""

import base64
import io
import os
from typing import List, Optional, Tuple, Union

import numpy as np
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import

LLM, _has_vllm = optional_import("vllm", "LLM")
SamplingParams, _ = optional_import("vllm", "SamplingParams")


def _audio_to_data_url(audio_array: np.ndarray, sampling_rate: int) -> str:
    """Encode a numpy audio array as a base64 WAV data URL for vLLM."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio_array, sampling_rate, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


def _decode_audio(audio_obj) -> dict:
    """Normalize various audio representations to {"array": ndarray, "sampling_rate": int}."""
    if isinstance(audio_obj, dict) and "array" in audio_obj and "sampling_rate" in audio_obj:
        return audio_obj

    type_name = type(audio_obj).__name__
    if type_name == "AudioDecoder":
        if hasattr(audio_obj, "get_all_samples"):
            decoded = audio_obj.get_all_samples()
            array = decoded.data if hasattr(decoded, "data") else (decoded.samples if hasattr(decoded, "samples") else decoded)
            if hasattr(array, "cpu"):
                array = array.cpu().numpy()
            sr = getattr(decoded, "sample_rate", getattr(decoded, "sampling_rate", 16000))
            return {"array": np.asarray(array, dtype=np.float32).flatten(), "sampling_rate": sr}

    raise ValueError(f"Unsupported audio type: {type(audio_obj)}")


@register_model("ultravox")
class Ultravox(lmms):
    """Ultravox audio-language model evaluated via vLLM offline inference.

    Ultravox models (fixie-ai/ultravox-*) accept interleaved audio and text
    in OpenAI-style chat messages.  vLLM handles the model-specific audio
    preprocessing internally.

    Args:
        model: HuggingFace model ID or local path.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory for model weights.
        batch_size: Requests per batch.
        trust_remote_code: Allow remote code execution during model loading.
        chat_template: Optional Jinja2 chat template override.
        **kwargs: Extra arguments forwarded to ``vllm.LLM()``.
    """

    def __init__(
        self,
        model: str = "fixie-ai/ultravox-v0_5-llama-3_2-1b",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        batch_size: int = 1,
        trust_remote_code: bool = True,
        chat_template: Optional[str] = None,
        disable_log_stats: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if not _has_vllm:
            raise ImportError("vLLM is required for Ultravox. Install with: pip install lmms_eval[ultravox]")

        self.model_id = model
        self.batch_size_per_gpu = int(batch_size)
        self.disable_log_stats = disable_log_stats

        # Load chat template from file if it looks like a path
        self.chat_template = None
        if chat_template is not None:
            if os.path.sep in chat_template or chat_template.endswith((".jinja", ".jinja2", ".j2")):
                if not os.path.isfile(chat_template):
                    raise FileNotFoundError(f"Chat template file not found: {chat_template}")
                with open(chat_template, "r") as f:
                    self.chat_template = f.read()
            else:
                self.chat_template = chat_template

        # Parse JSON-like string kwargs (same pattern as models/simple/vllm.py)
        import json

        for key, value in kwargs.items():
            if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
                try:
                    kwargs[key] = json.loads(value)
                except json.JSONDecodeError:
                    eval_logger.warning(f"Failed to parse JSON-like string for argument '{key}': {value}")

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        self.client = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            disable_log_stats=disable_log_stats,
            **kwargs,
        )

        self._rank = 0
        self._world_size = 1

        eval_logger.info(f"Ultravox loaded: model={model} tp={tensor_parallel_size} gpu_mem={gpu_memory_utilization}")

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return "cuda"

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Ultravox")

    def flatten(self, input):
        new_list = []
        for i in input:
            if isinstance(i, (list, tuple)):
                new_list.extend(i)
            else:
                new_list.append(i)
        return new_list

    def _build_messages(self, context: str, audios: list) -> list:
        """Build OpenAI-style messages with audio data URLs for vLLM.

        vLLM's Ultravox support expects audio as ``audio_url`` content parts
        with base64 data URLs.
        """
        content = []
        for audio_obj in audios:
            audio_dict = _decode_audio(audio_obj)
            data_url = _audio_to_data_url(audio_dict["array"], audio_dict["sampling_rate"])
            content.append({
                "type": "audio_url",
                "audio_url": {"url": data_url},
            })
        content.append({"type": "text", "text": context})
        return [{"role": "user", "content": content}]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return 0, x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]

            gen_kwargs = dict(all_gen_kwargs[0])
            gen_kwargs.setdefault("max_new_tokens", 4096)
            gen_kwargs.setdefault("temperature", 0)
            gen_kwargs.setdefault("top_p", 0.95)

            until = None
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]

            sampling_params = SamplingParams(
                max_tokens=gen_kwargs["max_new_tokens"],
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
            )

            # Build batched messages
            batched_messages = []
            for i, ctx in enumerate(contexts):
                audios = doc_to_visual[i](self.task_dict[task][split][doc_id[i]])
                if audios is None:
                    audios = []
                messages = self._build_messages(ctx, audios)
                batched_messages.append(messages)

            # Run vLLM inference
            chat_kwargs = {"sampling_params": sampling_params, "messages": batched_messages}
            if self.chat_template is not None:
                chat_kwargs["chat_template"] = self.chat_template

            try:
                responses = self.client.chat(**chat_kwargs)
                answers = [r.outputs[0].text for r in responses]
            except Exception as e:
                eval_logger.error(f"Error during Ultravox generation: {e}")
                answers = [""] * len(contexts)

            # Apply until tokens
            if until:
                for i, ans in enumerate(answers):
                    for term in until:
                        if term:
                            ans = ans.split(term)[0]
                    answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Ultravox")
