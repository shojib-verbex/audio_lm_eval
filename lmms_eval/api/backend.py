"""Model backend abstraction layer.

Provides a unified interface for the three inference backends:
- VLLMBackend: vLLM offline LLM class (fastest for supported models)
- TransformersBackend: HuggingFace AutoModel loading (broadest compatibility)
- APIBackend: OpenAI-compatible API endpoints (for pre-deployed models)

The ``resolve_backend`` helper implements the ``backend: auto`` config option,
which tries vLLM → Transformers → API in priority order.
"""

from __future__ import annotations

import abc
import gc
from typing import Any, Dict, List, Optional

from loguru import logger as eval_logger

from lmms_eval.imports import is_package_available, optional_import


class ModelBackend(abc.ABC):
    """Abstract base class for model inference backends.

    Each backend wraps a different inference engine and exposes a common
    ``generate`` interface.  Backends are responsible for model loading,
    GPU allocation, and generation.
    """

    backend_type: str = ""

    @abc.abstractmethod
    def load(self, model_id: str, **kwargs: Any) -> None:
        """Load the model identified by *model_id* with backend-specific kwargs."""

    @abc.abstractmethod
    def generate(
        self,
        messages: List[List[Dict[str, Any]]],
        **gen_kwargs: Any,
    ) -> List[str]:
        """Generate responses for a batch of message sequences.

        Parameters
        ----------
        messages:
            A list of conversations, where each conversation is a list of
            OpenAI-style message dicts (``{"role": ..., "content": ...}``).
        gen_kwargs:
            Generation parameters (``max_new_tokens``, ``temperature``, etc.).

        Returns
        -------
        List of generated response strings, one per input conversation.
        """

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if the required dependencies for this backend are installed."""

    def unload(self) -> None:
        """Release model resources and free GPU memory."""
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def is_loaded(self) -> bool:
        """Return True if a model is currently loaded."""
        return False


class VLLMBackend(ModelBackend):
    """Backend using vLLM's offline ``LLM`` class for in-process inference.

    Supports automatic batching, PagedAttention memory management, and tensor
    parallelism across GPUs.  This is the fastest path for models that vLLM
    supports (Ultravox, Qwen2-Audio, Llama, Whisper, etc.).
    """

    backend_type = "vllm"

    def __init__(self) -> None:
        self.client = None
        self._model_id: Optional[str] = None

    def is_available(self) -> bool:
        return is_package_available("vllm")

    def load(
        self,
        model_id: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        LLM, has_vllm = optional_import("vllm", "LLM")
        if not has_vllm:
            raise ImportError(
                "vLLM is required for VLLMBackend. Install with: pip install lmms_eval[vllm]"
            )

        self.client = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self._model_id = model_id
        eval_logger.info(
            f"VLLMBackend loaded model={model_id} "
            f"tp={tensor_parallel_size} gpu_mem={gpu_memory_utilization}"
        )

    def generate(
        self,
        messages: List[List[Dict[str, Any]]],
        **gen_kwargs: Any,
    ) -> List[str]:
        if self.client is None:
            raise RuntimeError("No model loaded. Call load() first.")

        SamplingParams, _ = optional_import("vllm", "SamplingParams")
        params = SamplingParams(
            max_tokens=gen_kwargs.get("max_new_tokens", 4096),
            temperature=gen_kwargs.get("temperature", 0),
            top_p=gen_kwargs.get("top_p", 0.95),
        )
        responses = self.client.chat(
            messages=messages,
            sampling_params=params,
        )
        return [r.outputs[0].text for r in responses]

    def unload(self) -> None:
        if self.client is not None:
            del self.client
            self.client = None
            self._model_id = None
        super().unload()

    @property
    def is_loaded(self) -> bool:
        return self.client is not None


class TransformersBackend(ModelBackend):
    """Backend using HuggingFace Transformers for local model loading.

    Supports ``device_map``, ``torch_dtype``, and quantization configs
    (bitsandbytes, GPTQ, AWQ).  Each model adapter subclass typically
    overrides ``preprocess()`` and ``generate()`` for model-specific input
    formatting.
    """

    backend_type = "transformers"

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self._model_id: Optional[str] = None

    def is_available(self) -> bool:
        return is_package_available("transformers")

    def load(
        self,
        model_id: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        ).eval()

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            self.processor = None

        self._model_id = model_id
        eval_logger.info(
            f"TransformersBackend loaded model={model_id} device_map={device_map}"
        )

    def generate(
        self,
        messages: List[List[Dict[str, Any]]],
        **gen_kwargs: Any,
    ) -> List[str]:
        if self.model is None:
            raise RuntimeError("No model loaded. Call load() first.")
        raise NotImplementedError(
            "TransformersBackend.generate() requires model-specific preprocessing. "
            "Use a registered model adapter (e.g., whisper, qwen2_audio) instead."
        )

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._model_id = None
        super().unload()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


class APIBackend(ModelBackend):
    """Backend wrapping an OpenAI-compatible API endpoint.

    Useful for models already deployed as services (shared vLLM server,
    commercial APIs like GPT-4o, Gemini) or as a fallback when models have
    irreconcilable dependency conflicts.
    """

    backend_type = "api"

    def __init__(self) -> None:
        self.client = None
        self._model_id: Optional[str] = None
        self._base_url: Optional[str] = None

    def is_available(self) -> bool:
        return is_package_available("openai")

    def load(
        self,
        model_id: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        import os

        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
            timeout=timeout,
            max_retries=max_retries,
        )
        self._model_id = model_id
        self._base_url = base_url
        eval_logger.info(
            f"APIBackend loaded model={model_id} base_url={base_url or 'default'}"
        )

    def generate(
        self,
        messages: List[List[Dict[str, Any]]],
        **gen_kwargs: Any,
    ) -> List[str]:
        if self.client is None:
            raise RuntimeError("No model loaded. Call load() first.")

        results = []
        for conversation in messages:
            response = self.client.chat.completions.create(
                model=self._model_id,
                messages=conversation,
                max_tokens=gen_kwargs.get("max_new_tokens", 4096),
                temperature=gen_kwargs.get("temperature", 0),
            )
            results.append(response.choices[0].message.content)
        return results

    def unload(self) -> None:
        self.client = None
        self._model_id = None
        self._base_url = None

    @property
    def is_loaded(self) -> bool:
        return self.client is not None


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

_BACKEND_CLASSES = {
    "vllm": VLLMBackend,
    "transformers": TransformersBackend,
    "api": APIBackend,
}

_AUTO_PRIORITY = ["vllm", "transformers", "api"]


def resolve_backend(
    backend: str = "auto",
    model_id: Optional[str] = None,
) -> ModelBackend:
    """Resolve a backend name to a concrete ``ModelBackend`` instance.

    Parameters
    ----------
    backend:
        One of ``"vllm"``, ``"transformers"``, ``"api"``, or ``"auto"``.
        ``"auto"`` tries each backend in priority order (vLLM → Transformers → API)
        and returns the first one whose dependencies are installed.
    model_id:
        Optional model identifier (used for logging only).

    Returns
    -------
    An unloaded ``ModelBackend`` instance.

    Raises
    ------
    ValueError
        If *backend* is not recognized.
    RuntimeError
        If ``"auto"`` is used and no backend has its dependencies installed.
    """
    if backend == "auto":
        for name in _AUTO_PRIORITY:
            instance = _BACKEND_CLASSES[name]()
            if instance.is_available():
                eval_logger.info(
                    f"Auto-resolved backend={name} for model={model_id or '?'}"
                )
                return instance
        raise RuntimeError(
            "No inference backend available. Install at least one of: "
            "pip install lmms_eval[vllm]  OR  pip install transformers  OR  pip install openai"
        )

    if backend not in _BACKEND_CLASSES:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: {', '.join(_BACKEND_CLASSES)} or 'auto'"
        )

    instance = _BACKEND_CLASSES[backend]()
    if not instance.is_available():
        extras_map = {"vllm": "vllm", "transformers": None, "api": None}
        extras = extras_map.get(backend)
        install_hint = f"pip install lmms_eval[{extras}]" if extras else f"pip install {backend}"
        raise RuntimeError(
            f"Backend '{backend}' is not available. Install dependencies with: {install_hint}"
        )

    return instance
