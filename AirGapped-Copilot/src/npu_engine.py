"""
AirGapped Copilot — NPU Engine
================================
ONNX Runtime inference engine targeting the Qualcomm Hexagon NPU
via QNNExecutionProvider with QnnHtp.dll backend.

Provides:
  - NPUEngine:      Base session manager with QNN EP / CPU fallback
  - EmbeddingEngine: Runs embedding model, returns normalized vectors
  - SLMEngine:       Autoregressive text generation on the NPU
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np

log = logging.getLogger("npu_engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# QNN EP provider options for Hexagon HTP (NPU)
QNN_PROVIDER_OPTIONS = {
    "backend_path": "QnnHtp.dll",
    "htp_performance_mode": "burst",
    "enable_htp_fp16_precision": "1",
    "device_id": "0",
}


# ═══════════════════════════════════════════════════════════════════════════
# Base NPU Engine
# ═══════════════════════════════════════════════════════════════════════════
class NPUEngine:
    """
    Base ONNX Runtime session manager.

    Tries QNNExecutionProvider (Hexagon HTP) first, then falls back to
    CPUExecutionProvider for non-Snapdragon dev machines.
    """

    def __init__(self, model_path: str, session_options: Optional[dict] = None):
        self.model_path = str(model_path)
        self.provider_name = "Unknown"
        self.is_npu_active = False
        self.session = None
        self._last_latency_ms = 0.0

        self._init_session(session_options)

    def _init_session(self, session_options: Optional[dict] = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if session_options:
            for k, v in session_options.items():
                setattr(opts, k, v)

        # --- Attempt: Register onnxruntime-qnn plugin ---
        try:
            import onnxruntime_qnn as qnn_ep
            ep_lib = qnn_ep.get_library_path()
            ort.register_execution_provider_library("QNNExecutionProvider", ep_lib)
            log.info(f"Registered QNN EP library from: {ep_lib}")
        except ImportError:
            log.info("onnxruntime-qnn not installed; QNN EP registration skipped.")
        except Exception as e:
            log.warning(f"QNN EP registration failed: {e}")

        providers_to_try = [
            ("QNNExecutionProvider", QNN_PROVIDER_OPTIONS, "Qualcomm Hexagon HTP (NPU)"),
            ("CPUExecutionProvider", {}, "CPU Fallback"),
        ]

        for provider, options, label in providers_to_try:
            try:
                if provider == "QNNExecutionProvider":
                    self.session = ort.InferenceSession(
                        self.model_path,
                        sess_options=opts,
                        providers=[provider],
                        provider_options=[options],
                    )
                else:
                    self.session = ort.InferenceSession(
                        self.model_path,
                        sess_options=opts,
                        providers=[provider],
                    )

                active = self.session.get_providers()
                self.provider_name = label
                self.is_npu_active = (
                    provider == "QNNExecutionProvider"
                    and "QNNExecutionProvider" in active
                )

                if self.is_npu_active:
                    log.info(f"✅  Inference on: {label}  (QnnHtp.dll)")
                else:
                    log.info(f"🔄  Inference on: {label}")
                return

            except Exception as e:
                log.warning(f"⚠️  {label} init failed: {e}")
                continue

        # All providers exhausted
        log.warning(
            "⚠️  No ONNX providers available. Engine running in MOCK mode."
        )
        self.provider_name = "Mock (No Runtime)"
        self.is_npu_active = False

    def run(self, input_dict: dict) -> list:
        """Run inference, returning outputs and recording latency."""
        if self.session is None:
            self._last_latency_ms = 0.0
            return []

        t0 = time.perf_counter()
        outputs = self.session.run(None, input_dict)
        t1 = time.perf_counter()
        self._last_latency_ms = (t1 - t0) * 1000.0
        return outputs

    @property
    def latency_ms(self) -> float:
        return self._last_latency_ms

    def get_input_names(self) -> List[str]:
        if self.session is None:
            return []
        return [inp.name for inp in self.session.get_inputs()]

    def get_output_names(self) -> List[str]:
        if self.session is None:
            return []
        return [out.name for out in self.session.get_outputs()]

    @property
    def status_text(self) -> str:
        if self.is_npu_active:
            return "Qualcomm Hexagon HTP — QnnHtp.dll"
        elif self.session is not None:
            return "CPU Fallback — onnxruntime"
        else:
            return "Mock Mode (No Runtime)"


# ═══════════════════════════════════════════════════════════════════════════
# Embedding Engine
# ═══════════════════════════════════════════════════════════════════════════
class EmbeddingEngine:
    """
    Generates 384-dimensional sentence embeddings using the
    all-MiniLM-L6-v2 model compiled for Qualcomm AI Hub / QNN.
    """

    EMBEDDING_DIM = 384
    MAX_SEQ_LENGTH = 128

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or MODELS_DIR / "embedding_model")
        candidates = [
            self.model_dir / "model.onnx",
            self.model_dir / "all-MiniLM-L6-v2.onnx",
            MODELS_DIR / "embedding" / "all-MiniLM-L6-v2.onnx",
            MODELS_DIR / "embedding" / "model.onnx",
        ]
        if (MODELS_DIR / "embedding").exists():
            candidates.extend(list((MODELS_DIR / "embedding").glob("*.onnx")))
        if self.model_dir.exists():
            candidates.extend(list(self.model_dir.glob("*.onnx")))

        self.model_path = None
        for c in candidates:
            if c.exists():
                self.model_path = c
                break
        if not self.model_path:
            self.model_path = candidates[0]

        self.engine: Optional[NPUEngine] = None
        self.tokenizer = None
        self._mock_mode = False

        self._init()

    def _init(self):
        # Load tokenizer safely (strictly local / offline for air-gapped execution)
        try:
            from transformers import AutoTokenizer
            try:
                # 1. Local path
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
            except Exception:
                try:
                    # 2. Local cache of pretrained
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
                    )
                except Exception:
                    self.tokenizer = None
            if self.tokenizer is not None:
                log.info("✅  Embedding tokenizer loaded (all-MiniLM-L6-v2)")
            else:
                log.info("ℹ️  Local embedding tokenizer cache not found; using fast offline character-level tokenizer.")
        except Exception as e:
            log.info(f"ℹ️  Embedding tokenizer using built-in fallback: {e}")

        # Load ONNX model
        if self.model_path and self.model_path.exists():
            try:
                self.engine = NPUEngine(str(self.model_path))
                log.info(f"✅  Embedding model loaded: {self.model_path}")
            except Exception as e:
                log.warning(f"⚠️  Embedding model load failed: {e}")
                self._mock_mode = True
        else:
            log.warning(
                f"⚠️  Embedding model not found at {self.model_path}\n"
                f"   Run: python scripts/download_ai_hub_models.py\n"
                f"   Using random mock embeddings."
            )
            self._mock_mode = True

    @property
    def active_provider(self) -> str:
        if self.engine:
            return self.engine.provider_name
        return "Mock Mode"

    @property
    def status_text(self) -> str:
        if self.engine:
            return self.engine.status_text
        return "Mock Mode (No Model Loaded)"

    def embed(self, text: str) -> np.ndarray:
        """Generate a normalized 384-dim embedding for a text string."""
        if self._mock_mode or self.engine is None:
            return self._mock_embed(text)

        # Tokenize
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.MAX_SEQ_LENGTH,
                return_tensors="np",
            )
            input_dict = {
                k: v.astype(np.int64)
                for k, v in tokens.items()
                if k in self.engine.get_input_names()
            }
        else:
            input_dict = self._simple_tokenize(text)

        outputs = self.engine.run(input_dict)
        if not outputs:
            return self._mock_embed(text)

        # Mean pooling + L2 normalisation
        embedding = outputs[0]
        if embedding.ndim == 3:
            embedding = embedding.mean(axis=1)
        embedding = embedding.flatten().astype(np.float32)

        if len(embedding) > self.EMBEDDING_DIM:
            embedding = embedding[: self.EMBEDDING_DIM]
        elif len(embedding) < self.EMBEDDING_DIM:
            embedding = np.pad(embedding, (0, self.EMBEDDING_DIM - len(embedding)))

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts, returning (N, 384) array."""
        return np.array([self.embed(t) for t in texts], dtype=np.float32)

    @property
    def latency_ms(self) -> float:
        return self.engine.latency_ms if self.engine else 0.0

    def _mock_embed(self, text: str) -> np.ndarray:
        """Deterministic mock embedding based on text hash."""
        rng = np.random.RandomState(hash(text) % (2**31))
        vec = rng.randn(self.EMBEDDING_DIM).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def _simple_tokenize(self, text: str) -> dict:
        """Fallback character-level tokenizer."""
        ids = [ord(c) % 30000 for c in text[: self.MAX_SEQ_LENGTH]]
        ids += [0] * (self.MAX_SEQ_LENGTH - len(ids))
        arr = np.array([ids], dtype=np.int64)
        mask = np.array(
            [[1 if i < len(text) else 0 for i in range(self.MAX_SEQ_LENGTH)]],
            dtype=np.int64,
        )
        return {"input_ids": arr, "attention_mask": mask}


# ═══════════════════════════════════════════════════════════════════════════
# SLM Engine  (Small Language Model — Autoregressive Generation)
# ═══════════════════════════════════════════════════════════════════════════
class SLMEngine:
    """
    Text generation engine using a Qualcomm AI Hub SLM
    (Llama-3.2-1B-Instruct) running on the Hexagon NPU.
    """

    MAX_CONTEXT = 512
    DEFAULT_MAX_TOKENS = 256

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or MODELS_DIR / "slm_model")
        candidates = [
            self.model_dir / "model.onnx",
            self.model_dir / "llama_v3_2_1b_instruct.onnx",
            MODELS_DIR / "slm" / "llama_v3_2_1b_instruct.onnx",
            MODELS_DIR / "slm" / "model.onnx",
        ]
        if (MODELS_DIR / "slm").exists():
            candidates.extend(list((MODELS_DIR / "slm").glob("*.onnx")))
        if self.model_dir.exists():
            candidates.extend(list(self.model_dir.glob("*.onnx")))

        self.model_path = None
        for c in candidates:
            if c.exists():
                self.model_path = c
                break
        if not self.model_path:
            self.model_path = candidates[0]

        self.engine: Optional[NPUEngine] = None
        self.tokenizer = None
        self._mock_mode = False
        self._total_latency_ms = 0.0
        self._tokens_generated = 0

        self._init()

    def _init(self):
        # Load tokenizer safely (check local cache first for air-gapped execution)
        try:
            from transformers import AutoTokenizer
            try:
                # 1. Check local model directory
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
            except Exception:
                try:
                    # 2. Check local huggingface cache
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "meta-llama/Llama-3.2-1B-Instruct",
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                except Exception:
                    # 3. Non-blocking attempt or local fallback
                    log.info("ℹ️  Local SLM tokenizer cache not found; will use built-in fallback tokenizer.")
                    self.tokenizer = None
            if self.tokenizer:
                log.info("✅  SLM tokenizer loaded (Llama-3.2-1B-Instruct)")
        except Exception as e:
            log.info(f"ℹ️  SLM tokenizer using built-in fallback: {e}")

        # Load ONNX model
        if self.model_path and self.model_path.exists():
            try:
                self.engine = NPUEngine(str(self.model_path))
                log.info(f"✅  SLM model loaded: {self.model_path}")
            except Exception as e:
                log.warning(f"⚠️  SLM model load failed: {e}")
                self._mock_mode = True
        else:
            log.warning(
                f"⚠️  SLM model not found at {self.model_path}\n"
                f"   Run: python scripts/download_ai_hub_models.py\n"
                f"   Using mock text generation."
            )
            self._mock_mode = True

    @property
    def active_provider(self) -> str:
        if self.engine:
            return self.engine.provider_name
        return "Mock Mode"

    def generate(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response for the given prompt."""
        if self._mock_mode or self.engine is None:
            return self._mock_generate(prompt, max_tokens)

        return self._onnx_generate(prompt, max_tokens, temperature)

    def _onnx_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Autoregressive generation loop on the ONNX model."""
        t_start = time.perf_counter()

        if self.tokenizer is not None:
            input_ids = self.tokenizer.encode(
                prompt, return_tensors="np", truncation=True,
                max_length=self.MAX_CONTEXT,
            )
        else:
            ids = [ord(c) % 30000 for c in prompt[-self.MAX_CONTEXT:]]
            input_ids = np.array([ids], dtype=np.int64)

        generated_ids = input_ids[0].tolist()
        input_names = self.engine.get_input_names()

        for step in range(max_tokens):
            cur_ids = np.array([generated_ids[-self.MAX_CONTEXT:]], dtype=np.int64)
            attn_mask = np.ones_like(cur_ids, dtype=np.int64)

            feed: Dict[str, np.ndarray] = {}
            if "input_ids" in input_names:
                feed["input_ids"] = cur_ids
            if "attention_mask" in input_names:
                feed["attention_mask"] = attn_mask

            try:
                outputs = self.engine.run(feed)
            except Exception as e:
                log.error(f"SLM inference error at step {step}: {e}")
                break

            if not outputs:
                break

            logits = outputs[0]
            if logits.ndim == 3:
                next_logits = logits[0, -1, :]
            elif logits.ndim == 2:
                next_logits = logits[0, :]
            else:
                next_logits = logits.flatten()

            if temperature <= 0.01:
                next_id = int(np.argmax(next_logits))
            else:
                scaled = next_logits / temperature
                exp = np.exp(scaled - np.max(scaled))
                probs = exp / exp.sum()
                next_id = int(np.random.choice(len(probs), p=probs))

            generated_ids.append(next_id)

            if self.tokenizer and next_id == self.tokenizer.eos_token_id:
                break

        t_end = time.perf_counter()
        self._total_latency_ms = (t_end - t_start) * 1000.0

        new_ids = generated_ids[len(input_ids[0]):]
        self._tokens_generated = len(new_ids)

        if self.tokenizer is not None:
            return self.tokenizer.decode(new_ids, skip_special_tokens=True)
        else:
            return "".join(chr(min(i, 127)) for i in new_ids)

    def _mock_generate(self, prompt: str, max_tokens: int) -> str:
        """Deterministic mock generation for dev/demo without a real model."""
        t_start = time.perf_counter()

        lines = prompt.strip().split("\n")
        context_lines = [l for l in lines if l.strip().startswith("[")]

        response_parts = ["Based on the provided documents, "]

        if context_lines:
            response_parts.append(
                f"I found {len(context_lines)} relevant passage(s). "
            )
            response_parts.append(
                "The documents indicate that the relevant information "
                "can be found in the cited sections. "
            )
            for i, ctx in enumerate(context_lines[:3], 1):
                src = ctx[:80].strip("[] ")
                response_parts.append(f"[Citation {i}: {src}...] ")
        else:
            response_parts.append(
                "No specific document context was provided for this query. "
                "Please ingest relevant PDF documents first using the "
                "Document Ingestion panel, then retry your question."
            )

        response_parts.append(
            "\n\nNote: This is a mock response generated without a real "
            "SLM model. Download the Qualcomm AI Hub models to enable "
            "genuine NPU-accelerated inference."
        )

        time.sleep(0.3)
        t_end = time.perf_counter()

        result = "".join(response_parts)
        self._total_latency_ms = (t_end - t_start) * 1000.0
        self._tokens_generated = len(result.split())

        return result

    @property
    def latency_ms(self) -> float:
        return self._total_latency_ms

    @property
    def tokens_per_sec(self) -> float:
        if self._total_latency_ms <= 0:
            return 0.0
        return (self._tokens_generated / self._total_latency_ms) * 1000.0

    @property
    def tokens_generated(self) -> int:
        return self._tokens_generated

    @property
    def status_text(self) -> str:
        if self.engine:
            return self.engine.status_text
        return "Mock Mode (No Model Loaded)"
