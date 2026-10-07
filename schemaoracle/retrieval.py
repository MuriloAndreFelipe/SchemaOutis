"""
Biencoder retrieval: the one ordering every arm reads.

Ported from SchemaContext's embedder.py, retrieval path only. Similarity follows
Magneto's `compute_cosine_similarity`, with explicit L2 normalisation.

Magneto's bipartite (Hungarian) pass is not applied here. It reassigns rank 1
across source columns, which is reranking, and would put a reranking effect
inside the ceiling that retrieval is meant to set.

`load_model` also accepts a fine-tuned state dict, which is how Magneto ships
its GDC retrievers (`mpnet-gdc-*.pth`): the base model is inferred from the
filename and the weights are overlaid. Each checkpoint was trained for one
serialisation, so `encoding_mode` has to match it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .serializers import build_column_profiles
from .vectors import VectorStore, as_store, fingerprint

MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Base checkpoints a fine-tuned `.pth` is overlaid on, keyed by the token in
# Magneto's checkpoint filenames (as `magneto.embedding_matcher.DEFAULT_MODELS`).
DEFAULT_MODELS = {
    "mpnet": MPNET_MODEL,
    "roberta": "sentence-transformers/all-roberta-large-v1",
    "e5": "intfloat/e5-base",
    "arctic": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "magneto": "vida-nyu/magneto-schema-retriever-gdc"
}

# Extensions treated as a fine-tuned state dict rather than a model id.
_WEIGHT_SUFFIXES = (".pth", ".pt", ".bin", ".ckpt")

_MODEL_CACHE: dict[str, object] = {}


def _remap_state_dict_keys(state_dict, target_keys):
    """
    Reconcile the `auto_model` -> `model` rename that sentence-transformers v5
    made, returning (state dict, note). Magneto's GDC checkpoints exist under
    both namings. Only keys are renamed, and only when nothing else differs.
    """
    keys = set(state_dict)
    target = set(target_keys)
    if keys == target:
        return state_dict, "keys matched as-is"

    for src, dst in (("auto_model.", "model."), ("model.", "auto_model.")):
        remapped = {k.replace(src, dst, 1): v for k, v in state_dict.items()}
        if set(remapped) == target:
            return remapped, f"remapped {src!r} -> {dst!r} on every key"

    return state_dict, "no remapping matched"


def _load_finetuned(path: Path, device: str):
    """
    The base SentenceTransformer with a fine-tuned state dict overlaid, as in
    `magneto.embedding_matcher`. The base is inferred from the filename
    (mpnet if unrecognised). Loaded strictly: a partial load raises.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    base_key = next((k for k in DEFAULT_MODELS if k in path.name.lower()), "mpnet")
    model = SentenceTransformer(DEFAULT_MODELS[base_key], device=device)

    state_dict = torch.load(str(path), map_location=device, weights_only=True)
    state_dict, note = _remap_state_dict_keys(state_dict, model.state_dict().keys())

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        expected = set(model.state_dict())
        got = set(state_dict)
        raise RuntimeError(
            f"{path.name} does not fit base model {DEFAULT_MODELS[base_key]!r}.\n"
            f"  key reconciliation: {note}\n"
            f"  missing from checkpoint : {len(expected - got)} "
            f"{sorted(expected - got)[:3]}\n"
            f"  unexpected in checkpoint: {len(got - expected)} "
            f"{sorted(got - expected)[:3]}\n"
            "If this is a Magneto GDC checkpoint, the '-fixed' variant carries "
            "the sentence-transformers v5 key names."
        ) from exc

    model.eval().to(device)
    return model


def looks_like_local_path(model_name: str) -> bool:
    """
    Whether `model_name` is a path rather than a HuggingFace id. A slash alone
    does not decide it, since HuggingFace ids contain one.
    """
    path = Path(model_name)
    return (
        path.suffix.lower() in _WEIGHT_SUFFIXES
        or bool(path.drive)
        or "\\" in model_name
        or model_name.startswith((".", "~"))
        or path.exists()
    )


def load_model(model_name: str = MPNET_MODEL):
    """
    Load and cache a SentenceTransformer: a HuggingFace id, a saved model
    directory, or a fine-tuned state dict. Paths are resolved here because
    SentenceTransformer rejects them as invalid repo ids.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(model_name)

    if looks_like_local_path(model_name):
        if path.is_file():
            if path.suffix.lower() not in _WEIGHT_SUFFIXES:
                raise ValueError(
                    f"{path} is a file but not a recognised checkpoint. "
                    f"Expected one of {_WEIGHT_SUFFIXES}, a directory holding a "
                    "saved SentenceTransformer, or a HuggingFace identifier."
                )
            model = _load_finetuned(path, device)
        elif path.is_dir():
            model = SentenceTransformer(str(path), device=device)
        else:
            raise FileNotFoundError(
                f"embedding model path does not exist: {path}\n"
                "  Pass an existing checkpoint or directory, or a HuggingFace "
                "identifier for a stock model."
            )
    else:
        try:
            model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        except OSError as exc:
            if "Repo id" in str(exc):
                raise OSError(
                    f"{model_name!r} was sent to HuggingFace as a repo id and "
                    "rejected.\n"
                    "  If it is a local file, restart the kernel so the current "
                    "`load_model` is used.\n"
                    f"  Original error: {exc}"
                ) from exc
            raise

    _MODEL_CACHE[model_name] = model
    return model


def free_embedder(model_name: str | None = None) -> list[str]:
    """
    Drop cached SentenceTransformers and free their GPU memory, so a local LLM
    can use the card. Returns the names evicted; None evicts all. torch is only
    touched if already imported.
    """
    import gc
    import sys

    evicted = (
        list(_MODEL_CACHE)
        if model_name is None
        else [model_name] if model_name in _MODEL_CACHE else []
    )
    for name in evicted:
        _MODEL_CACHE.pop(name, None)
    gc.collect()

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return evicted


# Approximate characters per token on these profiles; only sizes a batch.
_CHARS_PER_TOKEN = 3.5

# Tokens one forward pass may hold. A batch is padded to its longest member, so
# memory scales with batch_size x longest; budgeting that product keeps peak
# memory flat as profiles lengthen.
DEFAULT_TOKEN_BUDGET = 16_384


def plan_batch_size(texts, max_seq_length=None, batch_size: int = 64,
                    token_budget: int = DEFAULT_TOKEN_BUDGET) -> int:
    """Sequences per forward pass, sized from the longest text; at most `batch_size`."""
    if not texts:
        return batch_size
    longest = max(len(t) for t in texts)
    tokens = int(longest / _CHARS_PER_TOKEN) + 8
    if max_seq_length:
        tokens = min(tokens, int(max_seq_length))
    tokens = max(tokens, 16)
    return max(1, min(int(batch_size), token_budget // tokens))


def embed_profiles(
    profiles: dict[str, str],
    model,
    batch_size: int = 64,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> dict[str, np.ndarray]:
    """
    Encode serialised profiles: column name -> raw float32 vector (normalised
    later, as in Magneto). The batch is halved on a CUDA out-of-memory error.
    """
    names = list(profiles.keys())
    texts = [profiles[n] for n in names]
    planned = plan_batch_size(texts, getattr(model, "max_seq_length", None),
                              batch_size, token_budget)

    import torch

    size = planned
    while True:
        try:
            embeddings = model.encode(
                texts,
                batch_size=size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            return dict(zip(names, embeddings))
        except torch.cuda.OutOfMemoryError:
            if size <= 1:
                raise
            size = max(1, size // 2)
            torch.cuda.empty_cache()


def compute_cosine_similarity(
    embeddings_input: np.ndarray,
    embeddings_target: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Top-k cosine similarities, as Magneto computes them. Returns
    (top_k_scores, top_k_indices, full_similarity_matrix).
    """
    input_norm = np.linalg.norm(embeddings_input, axis=1, keepdims=True)
    target_norm = np.linalg.norm(embeddings_target, axis=1, keepdims=True)

    # An all-null column can embed to a zero vector; Magneto's unguarded
    # division gives NaN, which sorts unpredictably.
    input_norm = np.where(input_norm == 0, 1.0, input_norm)
    target_norm = np.where(target_norm == 0, 1.0, target_norm)

    similarities = (embeddings_input @ embeddings_target.T) / (input_norm * target_norm.T)

    k = min(top_k, similarities.shape[1])
    sorted_indices = np.argsort(-similarities, axis=1, kind="stable")
    top_k_indices = sorted_indices[:, :k]
    top_k_scores = np.take_along_axis(similarities, top_k_indices, axis=1)

    return top_k_scores, top_k_indices, similarities


class BiEncoderIndex:
    """
    Embedded source and target schemas, ranked by cosine.

    With `vector_store=` the vectors are persisted. The two sides are keyed
    separately, so GDC's shared 736-column target is embedded once for all ten
    studies; when both sides are cached, no model is loaded.
    """

    def __init__(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        model_name: str = MPNET_MODEL,
        encoding_mode: str = "header_values_verbose",
        sampling_mode: str = "priority_sampling",
        n_samples: int = 10,
        seed: int = 0,
        *,
        vector_store: "VectorStore | str | Path | None" = None,
        pair_name: str = "",
        batch_size: int = 64,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ):
        self.source_columns = list(source.columns)
        self.target_columns = list(target.columns)

        self.source_profiles = build_column_profiles(
            source, encoding_mode, sampling_mode, n_samples, seed
        )
        self.target_profiles = build_column_profiles(
            target, encoding_mode, sampling_mode, n_samples, seed
        )

        store = as_store(vector_store)
        fp_kwargs = dict(
            model_name=model_name, encoding_mode=encoding_mode,
            sampling_mode=sampling_mode, n_samples=n_samples, seed=seed,
        )
        self.source_key = fingerprint(self.source_profiles, **fp_kwargs)
        self.target_key = fingerprint(self.target_profiles, **fp_kwargs)

        src_mat = tgt_mat = None
        if store is not None:
            src_mat = store.load(self.source_key, self.source_columns)
            tgt_mat = store.load(self.target_key, self.target_columns)

        self.store_hits = {"source": src_mat is not None, "target": tgt_mat is not None}
        self.embedder_loaded = src_mat is None or tgt_mat is None

        if self.embedder_loaded:
            model = load_model(model_name)
            if src_mat is None:
                emb = embed_profiles(self.source_profiles, model,
                                     batch_size, token_budget)
                src_mat = np.stack([emb[c] for c in self.source_columns])
            if tgt_mat is None:
                emb = embed_profiles(self.target_profiles, model,
                                     batch_size, token_budget)
                tgt_mat = np.stack([emb[c] for c in self.target_columns])

        if store is not None and not store.read_only:
            save_kwargs = dict(
                pair_name=pair_name, model_name=model_name,
                encoding_mode=encoding_mode, sampling_mode=sampling_mode,
                n_samples=n_samples, seed=seed,
            )
            if not self.store_hits["source"]:
                store.save(self.source_key, self.source_columns, src_mat,
                           side="source", **save_kwargs)
            if not self.store_hits["target"]:
                store.save(self.target_key, self.target_columns, tgt_mat,
                           side="target", **save_kwargs)

        self._src_mat = src_mat
        self._tgt_mat = tgt_mat

        # The full matrix, so gold can be ranked however deep it sits.
        _, _, self.similarity = compute_cosine_similarity(
            self._src_mat, self._tgt_mat, top_k=len(self.target_columns)
        )

    @property
    def source_embeddings(self) -> np.ndarray:
        """``(n_source, dim)``, rows in `source_columns` order."""
        return self._src_mat

    @property
    def target_embeddings(self) -> np.ndarray:
        """``(n_target, dim)``, rows in `target_columns` order."""
        return self._tgt_mat

    def source_similarity(self) -> np.ndarray:
        """Cosine similarity of every source column to every other one."""
        _, _, sim = compute_cosine_similarity(
            self._src_mat, self._src_mat, top_k=len(self.source_columns)
        )
        return sim

    def cache_report(self) -> dict[str, object]:
        """What the vector store contributed, for the trace metadata."""
        return {
            "source_key": self.source_key,
            "target_key": self.target_key,
            "source_from_store": self.store_hits["source"],
            "target_from_store": self.store_hits["target"],
            "embedder_loaded": self.embedder_loaded,
        }

    def candidates(self, source_column: str, k: int) -> list[tuple[str, float]]:
        """The top-k target columns for one source column, best first."""
        if source_column not in self.source_columns:
            raise KeyError(f"unknown source column: {source_column!r}")
        i = self.source_columns.index(source_column)
        row = self.similarity[i]
        k = min(k, len(self.target_columns))
        # Ties broken by target name, for reproducibility.
        order = sorted(
            range(len(self.target_columns)),
            key=lambda j: (-float(row[j]), self.target_columns[j]),
        )
        return [(self.target_columns[j], float(row[j])) for j in order[:k]]

    def full_ranking(self, source_column: str) -> list[tuple[str, float]]:
        """Every target column, ranked."""
        return self.candidates(source_column, len(self.target_columns))
