"""
Persisted column embeddings.

A cached run never loads the embedder, which leaves the GPU free for a local
LLM. The key is a digest of the exact profile strings that would be embedded
plus the configuration, so a changed CSV or sampling setting misses the cache
instead of returning stale vectors.

Stored in DuckDB as `FLOAT[]`: one file, no server. There is no ANN index;
similarity is a dense matrix product over a few thousand vectors at most.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

# Bumped when the stored layout changes incompatibly.
SCHEMA_VERSION = 1

DEFAULT_STORE_PATH = "results/vectors/embeddings.duckdb"

_SIDES = ("source", "target")


def fingerprint(
    profiles: Mapping[str, str],
    *,
    model_name: str,
    encoding_mode: str,
    sampling_mode: str,
    n_samples: int,
    seed: int,
) -> str:
    """
    A stable digest of everything that determines these vectors: the profile
    text, in column order, and the configuration that produced it.
    """
    h = hashlib.sha256()
    h.update(f"v{SCHEMA_VERSION}\n".encode())
    for field in (model_name, encoding_mode, sampling_mode):
        h.update(f"{field}\n".encode("utf-8"))
    h.update(f"{int(n_samples)}\n{int(seed)}\n".encode())
    for name, text in profiles.items():
        # Length-prefixed so ("ab", "c") and ("a", "bc") cannot collide.
        h.update(f"{len(str(name))}:{name}\n{len(text)}:{text}\n".encode("utf-8"))
    return h.hexdigest()[:32]


class VectorStore:
    """DuckDB store of column embeddings, keyed by fingerprint. Opened lazily."""

    def __init__(self, path: str | Path = DEFAULT_STORE_PATH, read_only: bool = False):
        self.path = Path(path)
        self.read_only = read_only
        self._con: Any | None = None

    def _connect(self):
        if self._con is not None:
            return self._con

        import duckdb

        if not self.read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        elif not self.path.exists():
            raise FileNotFoundError(f"vector store not found: {self.path}")

        self._con = duckdb.connect(str(self.path), read_only=self.read_only)
        if not self.read_only:
            self._create_schema(self._con)
        return self._con

    @staticmethod
    def _create_schema(con) -> None:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                key             VARCHAR PRIMARY KEY,
                side            VARCHAR NOT NULL,
                pair_name       VARCHAR,
                model_name      VARCHAR,
                encoding_mode   VARCHAR,
                sampling_mode   VARCHAR,
                n_samples       INTEGER,
                seed            INTEGER,
                dim             INTEGER,
                n_columns       INTEGER,
                schema_version  INTEGER,
                created_at      TIMESTAMP,
                extra           VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                key         VARCHAR NOT NULL,
                position    INTEGER NOT NULL,
                column_name VARCHAR NOT NULL,
                vector      FLOAT[] NOT NULL,
                PRIMARY KEY (key, position)
            )
            """
        )

    def close(self) -> None:
        if self._con is not None:
            try:
                self._con.close()
            finally:
                self._con = None

    def __enter__(self) -> "VectorStore":
        self._connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def load(self, key: str, columns: Iterable[str]) -> np.ndarray | None:
        """
        Vectors for `key` as a ``(len(columns), dim)`` float32 matrix, or None
        on a miss or if the stored column names differ from `columns`.
        """
        wanted = list(columns)
        try:
            con = self._connect()
        except FileNotFoundError:
            return None

        rows = con.execute(
            "SELECT column_name, vector FROM embeddings WHERE key = ? ORDER BY position",
            [key],
        ).fetchall()
        if not rows:
            return None
        if [name for name, _ in rows] != wanted:
            return None

        return np.asarray([vec for _, vec in rows], dtype=np.float32)

    def has(self, key: str) -> bool:
        try:
            con = self._connect()
        except FileNotFoundError:
            return False
        row = con.execute("SELECT 1 FROM runs WHERE key = ?", [key]).fetchone()
        return row is not None

    def entries(self) -> list[dict[str, Any]]:
        """Every stored run, newest first."""
        try:
            con = self._connect()
        except FileNotFoundError:
            return []
        cols = [
            "key", "side", "pair_name", "model_name", "encoding_mode",
            "sampling_mode", "n_samples", "seed", "dim", "n_columns",
            "schema_version", "created_at",
        ]
        rows = con.execute(
            f"SELECT {', '.join(cols)} FROM runs ORDER BY created_at DESC"
        ).fetchall()
        return [dict(zip(cols, r)) for r in rows]

    def save(
        self,
        key: str,
        columns: Iterable[str],
        matrix: np.ndarray,
        *,
        side: str,
        pair_name: str = "",
        model_name: str = "",
        encoding_mode: str = "",
        sampling_mode: str = "",
        n_samples: int = 0,
        seed: int = 0,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Write one side's vectors, replacing any rows for `key`."""
        if self.read_only:
            raise RuntimeError(f"vector store opened read-only: {self.path}")
        if side not in _SIDES:
            raise ValueError(f"side must be one of {_SIDES}, got {side!r}")

        names = list(columns)
        arr = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
        if arr.ndim != 2 or arr.shape[0] != len(names):
            raise ValueError(
                f"matrix shape {arr.shape} does not match {len(names)} column names"
            )

        con = self._connect()
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("DELETE FROM embeddings WHERE key = ?", [key])
            con.execute("DELETE FROM runs WHERE key = ?", [key])
            con.executemany(
                "INSERT INTO embeddings (key, position, column_name, vector) "
                "VALUES (?, ?, ?, ?)",
                [
                    (key, i, name, arr[i].tolist())
                    for i, name in enumerate(names)
                ],
            )
            con.execute(
                "INSERT INTO runs (key, side, pair_name, model_name, encoding_mode, "
                "sampling_mode, n_samples, seed, dim, n_columns, schema_version, "
                "created_at, extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    key, side, pair_name, model_name, encoding_mode, sampling_mode,
                    int(n_samples), int(seed), int(arr.shape[1]), len(names),
                    SCHEMA_VERSION,
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    json.dumps(dict(extra or {}), default=str),
                ],
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

    def drop(self, key: str) -> None:
        con = self._connect()
        con.execute("DELETE FROM embeddings WHERE key = ?", [key])
        con.execute("DELETE FROM runs WHERE key = ?", [key])


def as_store(store: "VectorStore | str | Path | None") -> VectorStore | None:
    """A VectorStore from a path, a store, or None."""
    if store is None or isinstance(store, VectorStore):
        return store
    return VectorStore(store)
