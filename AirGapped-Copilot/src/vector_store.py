"""
AirGapped Copilot — Vector Store
==================================
Local vector database backed by LanceDB for storing and retrieving
document embeddings entirely offline. No server required.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

log = logging.getLogger("vector_store")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "vector_db"
TABLE_NAME = "document_chunks"
EMBEDDING_DIM = 384


class VectorStore:
    """
    LanceDB-backed local vector store for document chunk embeddings.
    Falls back to an in-memory + JSON-persisted store when LanceDB is unavailable.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = None
        self.table = None
        self._fallback_data: List[dict] = []
        self._fallback_file = self.db_path / "_fallback_store.json"
        self._use_fallback = False

        self._init_db()

    def _init_db(self):
        """Connect to (or create) the LanceDB database."""
        try:
            import lancedb
            self.db = lancedb.connect(str(self.db_path))
            log.info(f"✅  LanceDB connected: {self.db_path}")

            existing_tables = self.db.table_names()
            if TABLE_NAME in existing_tables:
                self.table = self.db.open_table(TABLE_NAME)
                count = self.table.count_rows()
                log.info(f"   Opened table '{TABLE_NAME}' with {count:,} rows")
            else:
                log.info(f"   Table '{TABLE_NAME}' will be created on first insert")

        except ImportError:
            log.warning("lancedb not installed. Using JSON-persisted fallback vector store.")
            self._use_fallback = True
            self._load_fallback()
        except Exception as e:
            log.error(f"LanceDB init failed: {e}. Using fallback.")
            self._use_fallback = True
            self._load_fallback()

    def _load_fallback(self):
        """Load persisted fallback data from disk."""
        if self._fallback_file.exists():
            try:
                with open(self._fallback_file, "r", encoding="utf-8") as f:
                    self._fallback_data = json.load(f)
                log.info(f"   Loaded {len(self._fallback_data)} records from fallback store")
            except Exception as e:
                log.error(f"Failed to load fallback store: {e}")
                self._fallback_data = []

    def _save_fallback(self):
        """Persist fallback data to disk."""
        try:
            with open(self._fallback_file, "w", encoding="utf-8") as f:
                json.dump(self._fallback_data, f)
        except Exception as e:
            log.error(f"Failed to save fallback store: {e}")

    # ------------------------------------------------------------------
    # Add documents
    # ------------------------------------------------------------------
    def add_documents(self, chunks: List[dict], embeddings: np.ndarray) -> int:
        """
        Insert document chunks with their embeddings.

        Parameters
        ----------
        chunks : list[dict]
            Each dict must have: text, source_file, page_number, chunk_index
        embeddings : np.ndarray
            Shape (N, 384) float32 array.

        Returns
        -------
        int  Number of chunks inserted.
        """
        if len(chunks) == 0:
            return 0

        assert len(chunks) == len(embeddings), (
            f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
        )

        records = []
        for i, chunk in enumerate(chunks):
            vec = embeddings[i].tolist()
            if len(vec) < EMBEDDING_DIM:
                vec.extend([0.0] * (EMBEDDING_DIM - len(vec)))
            elif len(vec) > EMBEDDING_DIM:
                vec = vec[:EMBEDDING_DIM]

            records.append({
                "id": str(uuid.uuid4()),
                "text": chunk.get("text", ""),
                "source_file": chunk.get("source_file", "unknown"),
                "page_number": int(chunk.get("page_number", 0)),
                "chunk_index": int(chunk.get("chunk_index", 0)),
                "vector": vec,
            })

        if not self._use_fallback and self.db is not None:
            try:
                if self.table is None:
                    self.table = self.db.create_table(TABLE_NAME, records)
                    log.info(f"   Created table '{TABLE_NAME}' with {len(records)} rows")
                else:
                    self.table.add(records)
                    log.info(f"   Added {len(records)} rows to '{TABLE_NAME}'")
                return len(records)
            except Exception as e:
                log.error(f"LanceDB insert failed: {e}. Falling back.")
                self._use_fallback = True

        # Fallback
        self._fallback_data.extend(records)
        self._save_fallback()
        log.info(f"   Added {len(records)} rows to fallback store")
        return len(records)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[dict]:
        """Cosine similarity search returning top-k matching chunks."""
        query_vec = query_vector.flatten().astype(np.float32).tolist()

        if len(query_vec) < EMBEDDING_DIM:
            query_vec.extend([0.0] * (EMBEDDING_DIM - len(query_vec)))
        elif len(query_vec) > EMBEDDING_DIM:
            query_vec = query_vec[:EMBEDDING_DIM]

        if not self._use_fallback and self.table is not None:
            try:
                results = (
                    self.table.search(query_vec)
                    .metric("cosine")
                    .limit(top_k)
                    .to_pandas()
                )
                output = []
                for _, row in results.iterrows():
                    output.append({
                        "text": row.get("text", ""),
                        "source_file": row.get("source_file", ""),
                        "page_number": int(row.get("page_number", 0)),
                        "chunk_index": int(row.get("chunk_index", 0)),
                        "score": float(1.0 - row.get("_distance", 1.0)),
                    })
                return output
            except Exception as e:
                log.error(f"LanceDB search failed: {e}")

        # Fallback: brute-force cosine similarity
        return self._fallback_search(query_vec, top_k)

    def _fallback_search(self, query_vec: list, top_k: int) -> List[dict]:
        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        scores = []
        for record in self._fallback_data:
            v = np.array(record["vector"], dtype=np.float32)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                sim = 0.0
            else:
                sim = float(np.dot(q, v) / (q_norm * v_norm))
            scores.append((sim, record))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, record in scores[:top_k]:
            results.append({
                "text": record["text"],
                "source_file": record["source_file"],
                "page_number": record["page_number"],
                "chunk_index": record["chunk_index"],
                "score": score,
            })
        return results

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def get_indexed_files(self) -> List[dict]:
        """Return list of indexed files with chunk counts."""
        if not self._use_fallback and self.table is not None:
            try:
                df = self.table.to_pandas()
                grouped = df.groupby("source_file").size().reset_index(name="chunk_count")
                return grouped.to_dict("records")
            except Exception as e:
                log.error(f"get_indexed_files failed: {e}")

        file_counts: Dict[str, int] = {}
        for record in self._fallback_data:
            f = record["source_file"]
            file_counts[f] = file_counts.get(f, 0) + 1
        return [{"source_file": f, "chunk_count": c} for f, c in file_counts.items()]

    def delete_file(self, filename: str) -> int:
        """Delete all chunks belonging to a specific file."""
        if not self._use_fallback and self.table is not None:
            try:
                before = self.table.count_rows()
                self.table.delete(f'source_file = "{filename}"')
                after = self.table.count_rows()
                deleted = before - after
                log.info(f"   Deleted {deleted} chunks for '{filename}'")
                return deleted
            except Exception as e:
                log.error(f"delete_file failed: {e}")

        before = len(self._fallback_data)
        self._fallback_data = [r for r in self._fallback_data if r["source_file"] != filename]
        deleted = before - len(self._fallback_data)
        if deleted:
            self._save_fallback()
        log.info(f"   Deleted {deleted} chunks for '{filename}'")
        return deleted

    def total_chunks(self) -> int:
        """Return total number of stored chunks."""
        if not self._use_fallback and self.table is not None:
            try:
                return self.table.count_rows()
            except Exception:
                pass
        return len(self._fallback_data)

    def clear(self):
        """Drop the entire table and start fresh."""
        if self.db is not None:
            try:
                self.db.drop_table(TABLE_NAME, ignore_missing=True)
                self.table = None
                log.info(f"   Cleared table '{TABLE_NAME}'")
            except Exception as e:
                log.error(f"clear failed: {e}")

        self._fallback_data.clear()
        self._save_fallback()
