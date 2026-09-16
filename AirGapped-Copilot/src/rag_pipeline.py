"""
AirGapped Copilot — RAG Pipeline
==================================
Full Retrieval-Augmented Generation chain:
  PDF → Chunks → Embeddings (NPU) → LanceDB → SLM (NPU) → Response
Everything runs 100% offline on the Snapdragon Hexagon NPU.
"""

import time
import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field

import numpy as np

from src.npu_engine import EmbeddingEngine, SLMEngine
from src.document_parser import DocumentParser, DocumentChunk
from src.vector_store import VectorStore

log = logging.getLogger("rag_pipeline")


@dataclass
class Citation:
    """A single citation referencing a source document."""
    source_file: str
    page_number: int
    chunk_index: int
    text: str
    relevance_score: float


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""
    answer: str
    citations: List[Citation] = field(default_factory=list)
    query: str = ""
    latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    tokens_per_sec: float = 0.0
    tokens_generated: int = 0
    context_chunks_used: int = 0


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    file_path: str
    filename: str
    success: bool
    chunks_created: int = 0
    error_message: str = ""
    latency_ms: float = 0.0


SYSTEM_PROMPT = """You are a precise, citation-aware AI assistant running entirely offline on a Qualcomm Snapdragon NPU. Your task is to answer questions based ONLY on the provided document context. Follow these rules:

1. ONLY use information from the provided context passages to answer.
2. If the context doesn't contain enough information, say so clearly.
3. Always reference which citation(s) support your answer using [Citation N] format.
4. Be concise, accurate, and professional.
5. For legal/medical documents, note any relevant caveats or qualifications."""

CONTEXT_TEMPLATE = """
--- DOCUMENT CONTEXT ---
{context}
--- END CONTEXT ---

USER QUESTION: {question}

ANSWER (cite sources using [Citation N]):"""


class RAGPipeline:
    """
    Orchestrates the full offline RAG chain.
    """

    def __init__(
        self,
        embedding_model_dir: Optional[str] = None,
        slm_model_dir: Optional[str] = None,
        db_path: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
    ):
        self.top_k = top_k

        log.info("=" * 50)
        log.info("  Initializing RAG Pipeline")
        log.info("=" * 50)

        self.parser = DocumentParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = VectorStore(db_path=db_path)
        self.embedding_engine = EmbeddingEngine(model_dir=embedding_model_dir)
        self.slm_engine = SLMEngine(model_dir=slm_model_dir)

        log.info("✅  RAG Pipeline initialized")
        log.info(f"   SLM:       {self.slm_engine.status_text}")
        log.info(f"   VectorDB:  {self.vector_store.total_chunks()} chunks stored")

    def ingest_chunks(self, chunks: List[DocumentChunk], source_name: str = "custom_document") -> IngestionResult:
        """Embed and store pre-parsed DocumentChunk objects."""
        t_start = time.perf_counter()
        try:
            chunk_dicts = [c.to_dict() for c in chunks]
            texts = [c.text for c in chunks]
            embeddings = self.embedding_engine.embed_batch(texts)
            inserted = self.vector_store.add_documents(chunk_dicts, embeddings)
            t_end = time.perf_counter()
            return IngestionResult(
                file_path=source_name,
                filename=source_name,
                success=True,
                chunks_created=inserted,
                latency_ms=(t_end - t_start) * 1000.0,
            )
        except Exception as e:
            t_end = time.perf_counter()
            return IngestionResult(
                file_path=source_name,
                filename=source_name,
                success=False,
                error_message=str(e),
                latency_ms=(t_end - t_start) * 1000.0,
            )

    def ingest_documents(
        self,
        file_paths: list,
        progress_callback=None,
    ) -> List[IngestionResult]:
        """Parse, embed, and store documents (or DocumentChunk lists)."""
        results: List[IngestionResult] = []
        if not file_paths:
            return results

        # Check if caller passed a list of DocumentChunks directly
        if all(hasattr(item, "text") and hasattr(item, "to_dict") for item in file_paths):
            res = self.ingest_chunks(file_paths, getattr(file_paths[0], "source_file", "document"))
            return [res]

        total = len(file_paths)
        for idx, fp in enumerate(file_paths):
            # Check if this item is a DocumentChunk
            if hasattr(fp, "text") and hasattr(fp, "to_dict"):
                res = self.ingest_chunks([fp], getattr(fp, "source_file", f"chunk_{idx}"))
                results.append(res)
                continue

            filename = Path(fp).name
            t_start = time.perf_counter()

            if progress_callback:
                progress_callback(idx, total, filename)

            try:
                chunks = self.parser.parse_file(fp)
                if not chunks:
                    results.append(IngestionResult(
                        file_path=str(fp), filename=filename, success=False,
                        error_message="No text extracted from document",
                    ))
                    continue

                chunk_dicts = [c.to_dict() for c in chunks]
                texts = [c.text for c in chunks]

                log.info(f"   Embedding {len(texts)} chunks...")
                embeddings = self.embedding_engine.embed_batch(texts)

                inserted = self.vector_store.add_documents(chunk_dicts, embeddings)

                t_end = time.perf_counter()
                results.append(IngestionResult(
                    file_path=str(fp), filename=filename, success=True,
                    chunks_created=inserted,
                    latency_ms=(t_end - t_start) * 1000.0,
                ))
                log.info(f"   ✅  {filename}: {inserted} chunks indexed")

            except Exception as e:
                t_end = time.perf_counter()
                log.error(f"   ❌  {filename}: {e}")
                results.append(IngestionResult(
                    file_path=fp, filename=filename, success=False,
                    error_message=str(e),
                    latency_ms=(t_end - t_start) * 1000.0,
                ))

        if progress_callback:
            progress_callback(total, total, "Done")

        return results

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> RAGResponse:
        """Run the full RAG pipeline for a user question."""
        k = top_k or self.top_k
        t_total_start = time.perf_counter()

        # Embed query
        t0 = time.perf_counter()
        query_embedding = self.embedding_engine.embed(question)
        t_embed = (time.perf_counter() - t0) * 1000.0

        # Retrieve context
        t0 = time.perf_counter()
        search_results = self.vector_store.search(query_embedding, top_k=k)
        t_retrieve = (time.perf_counter() - t0) * 1000.0

        citations: List[Citation] = []
        context_parts: List[str] = []

        for i, result in enumerate(search_results, start=1):
            citation = Citation(
                source_file=result["source_file"],
                page_number=result["page_number"],
                chunk_index=result["chunk_index"],
                text=result["text"],
                relevance_score=result.get("score", 0.0),
            )
            citations.append(citation)
            context_parts.append(
                f"[Citation {i} | {result['source_file']}, "
                f"Page {result['page_number']}]\n"
                f"{result['text']}"
            )

        context_text = "\n\n".join(context_parts) if context_parts else (
            "No documents have been indexed yet. Please ingest PDF documents first."
        )

        # Build prompt
        user_prompt = CONTEXT_TEMPLATE.format(context=context_text, question=question)
        full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

        # Generate answer
        t0 = time.perf_counter()
        answer = self.slm_engine.generate(
            prompt=full_prompt, max_tokens=max_tokens, temperature=temperature,
        )
        t_generate = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_total_start) * 1000.0

        return RAGResponse(
            answer=answer.strip(),
            citations=citations,
            query=question,
            latency_ms=t_total,
            embedding_latency_ms=t_embed,
            retrieval_latency_ms=t_retrieve,
            generation_latency_ms=t_generate,
            tokens_per_sec=self.slm_engine.tokens_per_sec,
            tokens_generated=self.slm_engine.tokens_generated,
            context_chunks_used=len(citations),
        )

    def get_indexed_files(self) -> List[dict]:
        return self.vector_store.get_indexed_files()

    def delete_file(self, filename: str) -> int:
        return self.vector_store.delete_file(filename)

    def total_chunks(self) -> int:
        return self.vector_store.total_chunks()

    def clear_database(self):
        self.vector_store.clear()

    @property
    def npu_status(self) -> str:
        return self.slm_engine.status_text

    @property
    def embedding_status(self) -> str:
        if self.embedding_engine._mock_mode:
            return "Mock Embeddings (No Model)"
        elif self.embedding_engine.engine and self.embedding_engine.engine.is_npu_active:
            return "Qualcomm Hexagon HTP — QnnHtp.dll"
        elif self.embedding_engine.engine:
            return "CPU Fallback — onnxruntime"
        return "Not Loaded"
