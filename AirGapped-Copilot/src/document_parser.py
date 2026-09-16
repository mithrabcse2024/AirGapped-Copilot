"""
AirGapped Copilot — Document Parser
=====================================
Extracts text from PDF files and splits into overlapping chunks
suitable for embedding and retrieval.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

log = logging.getLogger("document_parser")

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


class DocumentChunk:
    """A single chunk of text with source metadata."""

    __slots__ = ("text", "source_file", "page_number", "chunk_index")

    def __init__(self, text: str, source_file: str, page_number: int, chunk_index: int):
        self.text = text
        self.source_file = source_file
        self.page_number = page_number
        self.chunk_index = chunk_index

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
        }

    def __repr__(self):
        return (
            f"DocumentChunk(file={self.source_file!r}, "
            f"page={self.page_number}, chunk={self.chunk_index}, "
            f"len={len(self.text)})"
        )


class DocumentParser:
    """
    Parses PDF documents and splits them into overlapping text chunks.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_file(self, file_path: str) -> List[DocumentChunk]:
        """Parse a single PDF file and return a list of DocumentChunks."""
        path = Path(file_path)
        if not path.exists():
            log.error(f"File not found: {path}")
            return []
        valid_extensions = {".pdf", ".txt", ".md", ".text"}
        if path.suffix.lower() not in valid_extensions:
            log.warning(f"Skipping unsupported file type: {path}")
            return []

        filename = path.name
        log.info(f"📄  Parsing: {filename}")

        if path.suffix.lower() in {".txt", ".md", ".text"}:
            try:
                pages = {1: path.read_text(encoding="utf-8", errors="replace")}
            except Exception as e:
                log.error(f"Error reading text file {path}: {e}")
                return []
        else:
            # Try pypdf first
            pages = self._extract_pypdf(path)

            # If pypdf yields too little text, try pdfplumber
            total_chars = sum(len(p) for p in pages.values())
            if total_chars < 50:
                log.info(f"   pypdf yielded only {total_chars} chars; trying pdfplumber")
                pages = self._extract_pdfplumber(path)

        # Chunk all pages
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page_num in sorted(pages.keys()):
            text = pages[page_num].strip()
            if not text:
                continue

            page_chunks = self._split_text(text)
            for chunk_text in page_chunks:
                all_chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        source_file=filename,
                        page_number=page_num,
                        chunk_index=global_chunk_idx,
                    )
                )
                global_chunk_idx += 1

        total_chars = sum(len(p) for p in pages.values())
        log.info(
            f"   Extracted {len(all_chunks)} chunks from {len(pages)} pages "
            f"({total_chars:,} characters)"
        )
        return all_chunks

    def parse_files(self, file_paths: List[str]) -> List[DocumentChunk]:
        """Parse multiple PDF files and return combined chunks."""
        all_chunks: List[DocumentChunk] = []
        for fp in file_paths:
            chunks = self.parse_file(fp)
            all_chunks.extend(chunks)
        log.info(f"📚  Total: {len(all_chunks)} chunks from {len(file_paths)} files")
        return all_chunks

    def _extract_pypdf(self, path: Path) -> Dict[int, str]:
        pages: Dict[int, str] = {}
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pages[i] = text
        except ImportError:
            log.warning("pypdf not installed. Run: pip install pypdf")
        except Exception as e:
            log.error(f"pypdf extraction failed for {path.name}: {e}")
        return pages

    def _extract_pdfplumber(self, path: Path) -> Dict[int, str]:
        pages: Dict[int, str] = {}
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            for row in table:
                                row_text = " | ".join(
                                    str(cell) if cell else "" for cell in row
                                )
                                text += "\n" + row_text
                    pages[i] = text
        except ImportError:
            log.warning("pdfplumber not installed. Run: pip install pdfplumber")
        except Exception as e:
            log.error(f"pdfplumber extraction failed for {path.name}: {e}")
        return pages

    def _split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks with sentence-boundary awareness."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk = text[start:end]

            if end < text_len:
                for sep in [". ", "\n", "; ", "? ", "! "]:
                    last_sep = chunk.rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        chunk = chunk[: last_sep + len(sep)]
                        end = start + len(chunk)
                        break

            clean_chunk = chunk.strip()
            if clean_chunk:
                chunks.append(clean_chunk)

            # Guarantee strictly forward progress
            next_start = end - self.chunk_overlap
            if next_start <= start:
                start = end
            else:
                start = next_start

            if start >= text_len:
                break

        return chunks
