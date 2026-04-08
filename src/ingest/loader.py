"""
src/ingest/loader.py

Embeds LangChain Documents and persists them to ChromaDB.

Handles:
  - Embedding via Google Gemini (gemini-embedding-001)
  - API key loaded from macOS Keychain
  - Idempotent loading: re-running on the same PDF skips already-loaded docs
  - Loading a single statement or a full directory of PDFs
  - Returning a retriever ready for the RAG chain

Usage:
    from src.ingest.loader import StatementLoader

    loader = StatementLoader()
    loader.load_statement("data/raw/stmt_20250131.pdf")
    loader.load_directory("data/raw/")
    retriever = loader.as_retriever(k=5)
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .chunker import StatementChunker
from .parser import StatementParser


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "bank_statements"
EMBEDDING_MODEL = "gemini-embedding-001"


# ---------------------------------------------------------------------------
# Keychain helper
# ---------------------------------------------------------------------------

def _get_keychain_secret(account: str, service: str = "portfolio_advisor") -> str:
    """
    Retrieve a secret from macOS Keychain.
    Equivalent to:
        security find-generic-password -a <account> -s <service> -w
    """
    result = subprocess.run(
        ["security", "find-generic-password", "-a", account, "-s", service, "-w"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not retrieve '{account}' from Keychain service "
            f"'{service}'.\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class StatementLoader:
    """
    Parses → chunks → embeds → stores DCU statement PDFs into ChromaDB.

    Idempotency:
        Each document gets a deterministic ID derived from the source
        filename + chunk type + account type + date + content prefix.
        ChromaDB upsert semantics mean re-running on the same PDF is
        safe — existing docs are overwritten, not duplicated.
    """

    def __init__(
        self,
        chroma_dir:      str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.parser  = StatementParser()
        self.chunker = StatementChunker()

        gemini_api_key = _get_keychain_secret("gemini_api_key")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=gemini_api_key,
        )

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=chroma_dir,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def load_statement(self, pdf_path: str | Path) -> int:
        """
        Parse, chunk, embed and store a single PDF.
        Returns the number of documents added/updated.
        """
        pdf_path = Path(pdf_path)
        print(f"Loading: {pdf_path.name}")

        result = self.parser.parse(pdf_path)
        docs   = self.chunker.chunk(result)
        ids    = [self._make_id(doc) for doc in docs]

        self.vector_store.add_documents(documents=docs, ids=ids)

        print(f"  ✓ {len(docs)} documents upserted  "
              f"(period: {result.period})")
        return len(docs)

    def load_directory(self, directory: str | Path) -> int:
        """
        Load all PDFs found in a directory (non-recursive).
        Returns total documents added/updated across all files.
        """
        directory = Path(directory)
        pdfs = sorted(directory.glob("*.pdf"))

        if not pdfs:
            print(f"No PDFs found in {directory}")
            return 0

        print(f"Found {len(pdfs)} PDF(s) in {directory}\n")
        total = 0
        for pdf in pdfs:
            try:
                total += self.load_statement(pdf)
            except Exception as e:
                print(f"  ✗ Failed to load {pdf.name}: {e}")

        print(f"\nDone. {total} total documents in ChromaDB.")
        return total

    def as_retriever(self, k: int = 5, **kwargs):
        """
        Return a LangChain retriever over the vector store.

        Args:
            k: number of chunks to retrieve per query
        """
        search_kwargs = {"k": k}
        search_kwargs.update(kwargs)
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

    def as_retriever_with_filter(
        self,
        k:            int = 5,
        account_type: str | None = None,
        month:        str | None = None,
        year:         str | None = None,
        chunk_type:   str | None = None,
    ):
        """
        Convenience retriever with typed metadata filters.

        Example:
            retriever = loader.as_retriever_with_filter(
                k=4,
                account_type="checking",
                month="2025-01",
            )
        """
        where: dict = {}
        if account_type:
            where["account_type"] = account_type
        if month:
            where["month"] = month
        if year:
            where["year"] = year
        if chunk_type:
            where["chunk_type"] = chunk_type

        search_kwargs: dict = {"k": k}
        if where:
            if len(where) == 1:
                search_kwargs["filter"] = where
            else:
                search_kwargs["filter"] = {
                    "$and": [{k: v} for k, v in where.items()]
                }

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

    def collection_stats(self) -> dict:
        """Return a summary of what is currently in ChromaDB."""
        col   = self.vector_store._collection
        count = col.count()

        if count == 0:
            return {"total_docs": 0}

        results   = col.get(include=["metadatas"])
        metadatas = results["metadatas"]

        months     = sorted({m.get("month", "")        for m in metadatas if m.get("month")})
        acct_types = sorted({m.get("account_type", "") for m in metadatas if m.get("account_type")})
        sources    = sorted({m.get("source_file", "")  for m in metadatas if m.get("source_file")})

        return {
            "total_docs":    count,
            "months":        months,
            "account_types": acct_types,
            "source_files":  sources,
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    def _make_id(self, doc: Document) -> str:
        """
        Deterministic document ID so re-loading the same PDF
        upserts rather than duplicates.
        """
        meta = doc.metadata
        raw  = "|".join([
            meta.get("source_file",  ""),
            meta.get("chunk_type",   ""),
            meta.get("account_type", ""),
            meta.get("date",         ""),
            doc.page_content[:40],
        ])
        return hashlib.md5(raw.encode()).hexdigest()