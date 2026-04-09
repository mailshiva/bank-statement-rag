"""
src/rag/chain.py

Core RAG chain: question → retrieve → prompt → LLM → answer.

Two chain variants:
  - SimpleRAGChain   : single-turn, stateless
  - ChatRAGChain     : multi-turn, maintains conversation history

Uses:
  - ChromaDB as the vector store (via StatementLoader)
  - Gemini embeddings for retrieval
  - Claude (Anthropic) as the generation LLM

Usage:
    from src.rag.chain import ChatRAGChain

    chain = ChatRAGChain()
    answer = chain.ask("How much was my payroll in January?")
    answer = chain.ask("And how much went to the loan?")  # follow-up works
    chain.print_history()
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from ..ingest.loader import StatementLoader
from .prompts import RAG_PROMPT, RAG_PROMPT_WITH_HISTORY


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_K     = 5      # chunks retrieved per query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_keychain_secret(account: str, service: str = "portfolio_advisor") -> str:
    result = subprocess.run(
        ["security", "find-generic-password", "-a", account, "-s", service, "-w"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not retrieve '{account}' from Keychain service '{service}'.\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def _format_docs(docs: list[Document]) -> str:
    """
    Format retrieved documents into a numbered context block.
    Each doc shows its metadata header + content so the LLM
    can cite sources precisely.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header_parts = [f"[Doc {i}]"]

        if acct := meta.get("account_type"):
            header_parts.append(f"account={acct}")
        if date := meta.get("date"):
            header_parts.append(f"date={date}")
        if month := meta.get("month"):
            header_parts.append(f"month={month}")
        if chunk := meta.get("chunk_type"):
            header_parts.append(f"type={chunk}")

        header = " | ".join(header_parts)
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Turn dataclass (for history tracking)
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    question: str
    answer:   str
    sources:  list[Document] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simple single-turn chain
# ---------------------------------------------------------------------------

class SimpleRAGChain:
    """
    Stateless single-turn RAG chain.
    Good for one-off questions, testing, and batch processing.
    """

    def __init__(
        self,
        model:      str = DEFAULT_MODEL,
        k:          int = DEFAULT_K,
        chroma_dir: str = "chroma_db",
    ):
        anthropic_key = _get_keychain_secret("anthropic_api_key")

        self.llm = ChatAnthropic(
            model=model,
            api_key=anthropic_key,
            temperature=0,      # deterministic — we want facts, not creativity
            max_tokens=1024,
        )

        self.loader    = StatementLoader(chroma_dir=chroma_dir)
        self.retriever = self.loader.as_retriever(k=k)
        self._last_docs: list[Document] = []

        # LCEL chain
        self.chain = (
            {
                "context":  self.retriever | RunnableLambda(self._capture_and_format),
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        """Ask a single question. Returns the answer string."""
        return self.chain.invoke(question)

    def ask_with_sources(self, question: str) -> tuple[str, list[Document]]:
        """Ask a question and return (answer, source_documents)."""
        answer = self.chain.invoke(question)
        return answer, self._last_docs

    def _capture_and_format(self, docs: list[Document]) -> str:
        """Side-effect: capture retrieved docs for inspection."""
        self._last_docs = docs
        return _format_docs(docs)


# ---------------------------------------------------------------------------
# Multi-turn chat chain
# ---------------------------------------------------------------------------

class ChatRAGChain:
    """
    Stateful multi-turn RAG chain with conversation history.

    History is included in each prompt so Claude can resolve
    references like "and what about February?" or "compare that
    to my checking account."
    """

    def __init__(
        self,
        model:      str = DEFAULT_MODEL,
        k:          int = DEFAULT_K,
        chroma_dir: str = "chroma_db",
    ):
        anthropic_key = _get_keychain_secret("anthropic_api_key")

        self.llm = ChatAnthropic(
            model=model,
            api_key=anthropic_key,
            temperature=0,
            max_tokens=1024,
        )

        self.loader    = StatementLoader(chroma_dir=chroma_dir)
        self.retriever = self.loader.as_retriever(k=k)
        self.history:  list[Turn] = []

    def ask(self, question: str) -> str:
        """
        Ask a question. Conversation history is automatically
        included so follow-up questions resolve correctly.
        Returns the answer string.
        """
        # Retrieve relevant docs
        docs = self.retriever.invoke(question)

        # Build history as LangChain message tuples
        history_messages = []
        for turn in self.history:
            history_messages.append(("human",     turn.question))
            history_messages.append(("assistant", turn.answer))

        # Format prompt inputs
        prompt_input = {
            "context":  _format_docs(docs),
            "question": question,
            "history":  history_messages,
        }

        # Run: prompt → LLM → parse
        answer = (
            RAG_PROMPT_WITH_HISTORY
            | self.llm
            | StrOutputParser()
        ).invoke(prompt_input)

        # Store turn
        self.history.append(Turn(
            question=question,
            answer=answer,
            sources=docs,
        ))

        return answer

    def print_history(self) -> None:
        """Print the full conversation so far."""
        if not self.history:
            print("No conversation history yet.")
            return

        print("\n" + "=" * 70)
        print("CONVERSATION HISTORY")
        print("=" * 70)
        for i, turn in enumerate(self.history, 1):
            print(f"\n[{i}] Q: {turn.question}")
            print(f"    A: {turn.answer}")
            print(f"    Sources: {len(turn.sources)} docs retrieved")

    def print_last_sources(self) -> None:
        """Print the source documents from the last question."""
        if not self.history:
            print("No questions asked yet.")
            return

        last = self.history[-1]
        print(f"\nSources for: '{last.question}'")
        print("-" * 50)
        for doc in last.sources:
            meta = doc.metadata
            print(f"  [{meta.get('chunk_type')}] "
                  f"[{meta.get('account_type', '—')}] "
                  f"date={meta.get('date', '—')}")
            print(f"  {doc.page_content[:100]}...")

    def reset(self) -> None:
        """Clear conversation history."""
        self.history = []
        print("Conversation history cleared.")