"""
src/rag/prompts.py

Prompt templates for the bank statement RAG chain.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise and helpful personal finance assistant \
with access to the user's DCU bank statements.

RULES:
1. Answer ONLY using the context documents provided below.
2. If the context does not contain enough information to answer, say:
   "I don't have enough information in your statements to answer that."
3. Always cite which account and date your facts come from.
4. For amounts, always include the dollar sign and two decimal places.
5. Never guess, estimate, or use knowledge outside the provided context.
6. If the question spans multiple accounts, address each one separately.
7. Be concise — lead with the direct answer, then supporting detail.

CONTEXT:
{context}
"""

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  "{question}"),
])

# Multi-turn: includes chat history for follow-up questions
RAG_PROMPT_WITH_HISTORY = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{history}"),
    ("human", "{question}"),
])