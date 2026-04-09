"""
notebooks/test_chain.py

Tests the RAG chain end-to-end.
Run from project root: python notebooks/test_chain.py
"""

from src.rag.chain import ChatRAGChain

print("── Initialising RAG chain ────────────────────────────────────\n")
chain = ChatRAGChain(model="claude-haiku-4-5", k=5)
print("  ✓ Chain ready\n")

# ---------------------------------------------------------------------------
# 1. Single-turn questions
# ---------------------------------------------------------------------------
print("── Single-turn questions ─────────────────────────────────────\n")

questions = [
    "How much was my payroll deposit in January 2025?",
    "What is my current vehicle loan balance?",
    "How much interest did I pay on my loan in January?",
    "What is my savings account closing balance?",
    "Did I make any external transfers from checking?",
]

for q in questions:
    print(f"Q: {q}")
    answer = chain.ask(q)
    print(f"A: {answer}")
    print()

# ---------------------------------------------------------------------------
# 2. Multi-turn: follow-up questions
# ---------------------------------------------------------------------------
print("── Multi-turn conversation ───────────────────────────────────\n")

chain.reset()

turns = [
    "Give me a summary of my checking account activity in January.",
    "How much of that was payroll income?",
    "And how much went out as loan payments?",
    "What percentage of my income went to the loan payment?",
]

for q in turns:
    print(f"Q: {q}")
    answer = chain.ask(q)
    print(f"A: {answer}")
    print()

# ---------------------------------------------------------------------------
# 3. Edge case: question outside available data
# ---------------------------------------------------------------------------
print("── Edge case: out-of-scope question ─────────────────────────\n")

q = "How much did I spend on groceries in January?"
print(f"Q: {q}")
answer = chain.ask(q)
print(f"A: {answer}")
print()

# ---------------------------------------------------------------------------
# 4. Source inspection
# ---------------------------------------------------------------------------
print("── Source inspection ─────────────────────────────────────────\n")
chain.print_last_sources()

print("\n── Full history ──────────────────────────────────────────────")
chain.print_history()

print("\n── Done ─────────────────────────────────────────────────────")