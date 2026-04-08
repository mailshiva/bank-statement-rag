"""
notebooks/test_loader.py

Tests the full ingest pipeline end-to-end:
  PDF → parse → chunk → embed → ChromaDB → retrieve

Run from project root:
    python notebooks/test_loader.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.ingest.loader import StatementLoader

# ---------------------------------------------------------------------------
# 1. Load the statement
# ---------------------------------------------------------------------------
print("── Loading statement into ChromaDB ──────────────────────────\n")

loader = StatementLoader()
loader.load_statement("data/raw/stmt_20250131.pdf")

# ---------------------------------------------------------------------------
# 2. Collection stats
# ---------------------------------------------------------------------------
print("\n── Collection stats ──────────────────────────────────────────\n")

stats = loader.collection_stats()
for k, v in stats.items():
    print(f"  {k:<20}: {v}")

# ---------------------------------------------------------------------------
# 3. Similarity search spot checks
# ---------------------------------------------------------------------------
print("\n── Retrieval spot checks ─────────────────────────────────────\n")

vs = loader.vector_store

queries = [
    "How much was my payroll deposit in January?",
    "What is the current vehicle loan balance?",
    "How much interest did I pay on the loan?",
    "What is my savings account balance?",
    "Did I make any transfers out of checking?",
]

for query in queries:
    print(f"Q: {query}")
    results = vs.similarity_search(query, k=2)
    for doc in results:
        chunk_type = doc.metadata.get("chunk_type", "—")
        acct_type  = doc.metadata.get("account_type", "—")
        date       = doc.metadata.get("date", "—")
        print(f"  [{chunk_type}] [{acct_type}] date={date}")
        print(f"  {doc.page_content[:120]}...")
    print()

# ---------------------------------------------------------------------------
# 4. Filtered retrieval
# ---------------------------------------------------------------------------
print("── Filtered retrieval (checking only) ───────────────────────\n")

retriever = loader.as_retriever_with_filter(
    k=3,
    account_type="checking",
)
docs = retriever.invoke("how much money came in this month?")
for doc in docs:
    print(f"  [{doc.metadata.get('chunk_type')}] {doc.page_content[:120]}...")

print("\n── Done ─────────────────────────────────────────────────────")