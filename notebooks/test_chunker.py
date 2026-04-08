"""
notebooks/test_chunker.py

Smoke test for the chunker.
Run from project root: python notebooks/test_chunker.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.parser import StatementParser
from src.ingest.chunker import StatementChunker, _normalise_date

# ---------------------------------------------------------------------------
# 1. Date normalisation unit tests
# ---------------------------------------------------------------------------
print("── Date normalisation checks ────────────────────────────────")

cases = [
    # (raw_date, period_start, period_end, expected)
    ("JAN14", "01-01-25", "01-31-25", "2025-01-14"),  # normal single-year
    ("JAN03", "01-01-25", "01-31-25", "2025-01-03"),  # normal single-year
    ("DEC28", "12-01-24", "12-31-24", "2024-12-28"),  # end of year
    ("DEC31", "12-01-24", "01-15-25", "2024-12-31"),  # cross-year: Dec side
    ("JAN10", "12-01-24", "01-15-25", "2025-01-10"),  # cross-year: Jan side
]

all_ok = True
for raw, ps, pe, expected in cases:
    got    = _normalise_date(raw, ps, pe)
    status = "OK" if got == expected else f"FAIL — got {got}"
    print(f"  {raw} | {ps} → {pe} | expected {expected} | {status}")
    if "FAIL" in status:
        all_ok = False

print(f"\n  {'All date checks passed' if all_ok else 'DATE CHECK FAILURES above'}\n")

# ---------------------------------------------------------------------------
# 2. Full chunker smoke test
# ---------------------------------------------------------------------------
print("── Chunker output ───────────────────────────────────────────\n")

result = StatementParser().parse("data/raw/stmt_20250131.pdf")
docs   = StatementChunker().chunk(result)

print(f"Total documents produced: {len(docs)}  (expected: 11)\n")

for i, doc in enumerate(docs, 1):
    chunk_type = doc.metadata.get("chunk_type", "—")
    acct_type  = doc.metadata.get("account_type", "—")
    date       = doc.metadata.get("date", "—")
    print(f"[{i:02d}] [{chunk_type:<20}] [{acct_type:<10}] date={date}")
    print(f"      {doc.page_content[:100]}...")
    print(f"      metadata keys: {sorted(doc.metadata.keys())}\n")

# ---------------------------------------------------------------------------
# 3. Spot-check specific document content
# ---------------------------------------------------------------------------
print("── Spot checks ──────────────────────────────────────────────\n")

txn_docs     = [d for d in docs if d.metadata["chunk_type"] == "transaction"]
summary_docs = [d for d in docs if d.metadata["chunk_type"] == "section_summary"]
overview_doc = [d for d in docs if d.metadata["chunk_type"] == "statement_overview"]

print(f"Transaction docs : {len(txn_docs)}   (expected: 8)")
print(f"Summary docs     : {len(summary_docs)}   (expected: 3)")
print(f"Overview docs    : {len(overview_doc)}   (expected: 1)")

# Verify all transaction docs have ISO dates
bad_dates = [
    d for d in txn_docs
    if not d.metadata.get("date", "").startswith("2025-")
]
print(f"Docs with bad dates: {len(bad_dates)}  (expected: 0)")

# Verify checking transactions have a category
missing_cat = [
    d for d in txn_docs
    if d.metadata.get("account_type") == "checking"
    and "category" not in d.metadata
]
print(f"Checking txns missing category: {len(missing_cat)}  (expected: 0)")

# Verify loan docs have loan_number
missing_loan_num = [
    d for d in docs
    if d.metadata.get("account_type") == "loan"
    and "loan_number" not in d.metadata
]
print(f"Loan docs missing loan_number: {len(missing_loan_num)}  (expected: 0)")

print("\n── Done ─────────────────────────────────────────────────────")