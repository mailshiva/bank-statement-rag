"""
src/ingest/chunker.py

Converts a StatementResult into a list of LangChain Documents
ready for embedding and loading into ChromaDB.

Chunking strategy:
  - One Document per transaction  (fine-grained, best for specific queries)
  - One Document per section summary  (coarse, best for "overview" queries)
  - One Document for statement-level metadata

Each Document carries structured metadata so ChromaDB can filter by
account_type, month, year, transaction_type, etc. before doing
similarity search.

Usage:
    from src.ingest.chunker import StatementChunker
    from src.ingest.parser import StatementParser

    result = StatementParser().parse("data/raw/stmt_20250131.pdf")
    docs = StatementChunker().chunk(result)
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from .parser import (
    StatementResult,
    SavingsTransaction,
    CheckingTransaction,
    LoanTransaction,
)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "JAN": 1, "FEB": 2,  "MAR": 3,  "APR": 4,
    "MAY": 5, "JUN": 6,  "JUL": 7,  "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _parse_period_date(period_date: str) -> tuple[int, int, int]:
    """
    '01-31-25' → (2025, 1, 31)
    '12-01-24' → (2024, 12, 1)
    """
    month, day, year_short = period_date.split("-")
    return 2000 + int(year_short), int(month), int(day)


def _normalise_date(raw_date: str, period_start: str, period_end: str) -> str:
    """
    Convert 'JAN14' to a full ISO date '2025-01-14'.

    Year is resolved by matching the transaction month against the
    period range — handles multi-year statement sets safely.

    Strategy:
      1. Parse period start and end into (year, month, day) tuples.
      2. If period is within one calendar year → use that year.
      3. If period crosses a year boundary (e.g. Dec 2024 → Jan 2025):
           - months >= period_start month → period_start year
           - months <  period_start month → period_end year
    """
    txn_month_num = _MONTH_MAP[raw_date[:3]]
    txn_day       = int(raw_date[3:])

    start_year, start_month, _ = _parse_period_date(period_start)
    end_year,   end_month,   _ = _parse_period_date(period_end)

    if start_year == end_year:
        year = start_year
    else:
        # Cross-year: months on or after start_month belong to start_year
        if txn_month_num >= start_month:
            year = start_year
        else:
            year = end_year

    return f"{year}-{txn_month_num:02d}-{txn_day:02d}"


def _period_to_month(period_start: str) -> str:
    """'01-01-25' → '2025-01'"""
    year, month, _ = _parse_period_date(period_start)
    return f"{year}-{month:02d}"


def _period_to_year(period_start: str) -> str:
    """'01-01-25' → '2025'"""
    year, _, _ = _parse_period_date(period_start)
    return str(year)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class StatementChunker:
    """
    Produces three tiers of Documents from a StatementResult:

    Tier 1 — transaction-level  (one doc per row)
    Tier 2 — section summary    (one doc per account section)
    Tier 3 — statement metadata (one doc for the whole statement)

    Tier 1 is the workhorse for specific queries.
    Tier 2 catches "how much did I spend overall in January?" questions.
    Tier 3 catches "what accounts do I have?" questions.
    """

    def chunk(self, result: StatementResult) -> list[Document]:
        docs: list[Document] = []

        month  = _period_to_month(result.period.start)
        year   = _period_to_year(result.period.start)
        period = str(result.period)

        base_meta = {
            "source_file":       Path(result.source_file).name,
            "statement_period":  period,
            "month":             month,
            "year":              year,
        }

        # ── Tier 1: transactions ──────────────────────────────────────────
        for txn in result.savings.transactions:
            docs.append(self._savings_txn_doc(txn, result, base_meta))

        for txn in result.checking.transactions:
            docs.append(self._checking_txn_doc(txn, result, base_meta))

        for txn in result.loan.transactions:
            docs.append(self._loan_txn_doc(txn, result, base_meta))

        # ── Tier 2: section summaries ─────────────────────────────────────
        docs.append(self._savings_summary_doc(result, base_meta))
        docs.append(self._checking_summary_doc(result, base_meta))
        docs.append(self._loan_summary_doc(result, base_meta))

        # ── Tier 3: statement overview ────────────────────────────────────
        docs.append(self._statement_overview_doc(result, base_meta))

        return docs

    # ── Tier 1: transaction documents ─────────────────────────────────────

    @staticmethod
    def _savings_txn_doc(
        txn: SavingsTransaction,
        result: StatementResult,
        base: dict,
    ) -> Document:
        date_iso  = _normalise_date(txn.date, result.period.start, result.period.end)
        direction = "credit" if txn.amount >= 0 else "debit"

        page_content = (
            f"Savings account transaction on {date_iso}: "
            f"{txn.description}. "
            f"Amount: ${txn.amount:,.2f} ({direction}). "
            f"Running balance: ${txn.balance:,.2f}."
        )

        metadata = {
            **base,
            "account_type":    "savings",
            "account_number":  result.savings.account_number,
            "date":            date_iso,
            "description":     txn.description,
            "amount":          txn.amount,
            "balance":         txn.balance,
            "transaction_type": direction,
            "chunk_type":      "transaction",
        }

        return Document(page_content=page_content, metadata=metadata)

    @staticmethod
    def _checking_txn_doc(
        txn: CheckingTransaction,
        result: StatementResult,
        base: dict,
    ) -> Document:
        date_iso  = _normalise_date(txn.date, result.period.start, result.period.end)
        direction = "credit" if txn.amount >= 0 else "debit"
        category  = StatementChunker._classify_checking(txn.description, txn.amount)

        page_content = (
            f"Checking account transaction on {date_iso}: "
            f"{txn.description}. "
            f"Amount: ${txn.amount:,.2f} ({direction}). "
            f"Running balance: ${txn.balance:,.2f}. "
            f"Category: {category}."
        )

        metadata = {
            **base,
            "account_type":    "checking",
            "account_number":  result.checking.account_number,
            "date":            date_iso,
            "description":     txn.description,
            "amount":          txn.amount,
            "balance":         txn.balance,
            "transaction_type": direction,
            "category":        category,
            "chunk_type":      "transaction",
        }

        return Document(page_content=page_content, metadata=metadata)

    @staticmethod
    def _loan_txn_doc(
        txn: LoanTransaction,
        result: StatementResult,
        base: dict,
    ) -> Document:
        date_iso = _normalise_date(txn.date, result.period.start, result.period.end)

        page_content = (
            f"Vehicle loan payment on {date_iso}: "
            f"{txn.description}. "
            f"Payment amount: ${txn.payment:,.2f}. "
            f"Principal reduction: ${abs(txn.principal):,.2f}. "
            f"Remaining loan balance: ${txn.balance:,.2f}."
        )

        metadata = {
            **base,
            "account_type":    "loan",
            "loan_number":     result.loan.loan_number,
            "date":            date_iso,
            "description":     txn.description,
            "payment":         txn.payment,
            "principal":       txn.principal,
            "balance":         txn.balance,
            "transaction_type": "loan_payment",
            "chunk_type":      "transaction",
        }

        return Document(page_content=page_content, metadata=metadata)

    # ── Tier 2: section summary documents ─────────────────────────────────

    @staticmethod
    def _savings_summary_doc(
        result: StatementResult, base: dict
    ) -> Document:
        s          = result.savings
        txn_count  = len(s.transactions)
        total_credits = sum(t.amount for t in s.transactions if t.amount > 0)
        total_debits  = sum(t.amount for t in s.transactions if t.amount < 0)

        page_content = (
            f"Savings account summary for {base['month']} "
            f"(account #{s.account_number}). "
            f"Opening balance: ${s.previous_balance:,.2f}. "
            f"Closing balance: ${s.new_balance:,.2f}. "
            f"Total credits: ${total_credits:,.2f}. "
            f"Total debits: ${total_debits:,.2f}. "
            f"Number of transactions: {txn_count}."
        )

        metadata = {
            **base,
            "account_type":      "savings",
            "account_number":    s.account_number,
            "opening_balance":   s.previous_balance,
            "closing_balance":   s.new_balance,
            "total_credits":     total_credits,
            "total_debits":      total_debits,
            "transaction_count": txn_count,
            "chunk_type":        "section_summary",
        }

        return Document(page_content=page_content, metadata=metadata)

    @staticmethod
    def _checking_summary_doc(
        result: StatementResult, base: dict
    ) -> Document:
        c             = result.checking
        total_credits = sum(t.amount for t in c.transactions if t.amount > 0)
        total_debits  = sum(t.amount for t in c.transactions if t.amount < 0)
        txn_count     = len(c.transactions)
        payroll       = sum(
            t.amount for t in c.transactions
            if "PAYROLL" in t.description.upper()
        )
        loan_payments = abs(sum(
            t.amount for t in c.transactions
            if "TRANSFER FROM/TO 142" in t.description
        ))

        page_content = (
            f"Checking account summary for {base['month']} "
            f"(account #{c.account_number}). "
            f"Opening balance: ${c.previous_balance:,.2f}. "
            f"Closing balance: ${c.new_balance:,.2f}. "
            f"Total money in: ${total_credits:,.2f}. "
            f"Total money out: ${abs(total_debits):,.2f}. "
            f"Number of transactions: {txn_count}. "
            f"Payroll deposits: ${payroll:,.2f}. "
            f"Loan payments made: ${loan_payments:,.2f}."
        )

        metadata = {
            **base,
            "account_type":      "checking",
            "account_number":    c.account_number,
            "opening_balance":   c.previous_balance,
            "closing_balance":   c.new_balance,
            "total_credits":     total_credits,
            "total_debits":      total_debits,
            "transaction_count": txn_count,
            "chunk_type":        "section_summary",
        }

        return Document(page_content=page_content, metadata=metadata)

    @staticmethod
    def _loan_summary_doc(
        result: StatementResult, base: dict
    ) -> Document:
        l               = result.loan
        total_paid      = sum(t.payment for t in l.transactions)
        total_principal = sum(abs(t.principal) for t in l.transactions)

        page_content = (
            f"Vehicle loan summary for {base['month']} "
            f"(loan #{l.loan_number}). "
            f"APR: {l.apr}%. "
            f"Opening balance: ${l.previous_balance:,.2f}. "
            f"Closing balance: ${l.new_balance:,.2f}. "
            f"Total payments made: ${total_paid:,.2f}. "
            f"Total principal paid: ${total_principal:,.2f}. "
            f"Total interest charged: ${l.total_interest:,.2f}. "
            f"Next payment due: ${l.payment_due:,.2f}."
        )

        metadata = {
            **base,
            "account_type":    "loan",
            "loan_number":     l.loan_number,
            "apr":             l.apr,
            "opening_balance": l.previous_balance,
            "closing_balance": l.new_balance,
            "total_interest":  l.total_interest,
            "payment_due":     l.payment_due,
            "chunk_type":      "section_summary",
        }

        return Document(page_content=page_content, metadata=metadata)

    # ── Tier 3: statement overview document ───────────────────────────────

    @staticmethod
    def _statement_overview_doc(
        result: StatementResult, base: dict
    ) -> Document:
        page_content = (
            f"DCU bank statement overview for {base['month']}. "
            f"Statement period: {result.period}. "
            f"Accounts held: "
            f"Primary Savings (#{result.savings.account_number}) "
            f"closing balance ${result.savings.new_balance:,.2f}, "
            f"Free Checking (#{result.checking.account_number}) "
            f"closing balance ${result.checking.new_balance:,.2f}, "
            f"New Vehicle Loan (#{result.loan.loan_number}) "
            f"closing balance ${result.loan.new_balance:,.2f} "
            f"at {result.loan.apr}% APR."
        )

        metadata = {
            **base,
            "savings_balance":  result.savings.new_balance,
            "checking_balance": result.checking.new_balance,
            "loan_balance":     result.loan.new_balance,
            "chunk_type":       "statement_overview",
        }

        return Document(page_content=page_content, metadata=metadata)

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_checking(description: str, amount: float) -> str:
        """
        Rule-based category tagging for checking transactions.
        Extend this as you see more statement patterns.
        """
        desc = description.upper()
        if "PAYROLL" in desc or "DIRECT DEP" in desc:
            return "payroll"
        if "TRANSFER FROM/TO" in desc or "SHR TRANSFER" in desc:
            return "internal_transfer"
        if "EXT ACCOUNT TRF" in desc or "EXTERNAL" in desc:
            return "external_transfer"
        if "WITHDRAWAL" in desc and amount < 0:
            return "withdrawal"
        if amount > 0:
            return "credit"
        return "debit"