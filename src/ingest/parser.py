"""
src/ingest/parser.py

Parses DCU bank statement PDFs into structured section data.

Handles three account types:
  - Primary Savings
  - Free Checking
  - New Vehicle Loan

Usage:
    from src.ingest.parser import StatementParser

    parser = StatementParser()
    result = parser.parse("data/raw/stmt_20250131.pdf")

    print(result.period)           # StatementPeriod(start='01-01-25', end='01-31-25')
    print(result.checking.transactions)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_MONTH = r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}"
_DATE = re.compile(rf"^{_MONTH}\s+")
_AMT = r"-?[\d,]+\.\d{2}"

SECTION = {
    "savings": re.compile(r"PRIMARY SAVINGS ACCT#\s*\d+"),
    "checking": re.compile(r"FREE CHECKING ACCT#\s*\d+"),
    "loan": re.compile(r"NEW VEHICLE LOAN#\s*(\d+)"),
    "summary": re.compile(r"S\s*T\s*A\s*T\s*E\s*M\s*E\s*N\s*T\s+S\s*U\s*M\s*M\s*A\s*R\s*Y"),
    "boilerplate": re.compile(r"BILLING RIGHTS|Direct general inquiries|Rev:\s*\d"),
}

PERIOD = re.compile(r"(\d{2}-\d{2}-\d{2})\s+to\s+(\d{2}-\d{2}-\d{2})")

# Lines that mark end of transaction block within a section
STOP_SAVINGS = {
    "NEW BALANCE", "PREVIOUS BALANCE", "ANNUAL PERCENTAGE YIELD"
}
STOP_CHECKING = {
    "DEPOSITS, DIVIDENDS", "WITHDRAWALS, FEES",
    "TOTAL DIVIDENDS", "TOTAL DEPOSITS", "TOTAL FEES",
    "TOTAL WITHDRAWALS", "NEW BALANCE", "PREVIOUS BALANCE",
    "DATE AMOUNT",
    "DATE TRANSACTION",   # column header row
}
STOP_LOAN = {
    "INTEREST RATE DETAIL", "FEES CHARGED", "INTEREST CHARGED",
    "TOTALS YEAR", "EFFECTIVE DATES", "TRANSACTIONS",
    "TOTAL FEES", "TOTAL INTEREST",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StatementPeriod:
    start: str  # e.g. "01-01-25"
    end: str  # e.g. "01-31-25"

    def __str__(self) -> str:
        return f"{self.start} to {self.end}"


@dataclass
class SavingsTransaction:
    date: str
    description: str
    amount: float
    balance: float


@dataclass
class CheckingTransaction:
    date: str
    description: str  # may include continuation lines joined with " | "
    amount: float
    balance: float


@dataclass
class LoanTransaction:
    date: str
    description: str
    payment: float
    principal: float  # negative = reduction in principal
    balance: float


@dataclass
class SavingsSection:
    account_number: str = ""
    previous_balance: Optional[float] = None
    new_balance: Optional[float] = None
    transactions: list[SavingsTransaction] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)


@dataclass
class CheckingSection:
    account_number: str = ""
    previous_balance: Optional[float] = None
    new_balance: Optional[float] = None
    transactions: list[CheckingTransaction] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)


@dataclass
class LoanSection:
    loan_number: str = ""
    apr: Optional[float] = None
    previous_balance: Optional[float] = None
    new_balance: Optional[float] = None
    payment_due: Optional[float] = None
    total_interest: Optional[float] = None
    transactions: list[LoanTransaction] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)


@dataclass
class StatementResult:
    source_file: str
    period: Optional[StatementPeriod]
    savings: SavingsSection
    checking: CheckingSection
    loan: LoanSection
    summary_lines: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class StatementParser:
    """
    Parses a single DCU statement PDF into a StatementResult.

    Each public method is independently testable:
        - extract_lines()
        - split_sections()
        - parse_savings()
        - parse_checking()
        - parse_loan()
    """

    # ── Public entry point ─────────────────────────────────────────────────

    def parse(self, path: str | Path) -> StatementResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        lines = self.extract_lines(path)
        period = self._detect_period(lines)
        raw_sections = self.split_sections(lines)

        savings = self.parse_savings(raw_sections["savings"])
        checking = self.parse_checking(raw_sections["checking"])
        loan = self.parse_loan(raw_sections["loan"])

        # Pull structured metadata from raw section headers
        savings.account_number = self._extract_acct_number(lines, "savings")
        checking.account_number = self._extract_acct_number(lines, "checking")
        loan.loan_number = self._extract_loan_number(lines)
        loan.apr = self._extract_apr(lines)

        self._extract_balances(raw_sections["savings"], savings)
        self._extract_balances(raw_sections["checking"], checking)
        self._extract_loan_metadata(raw_sections["loan"], loan)

        return StatementResult(
            source_file=str(path),
            period=period,
            savings=savings,
            checking=checking,
            loan=loan,
            summary_lines=raw_sections["summary"],
        )

    # ── Text extraction ────────────────────────────────────────────────────

    def extract_lines(self, path: Path) -> list[str]:
        """Extract all text lines, skipping boilerplate pages."""
        lines = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if SECTION["boilerplate"].search(text):
                    continue
                lines.extend(text.splitlines())
        return lines

    # ── Section splitting ──────────────────────────────────────────────────

    def split_sections(self, lines: list[str]) -> dict[str, list[str]]:
        """
        Walk lines top-to-bottom, routing each into its named bucket.
        Returns a dict with keys: savings, checking, loan, summary.
        """
        buckets: dict[str, list[str]] = {
            "savings": [], "checking": [], "loan": [], "summary": []
        }
        current: Optional[str] = None

        for line in lines:
            s = line.strip()
            if not s:
                continue

            if SECTION["boilerplate"].search(s):
                current = None
                continue
            if SECTION["summary"].search(s):
                current = "summary"
                continue
            if SECTION["savings"].search(s):
                current = "savings"
                continue
            if SECTION["checking"].search(s):
                current = "checking"
                continue
            if SECTION["loan"].search(s):
                current = "loan"
                # Previous balance lives on this same header line
                buckets["loan"].append(s)
                continue

            if current:
                buckets[current].append(s)

        return buckets

    # ── Section parsers ────────────────────────────────────────────────────

    def parse_savings(self, lines: list[str]) -> SavingsSection:
        section = SavingsSection(raw_lines=lines)

        for line in lines:
            if any(kw in line for kw in STOP_SAVINGS):
                continue
            if not _DATE.match(line):
                continue

            nums = re.findall(_AMT, line)
            parts = line.split()
            if len(nums) >= 2:
                date = parts[0]
                description = self._extract_description(line, date, nums)
                section.transactions.append(SavingsTransaction(
                    date=date,
                    description=description,
                    amount=float(nums[-2].replace(",", "")),
                    balance=float(nums[-1].replace(",", "")),
                ))

        return section

    def parse_checking(self, lines: list[str]) -> CheckingSection:
        """
        Handles multi-line transactions where the description
        continues on the next line(s) with no date prefix.
        """
        section = CheckingSection(raw_lines=lines)
        pending: Optional[dict] = None

        for line in lines:
            if any(kw in line for kw in STOP_CHECKING):
                if pending:
                    section.transactions.append(
                        self._finalise_checking(pending)
                    )
                    pending = None
                continue

            if _DATE.match(line):
                # Reject inline summary rows: "JAN03 2,644.58 JAN17 2,692.55"
                # Real transactions have exactly one month abbreviation
                month_hits = re.findall(
                    r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}\b",
                    line
                )
                if len(month_hits) > 1:
                    continue

                if pending:
                    section.transactions.append(
                        self._finalise_checking(pending)
                    )
                nums = re.findall(_AMT, line)
                parts = line.split()
                date = parts[0]
                if len(nums) >= 2:
                    description = self._extract_description(line, date, nums)
                    pending = {
                        "date": date,
                        "description": description,
                        "amount": float(nums[-2].replace(",", "")),
                        "balance": float(nums[-1].replace(",", "")),
                        "continuation": [],
                    }
                else:
                    pending = None
            else:
                # Continuation line — belongs to previous transaction
                if pending:
                    pending["continuation"].append(line)

        if pending:
            section.transactions.append(self._finalise_checking(pending))

        return section

    def parse_loan(self, lines: list[str]) -> LoanSection:
        section = LoanSection(raw_lines=lines)

        for line in lines:
            if any(kw in line for kw in STOP_LOAN):
                continue
            if not _DATE.match(line):
                continue

            nums = re.findall(_AMT, line)
            parts = line.split()
            date = parts[0]

            # Loan rows have 3 amounts: payment, principal, balance
            if len(nums) >= 3:
                description = self._extract_description(line, date, nums, n_amounts=3)
                section.transactions.append(LoanTransaction(
                    date=date,
                    description=description,
                    payment=float(nums[-3].replace(",", "")),
                    principal=float(nums[-2].replace(",", "")),
                    balance=float(nums[-1].replace(",", "")),
                ))

        return section

    # ── Metadata extractors ────────────────────────────────────────────────

    def _detect_period(self, lines: list[str]) -> Optional[StatementPeriod]:
        for line in lines:
            m = PERIOD.search(line)
            if m:
                return StatementPeriod(start=m.group(1), end=m.group(2))
        return None

    def _extract_acct_number(self, lines: list[str], section: str) -> str:
        pattern = SECTION[section]
        for line in lines:
            m = pattern.search(line)
            if m:
                nums = re.findall(r"\d+", line)
                return nums[-1] if nums else ""
        return ""

    def _extract_loan_number(self, lines: list[str]) -> str:
        for line in lines:
            m = SECTION["loan"].search(line)
            if m:
                nums = re.findall(r"\d+", line)
                return nums[0] if nums else ""
        return ""

    def _extract_apr(self, lines: list[str]) -> Optional[float]:
        apr_pat = re.compile(r"ANNUAL PERCENTAGE RATE.*?([\d.]+)%")
        for line in lines:
            m = apr_pat.search(line)
            if m:
                return float(m.group(1))
        return None

    def _extract_balances(
            self,
            lines: list[str],
            section: SavingsSection | CheckingSection,
    ) -> None:
        for line in lines:
            if "PREVIOUS BALANCE" in line:
                nums = re.findall(_AMT, line)
                if nums:
                    section.previous_balance = float(nums[-1].replace(",", ""))
            if "NEW BALANCE" in line:
                nums = re.findall(_AMT, line)
                if nums:
                    section.new_balance = float(nums[-1].replace(",", ""))

    def _extract_loan_metadata(
            self, lines: list[str], loan: LoanSection
    ) -> None:
        for line in lines:
            if "PREVIOUS BALANCE" in line:
                nums = re.findall(_AMT, line)
                if nums:
                    loan.previous_balance = float(nums[-1].replace(",", ""))
            if "NEW BALANCE" in line:
                nums = re.findall(_AMT, line)
                if nums:
                    loan.new_balance = float(nums[-1].replace(",", ""))
            if "PAYMENT DUE:" in line:
                nums = re.findall(_AMT, line)
                if nums:
                    loan.payment_due = float(nums[-1].replace(",", ""))
            if "TOTAL INTEREST FOR THIS PERIOD" in line:
                nums = re.findall(_AMT, line)
                if nums:
                    loan.total_interest = float(nums[-1].replace(",", ""))

    # ── Helpers ────────────────────────────────────────────────────────────

    def _extract_description(
            self,
            line: str,
            date: str,
            nums: list[str],
            n_amounts: int = 2,
    ) -> str:
        """
        Extract description text between the date and the trailing amounts.
        Works by finding where the last N amounts begin and slicing.
        """
        # Find the start position of the first of the trailing amounts
        last_amounts = nums[-n_amounts:]
        search_from = line.index(date) + len(date)
        cutoff = len(line)

        for amt in last_amounts:
            idx = line.rfind(amt)
            if idx > search_from:
                cutoff = min(cutoff, idx)

        return line[search_from:cutoff].strip()

    def _finalise_checking(self, pending: dict) -> CheckingTransaction:
        description = pending["description"]
        if pending["continuation"]:
            description += " | " + " ".join(pending["continuation"])
        return CheckingTransaction(
            date=pending["date"],
            description=description,
            amount=pending["amount"],
            balance=pending["balance"],
        )