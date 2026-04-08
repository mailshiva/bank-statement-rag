"""
explore_structure.py

Deeper structural analysis of a DCU statement.
Run: python explore_structure.py path/to/statement.pdf
"""

import re
import sys
import pdfplumber
from dataclasses import dataclass, field
from typing import Optional

# ── Section boundary patterns ──────────────────────────────────────────────
SECTION_PATTERNS = {
    "savings":  re.compile(r"PRIMARY SAVINGS ACCT#\s*\d+"),
    "checking": re.compile(r"FREE CHECKING ACCT#\s*\d+"),
    "loan":     re.compile(r"NEW VEHICLE LOAN#\s*(\d+)"),
    "summary":  re.compile(r"STATEMENT SUMMARY"),
    "discard":  re.compile(r"Direct general inquiries|BILLING RIGHTS|Rev:\s*\d"),
}

# ── Transaction patterns per section ───────────────────────────────────────
TXN_SAVINGS = re.compile(
    r"^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}"
    r"\s+(.+?)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*$"
)

TXN_CHECKING = re.compile(
    r"^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}"
    r"\s+(\w+)\s+(.+?)\s+(-?[\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*$"
)

TXN_LOAN = re.compile(
    r"^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}"
    r"\s+(.+?)\s+([\d,]+\.\d{2})\s+(-?[\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*$"
)

STATEMENT_PERIOD = re.compile(r"(\d{2}-\d{2}-\d{2})\s+to\s+(\d{2}-\d{2}-\d{2})")


@dataclass
class Section:
    name: str
    raw_lines: list[str] = field(default_factory=list)
    transactions: list[dict] = field(default_factory=list)


def extract_full_text(path: str) -> list[str]:
    """Extract all text, page by page, skipping the boilerplate page."""
    lines = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            # Skip page 2 — it's purely legal boilerplate
            if "BILLING RIGHTS" in text or "Direct general inquiries" in text:
                print(f"  [Skipping page {i+1} — boilerplate]")
                continue
            lines.extend(text.splitlines())
    return lines


def split_into_sections(lines: list[str]) -> dict[str, Section]:
    """Walk lines top-to-bottom, routing each line into its section."""
    sections = {
        "savings":  Section("PRIMARY SAVINGS"),
        "checking": Section("FREE CHECKING"),
        "loan":     Section("NEW VEHICLE LOAN"),
        "summary":  Section("STATEMENT SUMMARY"),
    }
    current = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect section boundaries
        if SECTION_PATTERNS["discard"].search(stripped):
            current = None
            continue
        if SECTION_PATTERNS["savings"].search(stripped):
            current = "savings"
            continue
        if SECTION_PATTERNS["checking"].search(stripped):
            current = "checking"
            continue
        if SECTION_PATTERNS["loan"].search(stripped):
            current = "loan"
            continue
        if SECTION_PATTERNS["summary"].search(stripped):
            current = "summary"
            continue

        if current:
            sections[current].raw_lines.append(stripped)

    return sections


def parse_checking_transactions(lines: list[str]) -> list[dict]:
    """
    Checking transactions can span 2 lines:
      JAN06  WITHDRAWAL  -2,000.00  10,166.17
             EXT ACCOUNT TRF Digital Federal Credit U yZaDcpcfqMq0
    We merge continuation lines into the previous transaction's description.
    """
    DATE_PREFIX = re.compile(
        r"^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}\s+"
    )
    AMOUNT_LINE = re.compile(r"(-?[\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*$")

    transactions = []
    pending = None

    for line in lines:
        # Stop at summary sub-sections
        if any(kw in line for kw in [
            "DEPOSITS, DIVIDENDS", "WITHDRAWALS, FEES",
            "TOTAL DIVIDENDS", "TOTAL DEPOSITS", "TOTAL FEES",
            "TOTAL WITHDRAWALS", "NEW BALANCE", "PREVIOUS BALANCE",
        ]):
            if pending:
                transactions.append(pending)
                pending = None
            continue

        if DATE_PREFIX.match(line):
            if pending:
                transactions.append(pending)

            parts = line.split()
            date = parts[0]
            # Find amounts at end: last two numeric tokens
            m = AMOUNT_LINE.search(line)
            if m:
                amount = m.group(1).replace(",", "")
                balance = m.group(2).replace(",", "")
                # Description is everything between date and amounts
                desc_end = line.rfind(m.group(1))
                description = line[len(date):desc_end].strip()
                pending = {
                    "date": date,
                    "description": description,
                    "amount": float(amount),
                    "balance": float(balance),
                    "continuation": [],
                }
            else:
                pending = None
        else:
            # Continuation line — append to previous transaction's description
            if pending:
                pending["continuation"].append(line)

    if pending:
        transactions.append(pending)

    # Merge continuation into description
    for txn in transactions:
        if txn["continuation"]:
            txn["description"] += " | " + " ".join(txn["continuation"])
        del txn["continuation"]

    return transactions


def parse_savings_transactions(lines: list[str]) -> list[dict]:
    DATE_PREFIX = re.compile(
        r"^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}\s+"
    )
    STOP_KEYWORDS = ["NEW BALANCE", "PREVIOUS BALANCE", "ANNUAL PERCENTAGE YIELD"]
    transactions = []

    for line in lines:
        if any(kw in line for kw in STOP_KEYWORDS):
            continue
        if DATE_PREFIX.match(line):
            parts = line.split()
            date = parts[0]
            nums = re.findall(r"-?[\d,]+\.\d{2}", line)
            if len(nums) >= 2:
                transactions.append({
                    "date": date,
                    "description": " ".join(parts[1:len(parts) - 2]),
                    "amount": float(nums[-2].replace(",", "")),
                    "balance": float(nums[-1].replace(",", "")),
                })

    return transactions


def parse_loan_transactions(lines: list[str]) -> list[dict]:
    DATE_PREFIX = re.compile(
        r"^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}\s+"
    )
    STOP_KEYWORDS = [
        "INTEREST RATE DETAIL", "FEES CHARGED", "INTEREST CHARGED",
        "TOTALS YEAR", "EFFECTIVE DATES", "TRANSACTIONS",
        "TOTAL FEES", "TOTAL INTEREST",
    ]
    transactions = []

    for line in lines:
        if any(kw in line for kw in STOP_KEYWORDS):
            continue
        if DATE_PREFIX.match(line):
            nums = re.findall(r"-?[\d,]+\.\d{2}", line)
            parts = line.split()
            date = parts[0]
            if len(nums) >= 3:
                transactions.append({
                    "date": date,
                    "description": " ".join(parts[1:len(parts) - 3]),
                    "payment": float(nums[-3].replace(",", "")),
                    "principal": float(nums[-2].replace(",", "")),
                    "balance": float(nums[-1].replace(",", "")),
                })

    return transactions


def analyze(path: str):
    print(f"\nAnalyzing: {path}\n")

    lines = extract_full_text(path)
    print(f"Total lines extracted (excl. boilerplate): {len(lines)}\n")

    # Detect statement period
    full_text = "\n".join(lines)
    period_match = STATEMENT_PERIOD.search(full_text)
    if period_match:
        print(f"Statement period : {period_match.group(1)}  →  {period_match.group(2)}")

    sections = split_into_sections(lines)

    print("\n── SAVINGS ─────────────────────────────────────────")
    txns = parse_savings_transactions(sections["savings"].raw_lines)
    for t in txns:
        print(f"  {t}")

    print("\n── CHECKING ────────────────────────────────────────")
    txns = parse_checking_transactions(sections["checking"].raw_lines)
    for t in txns:
        print(f"  {t}")

    print("\n── LOAN ────────────────────────────────────────────")
    txns = parse_loan_transactions(sections["loan"].raw_lines)
    for t in txns:
        print(f"  {t}")

    print("\n── SUMMARY LINES ───────────────────────────────────")
    for line in sections["summary"].raw_lines:
        print(f"  {line}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore_structure.py <path_to_pdf>")
        sys.exit(1)
    analyze(sys.argv[1])