"""Quick smoke test — run from project root."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.ingest.parser import StatementParser

parser = StatementParser()
result = parser.parse("data/raw/stmt_20250131.pdf")

print(f"Period   : {result.period}")
print(f"Source   : {result.source_file}")

print(f"\n── Savings (acct #{result.savings.account_number})")
print(f"   Prev balance : {result.savings.previous_balance}")
print(f"   New balance  : {result.savings.new_balance}")
for t in result.savings.transactions:
    print(f"   {t}")

print(f"\n── Checking (acct #{result.checking.account_number})")
print(f"   Prev balance : {result.checking.previous_balance}")
print(f"   New balance  : {result.checking.new_balance}")
for t in result.checking.transactions:
    print(f"   {t}")

print(f"\n── Loan #{result.loan.loan_number}  APR: {result.loan.apr}%")
print(f"   Prev balance : {result.loan.previous_balance}")
print(f"   New balance  : {result.loan.new_balance}")
print(f"   Payment due  : {result.loan.payment_due}")
print(f"   Interest/period: {result.loan.total_interest}")
for t in result.loan.transactions:
    print(f"   {t}")

print(f"\n── Summary lines")
for line in result.summary_lines:
    print(f"   {line}")