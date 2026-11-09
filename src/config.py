"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_PNG = ROOT / "outputs" / "png"
OUT_PDF = ROOT / "outputs" / "pdf"
DOCS_DIR = ROOT / "docs"

RAW_CSV = DATA_DIR / "transactions_raw.csv"
TRUTH_CSV = DATA_DIR / "transactions_clean_truth.csv"
PLANTED_JSON = DATA_DIR / "planted_anomalies.json"

SEED = 42
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

# Fixed, scheduled outflows. They are known in advance, so they are left out of
# the operating-spend series used for cluster detection.
SCHEDULED_CATEGORIES = ("Payroll", "Rent")

# A fictional company, used so the sample reads like a real report.
COMPANY = "Calder Ashby Engineering"
COMPANY_SHORT = "Calder Ashby"
PERIOD_LABEL = "January-December 2025"
DISCLAIMER = (f"{COMPANY} is a fictional company. Every transaction, name and figure in this sample is "
              "synthetic and for illustration only.")

GITHUB_USER = "veridius-code"
REPO_NAME = "transaction-viz-showcase"
REPO_URL = f"https://github.com/{GITHUB_USER}/{REPO_NAME}"
PAGES_URL = f"https://{GITHUB_USER}.github.io/{REPO_NAME}/"
AUTHOR_LINE = "Raul Ungureanu, Data Scientist"
