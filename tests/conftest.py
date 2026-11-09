import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.analysis import run_pipeline  # noqa: E402


@pytest.fixture(scope="session")
def results():
    if not config.RAW_CSV.exists():
        from src import generate_data
        generate_data.main()
    return run_pipeline()
