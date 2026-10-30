from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "leukemia_6mp.csv"


@pytest.fixture(scope="session")
def leukemia_path() -> str:
    return str(DATA)


@pytest.fixture(scope="session")
def leukemia() -> pd.DataFrame:
    return pd.read_csv(DATA)
