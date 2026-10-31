"""Agent-facing tool layer.

Datasets are registered once with ``load_dataset`` and then referred to by id, so
raw data does not have to pass through the language model. Every result carries
a provenance record (dataset hash, function, inputs, package version, time) so
any number an agent reports can be traced back and reproduced.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .cox import cox_regression
from .design import events_required, power_from_events, sample_size, simulate_power
from .km import kaplan_meier
from .logrank import logrank_test

_DATASETS: dict[str, dict[str, Any]] = {}
_INDEX = None
DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "index"


def _provenance(function: str, inputs: dict, dataset_id: str | None = None) -> dict:
    prov = {
        "function": function,
        "inputs": inputs,
        "package": f"survagent {__version__}",
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if dataset_id is not None:
        ds = _DATASETS[dataset_id]
        prov["dataset"] = {"id": dataset_id, "path": ds["path"], "sha256": ds["sha256"]}
    return prov


def load_dataset(path: str) -> dict:
    """Register a CSV file and return its id, hash and a column summary."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"File not found: {p}")
    raw = p.read_bytes()
    df = pd.read_csv(p)
    dataset_id = uuid.uuid4().hex[:8]
    sha = hashlib.sha256(raw).hexdigest()
    _DATASETS[dataset_id] = {"df": df, "path": str(p), "sha256": sha}
    columns = [
        {
            "name": c,
            "dtype": str(df[c].dtype),
            "n_missing": int(df[c].isna().sum()),
            "n_unique": int(df[c].nunique()),
            "examples": [_jsonable(v) for v in df[c].dropna().unique()[:5]],
        }
        for c in df.columns
    ]
    return {"dataset_id": dataset_id, "path": str(p), "sha256": sha, "n_rows": len(df), "columns": columns}


def _subset(dataset_id: str, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if dataset_id not in _DATASETS:
        raise ValueError(f"Unknown dataset_id `{dataset_id}`. Call load_dataset first.")
    df = _DATASETS[dataset_id]["df"]
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Column(s) not in dataset: {missing_cols}. Available: {list(df.columns)}")
    sub = df[cols].dropna()
    notes = []
    dropped = len(df) - len(sub)
    if dropped:
        notes.append(
            f"Dropped {dropped} of {len(df)} rows with missing values in {cols} (complete-case analysis); "
            "results may be biased if data are not missing completely at random."
        )
    return sub, notes


def km_tool(dataset_id: str, time_col: str, event_col: str, group_col: str | None = None,
            conf_level: float = 0.95) -> dict:
    cols = [time_col, event_col] + ([group_col] if group_col else [])
    sub, notes = _subset(dataset_id, cols)
    if group_col:
        result = {
            str(g): kaplan_meier(d[time_col], d[event_col], conf_level)
            for g, d in sub.groupby(group_col)
        }
    else:
        result = {"all": kaplan_meier(sub[time_col], sub[event_col], conf_level)}
    return {"result": result, "data_notes": notes,
            "provenance": _provenance("kaplan_meier", {"time_col": time_col, "event_col": event_col,
                                      "group_col": group_col, "conf_level": conf_level}, dataset_id)}


def logrank_tool(dataset_id: str, time_col: str, event_col: str, group_col: str) -> dict:
    sub, notes = _subset(dataset_id, [time_col, event_col, group_col])
    res = logrank_test(sub[time_col], sub[event_col], sub[group_col])
    return {"result": res, "data_notes": notes,
            "provenance": _provenance("logrank_test", {"time_col": time_col, "event_col": event_col,
                                      "group_col": group_col}, dataset_id)}


def cox_tool(dataset_id: str, time_col: str, event_col: str, covariates: list[str],
             reference_levels: dict[str, str] | None = None, ties: str = "efron",
             conf_level: float = 0.95) -> dict:
    sub, notes = _subset(dataset_id, [time_col, event_col, *covariates])
    res = cox_regression(
        sub[time_col], sub[event_col], {c: sub[c].tolist() for c in covariates},
        reference_levels=reference_levels, ties=ties, conf_level=conf_level,
    )
    return {"result": res, "data_notes": notes,
            "provenance": _provenance("cox_regression", {"time_col": time_col, "event_col": event_col,
                                      "covariates": covariates, "reference_levels": reference_levels,
                                      "ties": ties, "conf_level": conf_level}, dataset_id)}


def events_required_tool(**kw) -> dict:
    return {"result": events_required(**kw), "provenance": _provenance("events_required", kw)}


def power_tool(**kw) -> dict:
    return {"result": power_from_events(**kw), "provenance": _provenance("power_from_events", kw)}


def sample_size_tool(**kw) -> dict:
    return {"result": sample_size(**kw), "provenance": _provenance("sample_size", kw)}


def simulate_power_tool(**kw) -> dict:
    return {"result": simulate_power(**kw), "provenance": _provenance("simulate_power", kw)}


def search_guidance_tool(query: str, k: int = 5, method: str | None = None) -> dict:
    """Search the guidance/literature index (loaded once, on first use)."""
    global _INDEX
    if _INDEX is None:
        from .rag.index import GuidanceIndex

        _INDEX = GuidanceIndex(os.environ.get("SURVAGENT_INDEX", DEFAULT_INDEX_DIR))
    res = _INDEX.search(query, k, method)
    return {"result": res, "provenance": _provenance(
        "search_guidance", {"query": query, "k": k, "method": res["method"],
                            "corpus_sha256": res["corpus_sha256"]})}


def _jsonable(v):
    try:
        return v.item()
    except AttributeError:
        return v
