"""Tool layer (provenance, missing data, errors) and the MCP server over stdio."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from survagent import tools


def test_load_and_analyse_with_provenance(leukemia_path):
    ds = tools.load_dataset(leukemia_path)
    assert ds["n_rows"] == 42 and len(ds["sha256"]) == 64
    out = tools.logrank_tool(ds["dataset_id"], "time", "event", "treatment")
    assert out["provenance"]["dataset"]["sha256"] == ds["sha256"]
    assert out["result"]["p_value"] < 0.001


def test_missing_rows_are_reported(tmp_path, leukemia):
    df = leukemia.copy()
    df.loc[0, "time"] = None
    p = tmp_path / "with_missing.csv"
    df.to_csv(p, index=False)
    ds = tools.load_dataset(str(p))
    out = tools.cox_tool(ds["dataset_id"], "time", "event", ["treatment"])
    assert out["result"]["n"] == 41
    assert any("Dropped 1 of 42" in n for n in out["data_notes"])


def test_unknown_column_message_lists_columns(leukemia_path):
    ds = tools.load_dataset(leukemia_path)
    with pytest.raises(ValueError, match="Available"):
        tools.km_tool(ds["dataset_id"], "days", "event")


mcp = pytest.importorskip("mcp")


def _tiny_index(tmp_path):
    from survagent.rag.index import build_index

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "pubmed.jsonl").write_text(json.dumps(
        {"pmid": "42", "title": "Restricted mean survival time",
         "abstract": "RMST is an alternative to the hazard ratio when hazards are not proportional.",
         "journal": "Stat Med", "year": "2020"}) + "\n")
    (corpus / "sources.json").write_text(json.dumps([{"kind": "pubmed", "file": "pubmed.jsonl"}]))
    build_index(corpus, tmp_path / "index", dense=False)
    return str(tmp_path / "index")


def test_mcp_server_end_to_end(leukemia_path, tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    index_dir = _tiny_index(tmp_path)

    async def run():
        src = str(Path(__file__).resolve().parents[1] / "src")
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "survagent.mcp_server"], env={"PYTHONPATH": src, "SURVAGENT_INDEX": index_dir}
        )
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                names = {t.name for t in (await session.list_tools()).tools}
                ds = await session.call_tool("load_dataset", {"path": leukemia_path})
                ds_id = json.loads(ds.content[0].text)["dataset_id"]
                res = await session.call_tool(
                    "cox_regression",
                    {"dataset_id": ds_id, "time_col": "time", "event_col": "event", "covariates": ["treatment"]},
                )
                bad = await session.call_tool("events_required", {"hazard_ratio": 1.0})
                hits = await session.call_tool("search_guidance", {"query": "non-proportional hazards RMST"})
                return names, json.loads(res.content[0].text), bad, json.loads(hits.content[0].text)

    names, cox, bad, hits = asyncio.run(run())
    assert {"load_dataset", "cox_regression", "sample_size", "simulate_power", "search_guidance"} <= names
    assert hits["result"]["results"][0]["source"].startswith("PMID 42")
    assert cox["result"]["coefficients"][0]["coef"] == pytest.approx(1.572, abs=1e-3)
    is_error = getattr(bad, "is_error", None)
    if is_error is None:  # MCP SDK 1.x
        is_error = bad.isError
    assert is_error  # invalid input comes back as a tool error, not a crash
    assert "no effect" in bad.content[0].text  # with a message the agent can act on
