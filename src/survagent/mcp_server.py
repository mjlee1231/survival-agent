"""MCP server exposing the survival-analysis tools to AI agents.

Run with ``survagent-mcp`` (stdio transport) after ``pip install -e .``.
"""

from __future__ import annotations

import functools

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server
    from mcp.server.fastmcp.exceptions import ToolError

from . import tools

INSTRUCTIONS = """\
Statistical tools for time-to-event (survival) analysis and trial design.
Workflow: call load_dataset on a CSV first, then pass its dataset_id to the
analysis tools. When you report results:
- always give estimates with their confidence intervals, not point estimates alone;
- pass on every item in `notes` and `data_notes` (assumption checks and warnings);
- cite the provenance (function, dataset sha256) so results can be reproduced;
- do not claim causality from observational data.
"""

server = _Server("survagent", instructions=INSTRUCTIONS)


def _tool(fn):
    """Register a tool; input problems reach the agent as readable tool errors."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    return server.tool()(wrapper)


@_tool
def load_dataset(path: str) -> dict:
    """Register a CSV file. Returns dataset_id, SHA-256 hash, row count and a column summary."""
    return tools.load_dataset(path)


@_tool
def kaplan_meier(dataset_id: str, time_col: str, event_col: str, group_col: str | None = None,
                 conf_level: float = 0.95) -> dict:
    """Kaplan–Meier survival curves (overall or by group) with log-log CIs and median survival.
    event_col must be coded 1 = event, 0 = censored."""
    return tools.km_tool(dataset_id, time_col, event_col, group_col, conf_level)


@_tool
def logrank_test(dataset_id: str, time_col: str, event_col: str, group_col: str) -> dict:
    """Log-rank test comparing survival across two or more groups."""
    return tools.logrank_tool(dataset_id, time_col, event_col, group_col)


@_tool
def cox_regression(dataset_id: str, time_col: str, event_col: str, covariates: list[str],
                   reference_levels: dict[str, str] | None = None, ties: str = "efron",
                   conf_level: float = 0.95) -> dict:
    """Cox proportional hazards model: hazard ratios with CIs, global tests, and a
    proportional-hazards check. Text covariates are dummy-coded against reference_levels."""
    return tools.cox_tool(dataset_id, time_col, event_col, covariates, reference_levels, ties, conf_level)


@_tool
def events_required(hazard_ratio: float, alpha: float = 0.05, power: float = 0.8,
                    allocation: float = 0.5) -> dict:
    """Events needed for a two-sided log-rank test (Schoenfeld formula)."""
    return tools.events_required_tool(hazard_ratio=hazard_ratio, alpha=alpha, power=power,
                                      allocation=allocation)


@_tool
def power_from_events(hazard_ratio: float, events: int, alpha: float = 0.05,
                      allocation: float = 0.5) -> dict:
    """Approximate power of a two-sided log-rank test for a given number of events."""
    return tools.power_tool(hazard_ratio=hazard_ratio, events=events, alpha=alpha, allocation=allocation)


@_tool
def sample_size(hazard_ratio: float, median_control: float, accrual: float, followup: float,
                alpha: float = 0.05, power: float = 0.8, allocation: float = 0.5) -> dict:
    """Total sample size for a two-arm survival trial (exponential survival, uniform accrual).
    Use the same time unit for median_control, accrual and followup."""
    return tools.sample_size_tool(hazard_ratio=hazard_ratio, median_control=median_control,
                                  accrual=accrual, followup=followup, alpha=alpha, power=power,
                                  allocation=allocation)


@_tool
def simulate_power(hazard_ratio: float, n_total: int, median_control: float, accrual: float,
                   followup: float, alpha: float = 0.05, allocation: float = 0.5,
                   n_sim: int = 1000, seed: int = 2026) -> dict:
    """Monte Carlo check of log-rank power for a proposed design (returns a Monte Carlo CI)."""
    return tools.simulate_power_tool(hazard_ratio=hazard_ratio, n_total=n_total,
                                     median_control=median_control, accrual=accrual,
                                     followup=followup, alpha=alpha, allocation=allocation,
                                     n_sim=n_sim, seed=seed)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
