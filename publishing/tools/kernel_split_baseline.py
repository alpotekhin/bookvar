"""Restore the historical whole owner for preservation checks, not execution."""
import json
from pathlib import Path

from topic_consolidation_baseline import before_topic_consolidation

ROOT = Path(__file__).resolve().parents[2]
KERNELS = Path("00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md")


def kernels_before_graph_split():
    text = before_topic_consolidation((ROOT / KERNELS).read_text(), KERNELS)
    route = json.loads((ROOT / "docs/audit/2026-10-29-reader-route-navigation.json").read_text())
    row = next(item for item in route["files"] if item["source"] == KERNELS.as_posix())
    for change in reversed(row["changes"]):
        if text.count(change["new"]) != 1:
            raise AssertionError("Unknown historical kernel navigation")
        text = text.replace(change["new"], change["old"], 1)
    return text
