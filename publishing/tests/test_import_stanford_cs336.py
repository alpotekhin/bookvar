from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COURSE_ROOT = REPOSITORY_ROOT / "05 Источники/Courses/Stanford CS336 Spring 2026"
IMPORTER_PATH = REPOSITORY_ROOT / "publishing/tools/import_stanford_cs336.py"


def read_document(name: str) -> dict:
    return json.loads((COURSE_ROOT / name).read_text("utf-8"))


class StanfordSemanticAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = read_document("source-manifest.yml")
        cls.units = read_document("source-units.yml")
        cls.coverage = read_document("coverage.yml")
        cls.visuals = read_document("visuals.yml")
        cls.lock = read_document("snapshot-lock.json")

    def test_units_are_semantic_and_administration_is_explicitly_excluded(self) -> None:
        units = self.units["units"]
        forbidden_proxy_kinds = {"page", "rendered-text", "rendered-link"}
        self.assertTrue(forbidden_proxy_kinds.isdisjoint({unit["kind"] for unit in units}))
        self.assertFalse(any(re.match(r"(?:Handout|Safety supplement) page \d+", unit["title"]) for unit in units))

        coverage_by_unit = {row["source_unit"]: row for row in self.coverage["rows"]}
        administrative = [unit for unit in units if unit["kind"] == "administrative"]
        self.assertGreaterEqual(len(administrative), 6)
        for unit in administrative:
            row = coverage_by_unit[unit["id"]]
            self.assertEqual(row["disposition"], "excluded")
            self.assertTrue(row.get("reason"))
            self.assertTrue(row.get("evidence"))

        required_semantics = {"section", "derivation", "experiment", "worked-example", "failure-mode"}
        self.assertTrue(required_semantics.issubset({unit["kind"] for unit in units}))

    def test_visuals_link_to_units_without_page_raster_duplicates(self) -> None:
        units_by_id = {unit["id"]: unit for unit in self.units["units"]}
        seen_semantic_locations: set[tuple[str, str]] = set()
        for visual in self.visuals["rows"]:
            linked = visual.get("source_units")
            self.assertIsInstance(linked, list)
            self.assertTrue(linked)
            for unit_id in linked:
                self.assertIn(unit_id, units_by_id)
                self.assertEqual(units_by_id[unit_id]["source_object"], visual["source_object"])
            key = (visual["source_object"], visual["source_location"])
            self.assertNotIn(key, seen_semantic_locations)
            seen_semantic_locations.add(key)
            self.assertNotRegex(visual["question"], r"What (?:visual argument|argument does embedded image)")

        multi_event = [
            row for row in self.visuals["rows"]
            if row["source_object"].startswith("lecture-")
            and "var/traces" in row["source_location"]
            and len(row["sequence_members"]) > 1
        ]
        multi_page = [
            row for row in self.visuals["rows"]
            if "#pages=" in row["source_location"] and len(row["sequence_members"]) > 1
        ]
        self.assertTrue(multi_event)
        self.assertTrue(multi_page)

    def test_lecture_6_accelerator_table_is_one_progressive_sequence(self) -> None:
        matches = [
            row for row in self.visuals["rows"]
            if row["source_object"] == "lecture-06"
            and row.get("semantic_id") == "accelerator-memory-hierarchy-table"
        ]
        self.assertEqual(len(matches), 1)
        if not matches:
            return
        table = matches[0]
        self.assertEqual(table["visual_kind"], "table")
        self.assertEqual(table["source_pages"], list(range(8, 21)))
        self.assertEqual(len(table["sequence_members"]), 13)

    def test_assignment_2_figures_have_source_units_and_coverage(self) -> None:
        unit_ids = {unit["id"] for unit in self.units["units"]}
        covered = {row["source_unit"] for row in self.coverage["rows"]}
        assignment_visuals = {
            row["semantic_id"]: row
            for row in self.visuals["rows"]
            if row["source_object"] == "assignment-02"
            and isinstance(row.get("semantic_id"), str)
        }
        required = {"nsight-systems-trace", "distributed-rank-topology"}
        self.assertTrue(required.issubset(set(assignment_visuals)))
        for semantic_id in required:
            row = assignment_visuals[semantic_id]
            self.assertEqual(len(row["source_units"]), 1)
            self.assertIn(row["source_units"][0], unit_ids)
            self.assertIn(row["source_units"][0], covered)

    def test_provenance_records_real_tools_extractor_and_parent_hashes(self) -> None:
        provenance = self.visuals.get("extraction_provenance")
        self.assertIsInstance(provenance, dict)
        if not isinstance(provenance, dict):
            return
        self.assertEqual(provenance["edtrace"], "0.1.15")
        self.assertEqual(provenance["pdfinfo"], "26.04.0")
        self.assertEqual(provenance["pdftotext"], "26.04.0")
        self.assertEqual(provenance["pdfimages"], "26.04.0")
        self.assertRegex(provenance["extractor_sha256"], r"^[a-f0-9]{64}$")
        self.assertTrue(provenance["extractor_revision"])
        for row in self.visuals["rows"]:
            self.assertRegex(row["parent_sha256"], r"^[a-f0-9]{64}$")
            self.assertRegex(row["extractor_sha256"], r"^[a-f0-9]{64}$")

    def test_deterministic_validator_rejects_self_consistent_truncation(self) -> None:
        spec = importlib.util.spec_from_file_location("import_stanford_cs336", IMPORTER_PATH)
        assert spec and spec.loader
        importer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = importer
        spec.loader.exec_module(importer)
        validator = getattr(importer, "validate_generated_documents", None)
        self.assertIsNotNone(validator, "importer must expose deterministic generated-document validation")
        if validator is None:
            return

        expected = {
            "source-manifest.yml": self.manifest,
            "source-units.yml": self.units,
            "coverage.yml": self.coverage,
            "visuals.yml": self.visuals,
        }
        truncated = copy.deepcopy(expected)
        removed = truncated["source-units.yml"]["units"].pop()
        truncated["coverage.yml"]["rows"] = [
            row for row in truncated["coverage.yml"]["rows"]
            if row["source_unit"] != removed["id"]
        ]
        truncated_lock = copy.deepcopy(self.lock)
        for filename, document in truncated.items():
            audit = truncated_lock["generated_audit"]["ledgers"][filename]
            audit["sha256"] = hashlib.sha256(importer.document_bytes(document)).hexdigest()
            key = "objects" if filename == "source-manifest.yml" else (
                "units" if filename == "source-units.yml" else "rows"
            )
            audit["records"] = len(document[key])
        failures: list[str] = []
        validator(truncated, expected, truncated_lock, failures)
        self.assertTrue(any("deterministic extraction mismatch" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
