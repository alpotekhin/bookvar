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


def load_importer():
    spec = importlib.util.spec_from_file_location("import_stanford_cs336", IMPORTER_PATH)
    assert spec and spec.loader
    importer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = importer
    spec.loader.exec_module(importer)
    return importer


def overlay_fixture_documents() -> tuple[dict, dict, dict, dict]:
    manifest = {
        "schema_version": 1,
        "objects": [
            {
                "id": "lecture-reviewed",
                "kind": "lecture",
                "title": "Reviewed lecture",
                "source": "source.md",
            },
            {
                "id": "lecture-future",
                "kind": "lecture",
                "title": "Future lecture",
                "source": "future.md",
            },
        ],
    }
    units = {
        "schema_version": 1,
        "units": [
            {
                "id": "lecture-reviewed-unit-foundation",
                "source_object": "lecture-reviewed",
                "kind": "section",
                "title": "Foundation",
                "source_location": "source.md#L1-L8",
            },
            {
                "id": "lecture-reviewed-unit-caveat",
                "source_object": "lecture-reviewed",
                "kind": "failure-mode",
                "title": "Caveat",
                "source_location": "source.md#L9-L12",
            },
            {
                "id": "lecture-future-unit-pending",
                "source_object": "lecture-future",
                "kind": "section",
                "title": "Pending",
                "source_location": "future.md#L1-L4",
            },
        ],
    }
    coverage = {
        "schema_version": 1,
        "rows": [
            {
                "source_unit": "lecture-reviewed-unit-foundation",
                "source_object": "lecture-reviewed",
                "title": "Foundation",
                "kind": "section",
                "source_location": "source.md#L1-L8",
                "disposition": "source-only",
                "destination": "05 Источники/Courses/Stanford CS336 Spring 2026/_index.md",
                "destination_anchor": "lecture-reviewed",
                "reason": "Editorial integration is pending.",
                "evidence": "Semantic source unit extracted from the pinned snapshot.",
            },
            {
                "source_unit": "lecture-reviewed-unit-caveat",
                "source_object": "lecture-reviewed",
                "title": "Caveat",
                "kind": "failure-mode",
                "source_location": "source.md#L9-L12",
                "disposition": "source-only",
                "destination": "05 Источники/Courses/Stanford CS336 Spring 2026/_index.md",
                "destination_anchor": "lecture-reviewed",
                "reason": "Editorial integration is pending.",
                "evidence": "Semantic source unit extracted from the pinned snapshot.",
            },
            {
                "source_unit": "lecture-future-unit-pending",
                "source_object": "lecture-future",
                "title": "Pending",
                "kind": "section",
                "source_location": "future.md#L1-L4",
                "disposition": "source-only",
                "destination": "05 Источники/Courses/Stanford CS336 Spring 2026/_index.md",
                "destination_anchor": "lecture-future",
                "reason": "Editorial integration is pending.",
                "evidence": "Semantic source unit extracted from the pinned snapshot.",
            },
        ],
    }
    visuals = {
        "schema_version": 1,
        "rows": [
            {
                "id": "lecture-reviewed-visual-foundation",
                "source_object": "lecture-reviewed",
                "source_units": ["lecture-reviewed-unit-foundation"],
                "source_location": "source.pdf#pages=1-2",
                "disposition": "source-only",
                "destination": "05 Источники/Courses/Stanford CS336 Spring 2026/_index.md",
                "destination_anchor": "lecture-reviewed",
                "reason": "Editorial integration is pending.",
                "evidence": "Meaningful visual extracted from the pinned snapshot.",
            },
            {
                "id": "lecture-future-visual-pending",
                "source_object": "lecture-future",
                "source_units": ["lecture-future-unit-pending"],
                "source_location": "future.pdf#page=1",
                "disposition": "source-only",
                "destination": "05 Источники/Courses/Stanford CS336 Spring 2026/_index.md",
                "destination_anchor": "lecture-future",
                "reason": "Editorial integration is pending.",
                "evidence": "Meaningful visual extracted from the pinned snapshot.",
            },
        ],
    }
    return manifest, units, coverage, visuals


def complete_overlay() -> dict:
    return {
        "schema_version": 1,
        "reviewed_objects": ["lecture-reviewed"],
        "coverage": {
            "lecture-reviewed-unit-foundation": {
                "disposition": "integrated",
                "destination": "publishing/tests/fixtures/overlay-destination.md",
                "destination_anchor": "foundations",
            },
            "lecture-reviewed-unit-caveat": {
                "disposition": "source-only",
                "reason": "The caveat is retained in the source hub for a later systems chapter.",
                "evidence": "Editorial review 2026-09-04; source.md#L9-L12.",
            },
        },
        "visuals": {
            "lecture-reviewed-visual-foundation": {
                "disposition": "source-only",
                "reason": "The source figure adds no teaching value beyond the integrated prose.",
                "evidence": "Editorial review 2026-09-04; source.pdf#pages=1-2.",
            }
        },
    }


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

    def test_lecture_10_uses_reviewed_disjoint_ranges_and_pruning_context(self) -> None:
        units_by_id = {unit["id"]: unit for unit in self.units["units"]}
        expected_ranges = {
            "lecture-10-section-133-attention-layers-focusing-on-the-matrix-multiplications-with-flashattention": [[219, 260], [287, 369]],
            "lecture-10-section-223-taking-shortcuts-lossy": [[27, 32], [371, 487], [490, 502]],
            "lecture-10-section-311-activation-aware-quantization-awq": [[480, 487]],
            "lecture-10-section-341-use-shortcuts-but-double-check-lossless": [[43, 45], [507, 550]],
            "lecture-10-section-381-handling-dynamic-workloads": [[46, 54], [553, 607]],
        }
        for unit_id, ranges in expected_ranges.items():
            unit = units_by_id[unit_id]
            self.assertEqual(unit["source_line_ranges"], ranges)
            self.assertNotIn("source_line_start", unit)
            self.assertNotIn("source_line_end", unit)

        pruning = units_by_id["lecture-10-model-pruning-distillation"]
        self.assertEqual(pruning["source_line_start"], 490)
        self.assertEqual(pruning["source_line_end"], 502)

        visuals_by_id = {row["id"]: row for row in self.visuals["rows"]}
        for step in (325, 331):
            visual = visuals_by_id[
                f"lecture-10-visual-figure-step-{step}-rendering-1"
            ]
            self.assertIn("lecture-10-model-pruning-distillation", visual["source_units"])
            self.assertIn("Structured pruning", visual["question"])
            self.assertNotIn("Activation-aware quantization", visual["question"])

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
        importer = load_importer()
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

    def test_pdf_page_closure_rejects_self_consistent_semantic_omission(self) -> None:
        importer = load_importer()
        closure_validator = getattr(importer, "validate_pdf_page_closure", None)
        self.assertIsNotNone(
            closure_validator,
            "importer must expose reviewed-PDF semantic page-closure validation",
        )
        if closure_validator is None:
            return

        complete_failures: list[str] = []
        closure_validator(self.units, self.coverage, complete_failures)
        self.assertEqual(complete_failures, [])

        expected = {
            "source-manifest.yml": self.manifest,
            "source-units.yml": self.units,
            "coverage.yml": self.coverage,
            "visuals.yml": self.visuals,
        }
        omitted = copy.deepcopy(expected)
        semantic_id = "performance-recap-transition"
        omitted_units = omitted["source-units.yml"]["units"]
        removed_units = [
            unit
            for unit in omitted_units
            if unit.get("source_object") == "lecture-05"
            and unit.get("semantic_id") == semantic_id
        ]
        self.assertEqual(len(removed_units), 1)
        if len(removed_units) != 1:
            return
        removed_id = removed_units[0]["id"]
        omitted["source-units.yml"]["units"] = [
            unit for unit in omitted_units if unit.get("id") != removed_id
        ]
        omitted["coverage.yml"]["rows"] = [
            row
            for row in omitted["coverage.yml"]["rows"]
            if row.get("source_unit") != removed_id
        ]

        omitted_lock = copy.deepcopy(self.lock)
        for filename, document in omitted.items():
            audit = omitted_lock["generated_audit"]["ledgers"][filename]
            audit["sha256"] = hashlib.sha256(importer.document_bytes(document)).hexdigest()
            key = "objects" if filename == "source-manifest.yml" else (
                "units" if filename == "source-units.yml" else "rows"
            )
            audit["records"] = len(document[key])

        deterministic_failures: list[str] = []
        importer.validate_generated_documents(
            omitted,
            omitted,
            omitted_lock,
            deterministic_failures,
        )
        self.assertEqual(deterministic_failures, [])

        closure_failures: list[str] = []
        closure_validator(
            omitted["source-units.yml"],
            omitted["coverage.yml"],
            closure_failures,
        )
        self.assertTrue(
            any(
                "lecture-05: semantic page closure missing pages [49]" in failure
                for failure in closure_failures
            )
        )


class StanfordEditorialOverlayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.importer = load_importer()
        self.manifest, self.units, self.coverage, self.visuals = overlay_fixture_documents()

    def apply(self, overlay: dict) -> tuple[dict, dict, str]:
        return self.importer.apply_editorial_overlay(
            self.manifest,
            self.units,
            self.coverage,
            self.visuals,
            overlay,
            repository_root=REPOSITORY_ROOT,
        )

    def test_unknown_source_id_is_rejected(self) -> None:
        overlay = complete_overlay()
        overlay["coverage"]["lecture-reviewed-unit-unknown"] = {
            "disposition": "source-only",
            "reason": "Reviewed and deliberately deferred.",
            "evidence": "Editorial review 2026-09-04.",
        }
        with self.assertRaisesRegex(ValueError, "unknown coverage source unit"):
            self.apply(overlay)

    def test_semantic_source_field_override_is_rejected(self) -> None:
        overlay = complete_overlay()
        overlay["coverage"]["lecture-reviewed-unit-foundation"]["title"] = "Rewritten title"
        with self.assertRaisesRegex(ValueError, "unsupported coverage editorial field: title"):
            self.apply(overlay)

    def test_missing_decision_in_reviewed_object_is_rejected(self) -> None:
        overlay = complete_overlay()
        del overlay["coverage"]["lecture-reviewed-unit-caveat"]
        with self.assertRaisesRegex(ValueError, "missing coverage decision.*lecture-reviewed-unit-caveat"):
            self.apply(overlay)

    def test_invalid_destination_is_rejected_before_regeneration(self) -> None:
        overlay = complete_overlay()
        overlay["coverage"]["lecture-reviewed-unit-foundation"]["destination_anchor"] = "absent"
        with self.assertRaisesRegex(ValueError, "destination anchor does not exist"):
            self.apply(overlay)

    def test_regeneration_preserves_future_baseline_and_records_overlay_sha(self) -> None:
        overlay = complete_overlay()
        first_coverage, first_visuals, first_sha = self.apply(overlay)
        second_coverage, second_visuals, second_sha = self.apply(copy.deepcopy(overlay))

        self.assertEqual(first_coverage, second_coverage)
        self.assertEqual(first_visuals, second_visuals)
        self.assertEqual(first_sha, second_sha)
        self.assertRegex(first_sha, r"^[a-f0-9]{64}$")
        self.assertEqual(first_coverage["editorial_overlay_sha256"], first_sha)
        self.assertEqual(first_visuals["editorial_overlay_sha256"], first_sha)

        future_coverage = next(
            row for row in first_coverage["rows"]
            if row["source_unit"] == "lecture-future-unit-pending"
        )
        future_visual = next(
            row for row in first_visuals["rows"]
            if row["id"] == "lecture-future-visual-pending"
        )
        self.assertEqual(future_coverage, self.coverage["rows"][2])
        self.assertEqual(future_visual, self.visuals["rows"][1])


if __name__ == "__main__":
    unittest.main()
