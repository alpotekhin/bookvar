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
COURSE_ROOT = REPOSITORY_ROOT / "05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025"
IMPORTER_PATH = REPOSITORY_ROOT / "publishing/tools/import_berkeley_agents.py"
EXPECTED_READING_DISTRIBUTION = [3, 3, 3, 3, 2, 4, 2, 4, 3, 4, 2, 4]


def read_document(name: str) -> dict:
    return json.loads((COURSE_ROOT / name).read_text("utf-8"))


class BerkeleyAgentsSourceAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = read_document("source-manifest.yml")
        cls.units = read_document("source-units.yml")
        cls.coverage = read_document("coverage.yml")
        cls.visuals = read_document("visuals.yml")
        cls.lock = read_document("snapshot-lock.json")

    def test_exact_meeting_deck_and_reading_inventory(self) -> None:
        bundles = self.manifest["meeting_bundles"]
        self.assertEqual(len(bundles), 12)
        self.assertEqual([len(bundle["readings"]) for bundle in bundles], EXPECTED_READING_DISTRIBUTION)
        self.assertEqual(sum(len(bundle["readings"]) for bundle in bundles), 37)
        self.assertEqual(sum(len(bundle["decks"]) for bundle in bundles), 13)

        objects = {row["id"]: row for row in self.manifest["objects"]}
        flattened_readings = [reading for bundle in bundles for reading in bundle["readings"]]
        self.assertEqual(len(flattened_readings), len(set(flattened_readings)))
        self.assertEqual(len(flattened_readings), 37)
        for bundle in bundles:
            self.assertIn(bundle["recording"], objects)
            for object_id in [*bundle["decks"], *bundle["readings"]]:
                self.assertIn(object_id, objects)
                self.assertEqual(objects[object_id]["meeting_date"], bundle["meeting_date"])

        reading_objects = [row for row in objects.values() if row.get("catalogue_kind") == "reading"]
        self.assertEqual(len(reading_objects), 37)
        for reading in reading_objects:
            self.assertTrue(reading["title"])
            self.assertTrue(reading["authors"])
            self.assertRegex(str(reading["year"]), r"^\d{4}$")
            self.assertTrue(reading["canonical_project_url"].startswith("https://"))
            self.assertEqual(reading["relationship_to_deck"], "primary source for technical claims")

    def test_syllabus_membership_fails_closed_on_unknown_artifact_link(self) -> None:
        importer = self._load_importer()
        payload = (COURSE_ROOT / "Metadata/syllabus.html").read_bytes()
        needle = b'<a href="https://arxiv.org/abs/2309.03409">Large Language Models as Optimizers</a>'
        replacement = needle + b' <br /> - <a href="https://example.org/unexpected-reading">Unexpected reading</a>'
        self.assertIn(needle, payload)
        failures: list[str] = []
        importer.validate_syllabus_inventory(payload.replace(needle, replacement, 1), failures)
        self.assertTrue(any("unexpected artifact link" in failure for failure in failures))

    def test_jan_27_preserves_the_extra_intro_deck(self) -> None:
        first = self.manifest["meeting_bundles"][0]
        self.assertEqual(first["meeting_date"], "2025-01-27")
        self.assertEqual(first["decks"], ["meeting-01-intro", "meeting-01-slides"])
        objects = {row["id"]: row for row in self.manifest["objects"]}
        self.assertEqual(objects["meeting-01-intro"]["page_count"], 16)
        self.assertEqual(objects["meeting-01-intro"]["artifact_role"], "additional official intro deck in meeting 1")

    def test_recordings_preserve_exact_official_youtube_metadata(self) -> None:
        expected = [
            ("g0Dwtf3BH-0", 4892, "Adv. LLM Agents MOOC | UC Berkeley Sp25 | Inference-Time Techniques for LLM Reasoning by Xinyun Chen"),
            ("_MNlLhU33H0", 4607, "Adv. LLM Agents MOOC | UC Berkeley CS294-280 Sp25 | Learning to Reason with LLMs by Jason Weston"),
            ("zvI4UN2_i-w", 5559, "Adv. LLM Agents MOOC | UC Berkeley Sp25 | Reasoning, Memory & Planning of Language Agents by Yu Su"),
            ("cMiu3A7YBks", 4853, "Adv. LLM Agents MOOC | UC Berkeley Sp25 | Open Training Recipes: LLM Reasoning by Hanna Hajishirzi"),
            ("JCk6qJtaCSU", 5222, "Adv. LLM Agents MOOC | UC Berkeley Sp25 | Code Agents & AI Vulnerability Detection by Charles Sutton"),
            ("RPINOYM12RU", 4662, "Adv. LLM Agents MOOC | UC Berkeley Sp25 | Multimodal Autonomous AI Agents by Ruslan Salakhutdinov"),
            ("n__Tim8K2IY", 5301, "Adv. LLM Agents MOOC | UC Berkeley Sp25 | Multimodal Agents – Perception to Action by Caiming Xiong"),
            ("3gaEMscOMAU", 4448, "Adv. LLM Agents MOOC | UC Berkeley CS294-280 Sp25 | AlphaProof RL Meets Formal Math by Thomas Hubert"),
            ("cLhWEyMQ4mQ", 3127, "Adv. LLM Agents MOOC | UC Berkeley Sp25 | LMs for Autoformalization+Theorem Proving by Kaiyu Yang"),
            ("Gy5Nm17l9oo", 4332, "Adv. LLM Agents MOOC | UC Berkeley CS294-280 Sp25 | Informal+Formal MathReasoning by Sean Welleck"),
            ("IHc0TEMrEdY", 5258, "Adv. LLM Agents MOOC | UC Berkeley Sp25 | Abstraction, Discovery w/ LLM Agents by Swarat Chaudhuri"),
            ("ti6yPE2VPZc", 6644, "Adv. LLM Agents MOOC | UC Berkeley CS294-280 Sp25 | Towards Safe & Secure Agentic AI by Dawn Song"),
        ]
        objects = {row["id"]: row for row in self.manifest["objects"]}
        actual = []
        for bundle in self.manifest["meeting_bundles"]:
            recording = objects[bundle["recording"]]
            actual.append((recording["video_id"], recording["video_metadata"]["duration_seconds"], recording["title"]))
            self.assertEqual(recording["channel"], "Berkeley RDI")
            self.assertEqual(recording["channel_id"], "UCB67PxhB5LAWEbI4etQS7aw")
            self.assertEqual(recording["channel_handle"], "@BerkeleyRDI")
            self.assertEqual(recording["metadata_retrieved_at"], "2026-09-04")
            self.assertEqual(recording["metadata_evidence_url"], recording["canonical_url"])
        self.assertEqual(actual, expected)

    def test_all_deck_pages_have_exact_semantic_closure(self) -> None:
        objects = {row["id"]: row for row in self.manifest["objects"]}
        deck_ids = [deck for bundle in self.manifest["meeting_bundles"] for deck in bundle["decks"]]
        units_by_object: dict[str, list[dict]] = {deck: [] for deck in deck_ids}
        for unit in self.units["units"]:
            if unit["source_object"] in units_by_object:
                units_by_object[unit["source_object"]].append(unit)

        forbidden_proxy_kinds = {"page", "heading", "rendered-text", "rendered-link"}
        for deck_id in deck_ids:
            seen: list[int] = []
            deck_units = units_by_object[deck_id]
            self.assertTrue(deck_units, deck_id)
            for unit in deck_units:
                self.assertNotIn(unit["kind"], forbidden_proxy_kinds)
                self.assertNotRegex(unit["title"], r"(?i)^(?:slide|deck|pdf) pages? \d+")
                start = unit.get("page", unit.get("page_start"))
                end = unit.get("page", unit.get("page_end"))
                self.assertIsInstance(start, int)
                self.assertIsInstance(end, int)
                seen.extend(range(start, end + 1))
            self.assertEqual(seen, list(range(1, objects[deck_id]["page_count"] + 1)), deck_id)

        coverage_by_unit = {row["source_unit"]: row for row in self.coverage["rows"]}
        for unit in self.units["units"]:
            if unit["source_object"] not in units_by_object:
                continue
            row = coverage_by_unit[unit["id"]]
            if unit["kind"] == "administrative":
                self.assertEqual(row["disposition"], "excluded")
                self.assertTrue(row["reason"])
                self.assertTrue(row["evidence"])

    def test_visual_ledger_closes_every_deck_page_and_links_provenance(self) -> None:
        deck_ids = [deck for bundle in self.manifest["meeting_bundles"] for deck in bundle["decks"]]
        objects = {row["id"]: row for row in self.manifest["objects"]}
        units = {row["id"]: row for row in self.units["units"]}
        rows_by_object: dict[str, list[dict]] = {deck: [] for deck in deck_ids}
        for row in self.visuals["rows"]:
            if row["source_object"] in rows_by_object:
                rows_by_object[row["source_object"]].append(row)

        for deck_id in deck_ids:
            pages: list[int] = []
            self.assertTrue(rows_by_object[deck_id], deck_id)
            for row in rows_by_object[deck_id]:
                self.assertTrue(row["source_units"])
                for unit_id in row["source_units"]:
                    self.assertEqual(units[unit_id]["source_object"], deck_id)
                self.assertRegex(row["parent_sha256"], r"^[a-f0-9]{64}$")
                self.assertRegex(row["extractor_sha256"], r"^[a-f0-9]{64}$")
                self.assertTrue(row["extractor_revision"])
                self.assertTrue(row["extraction_tool"])
                pages.extend(row["source_pages"])
            self.assertEqual(pages, list(range(1, objects[deck_id]["page_count"] + 1)), deck_id)

    def test_required_multi_member_sequences_are_real(self) -> None:
        by_semantic_id = {row.get("semantic_id"): row for row in self.visuals["rows"]}
        required = {
            "yu-su-agent-first-vs-llm-first": "meeting-03-slides",
            "yu-su-hipporag-memory-sequence": "meeting-03-slides",
            "yu-su-world-model-planning-sequence": "meeting-03-slides",
            "charles-sutton-vulnerability-discovery-loop": "meeting-05-slides",
            "kaiyu-yang-lean-theorem-proving-pipeline": "meeting-09-slides",
            "swarat-chaudhuri-lasr-concept-library-sequence": "meeting-11-slides",
            "dawn-song-agentic-threat-model-sequence": "meeting-12-slides",
            "dawn-song-privilege-control-sequence": "meeting-12-slides",
        }
        for semantic_id, source_object in required.items():
            self.assertIn(semantic_id, by_semantic_id)
            row = by_semantic_id[semantic_id]
            self.assertEqual(row["source_object"], source_object)
            self.assertGreater(len(row["source_pages"]), 1)
            self.assertEqual(len(row["source_pages"]), len(row["sequence_members"]))
            self.assertEqual(
                [member["order"] for member in row["sequence_members"]],
                list(range(1, len(row["sequence_members"]) + 1)),
            )

    def test_reading_coverage_keeps_primary_secondary_hierarchy(self) -> None:
        bundles_by_reading = {
            reading: bundle for bundle in self.manifest["meeting_bundles"] for reading in bundle["readings"]
        }
        reading_units = {
            unit["id"]: unit for unit in self.units["units"] if unit["source_object"] in bundles_by_reading
        }
        rows = {row["source_unit"]: row for row in self.coverage["rows"]}
        self.assertEqual(len(reading_units), 37)
        for unit_id, unit in reading_units.items():
            row = rows[unit_id]
            bundle = bundles_by_reading[unit["source_object"]]
            self.assertEqual(row["primary_sources"], [unit["source_object"]])
            self.assertEqual(row["secondary_sources"], bundle["decks"])
            self.assertEqual(row["disposition"], "source-only")

    def test_practice_provenance_is_explicit_and_mirror_is_excluded(self) -> None:
        practice = self.manifest["practice_provenance"]
        self.assertEqual(practice["official_public_status"], "lab and project confirmed; no verified official lab artifact exposed")
        self.assertEqual(practice["drive_artifact_status"], "not pinned; unavailable for use")
        self.assertIn("file ID", practice["drive_pin_requirements"])
        self.assertIn("export checksum", practice["drive_pin_requirements"])
        self.assertIn("retrieval date", practice["drive_pin_requirements"])

        objects = {row["id"]: row for row in self.manifest["objects"]}
        mirror = objects["practice-precioux-discovery"]
        self.assertEqual(mirror["source_authority"], "third-party-mirror")
        self.assertEqual(mirror["revision_or_checksum"], "16f7b26346ec4b8978199e6d35bcee96b128bd0a")
        self.assertIn("/tree/16f7b26346ec4b8978199e6d35bcee96b128bd0a", mirror["canonical_url"])
        mirror_units = [unit for unit in self.units["units"] if unit["source_object"] == mirror["id"]]
        self.assertEqual(len(mirror_units), 1)
        coverage = next(row for row in self.coverage["rows"] if row["source_unit"] == mirror_units[0]["id"])
        self.assertEqual(coverage["disposition"], "excluded")
        self.assertEqual(coverage["reason"], "discovery lead only")
        self.assertIn("cannot substantiate an official lab contract", coverage["evidence"])

        hub = (COURSE_ROOT / "_index.md").read_text("utf-8")
        self.assertIn("does not expose a verified official lab artifact", hub)
        self.assertIn("adaptation inspired by the course", hub)

    def test_baseline_has_no_textbook_integration_and_no_invented_license(self) -> None:
        self.assertTrue(
            all(row["disposition"] in {"source-only", "excluded"} for row in self.coverage["rows"])
        )
        self.assertTrue(
            all(row["disposition"] in {"source-only", "excluded"} for row in self.visuals["rows"])
        )
        course_objects = [
            row for row in self.manifest["objects"] if row["source_authority"] == "official-course"
        ]
        self.assertTrue(course_objects)
        self.assertTrue(all(row["rights_status"] == "permission-recorded" for row in course_objects))
        for row in self.visuals["rows"]:
            if row["rights_status"] == "permission-recorded":
                self.assertEqual(row["license_identifier"], "User permission record (not a license)")

    def test_deterministic_validator_rejects_self_consistent_truncation(self) -> None:
        importer = self._load_importer()
        expected = {
            "source-manifest.yml": self.manifest,
            "source-units.yml": self.units,
            "coverage.yml": self.coverage,
            "visuals.yml": self.visuals,
        }
        truncated = copy.deepcopy(expected)
        removed = truncated["source-units.yml"]["units"].pop()
        truncated["coverage.yml"]["rows"] = [
            row for row in truncated["coverage.yml"]["rows"] if row["source_unit"] != removed["id"]
        ]
        truncated["visuals.yml"]["rows"] = [
            row for row in truncated["visuals.yml"]["rows"] if removed["id"] not in row.get("source_units", [])
        ]
        forged_lock = copy.deepcopy(self.lock)
        for filename, document in truncated.items():
            audit = forged_lock["generated_audit"]["ledgers"][filename]
            audit["sha256"] = hashlib.sha256(importer.document_bytes(document)).hexdigest()
            audit["records"] = len(document["objects" if filename == "source-manifest.yml" else ("units" if filename == "source-units.yml" else "rows")])
        failures: list[str] = []
        importer.validate_generated_documents(truncated, expected, forged_lock, failures)
        self.assertTrue(any("deterministic extraction mismatch" in failure for failure in failures))

    def test_page_closure_rejects_semantic_omission_even_with_forged_lock(self) -> None:
        importer = self._load_importer()
        failures: list[str] = []
        importer.validate_pdf_page_closure(self.manifest, self.units, self.coverage, failures)
        self.assertEqual(failures, [])

        omitted_units = copy.deepcopy(self.units)
        victim = next(
            unit for unit in omitted_units["units"]
            if unit.get("semantic_id") == "charles-sutton-vulnerability-discovery-loop"
        )
        omitted_units["units"] = [unit for unit in omitted_units["units"] if unit["id"] != victim["id"]]
        omitted_coverage = copy.deepcopy(self.coverage)
        omitted_coverage["rows"] = [
            row for row in omitted_coverage["rows"] if row["source_unit"] != victim["id"]
        ]
        failures = []
        importer.validate_pdf_page_closure(self.manifest, omitted_units, omitted_coverage, failures)
        self.assertTrue(any("meeting-05-slides: semantic page closure missing pages" in failure for failure in failures))


    def test_source_marked_confidential_page_overrides_general_permission(self) -> None:
        objects = {row["id"]: row for row in self.manifest["objects"]}
        deck = objects["meeting-06-slides"]
        self.assertEqual(deck["rights_status"], "permission-recorded")
        self.assertEqual(deck["rights_overrides"], [{
            "physical_page": 117,
            "marking": "NVIDIA CONFIDENTIAL. DO NOT DISTRIBUTE.",
            "disposition": "excluded",
            "rights_scope": "do-not-reuse; preserve unchanged official archive only",
            "precedence": "overrides the general permission record for this page",
        }])

        row = next(
            visual for visual in self.visuals["rows"]
            if visual.get("semantic_id") == "nvidia-confidential-page"
        )
        self.assertEqual(row["source_pages"], [117])
        self.assertEqual(row["disposition"], "excluded")
        self.assertEqual(row["rights_status"], "unknown")
        self.assertIn("do-not-reuse", row["reason"])
        self.assertIn("CONFIDENTIAL", row["evidence"])
        self.assertNotIn("license_identifier", row)
        self.assertNotIn("license_url", row)
    @staticmethod
    def _load_importer():
        spec = importlib.util.spec_from_file_location("import_berkeley_agents", IMPORTER_PATH)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
