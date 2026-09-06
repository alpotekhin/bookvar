from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import tempfile
import unittest
from collections import Counter
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
        cls.inventory = read_document("artifact-inventory.json")
        cls.editorial = read_document("editorial-map.yml")

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

    def test_syllabus_membership_rejects_whole_unknown_teaching_row(self) -> None:
        importer = self._load_importer()
        payload = (COURSE_ROOT / "Metadata/syllabus.html").read_bytes()
        extra_row = b"""
        <table><tr>
          <td>Apr 28</td>
          <td><a href="https://www.youtube.com/live/reviewer-attack-13">Unknown recording</a></td>
          <td><a href="https://rdi.berkeley.edu/adv-llm-agents/slides/unknown-13.pdf">Unknown deck</a></td>
          <td><a href="https://arxiv.org/abs/2504.99999">Unknown reading</a></td>
        </tr></table>
        """
        failures: list[str] = []
        importer.validate_syllabus_inventory(payload + extra_row, failures)
        self.assertTrue(any("unexpected teaching row" in failure for failure in failures), failures)

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

    def test_sampled_semantics_preserve_substantive_closing_pages(self) -> None:
        review = read_document("semantic-review.json")
        reviewed_decks = {deck["id"]: deck for deck in review["decks"]}

        meeting_four_page = next(
            section
            for section in reviewed_decks["meeting-04-slides"]["sections"]
            if section["page_start"] <= 155 <= section["page_end"]
        )
        self.assertEqual(
            {
                key: meeting_four_page[key]
                for key in ("title", "semantic_id", "kind", "page_start", "page_end", "visual_kind")
            },
            {
                "title": "Human Preference Evaluation",
                "semantic_id": "human-preference-evaluation",
                "kind": "figure",
                "page_start": 155,
                "page_end": 155,
                "visual_kind": "evaluation-chart",
            },
        )
        self.assertNotIn("disposition", meeting_four_page)

        meeting_ten = reviewed_decks["meeting-10-slides"]["sections"]
        recap = next(section for section in meeting_ten if section["page_start"] <= 116 <= section["page_end"])
        self.assertEqual((recap["page_start"], recap["page_end"]), (106, 117))
        self.assertEqual(recap["semantic_id"], "minictx")
        self.assertEqual(recap["title"], "miniCTX, accessibility gaps, and prover-method recap")
        self.assertEqual(recap["kind"], "experiment")
        self.assertNotIn("disposition", recap)
        closing = next(section for section in meeting_ten if section["page_start"] <= 118 <= section["page_end"])
        self.assertEqual((closing["page_start"], closing["page_end"]), (118, 118))
        self.assertEqual(closing["kind"], "administrative")
        self.assertEqual(closing["disposition"], "excluded")

        units = {unit["semantic_id"]: unit for unit in self.units["units"] if "semantic_id" in unit}
        coverage = {row["source_unit"]: row for row in self.coverage["rows"]}
        for semantic_id in ("human-preference-evaluation", "minictx"):
            self.assertEqual(coverage[units[semantic_id]["id"]]["disposition"], "source-only")
        self.assertEqual(coverage[units["meeting-10-closing"]["id"]]["disposition"], "excluded")

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

    def test_visual_review_method_is_codex_assisted_and_reviewer_is_explicit(self) -> None:
        expected_method = (
            "Codex-assisted ordered contact-sheet review plus "
            "pdftotext -layout (Poppler 26.04.0)"
        )
        self.assertTrue(all(row["extraction_tool"] == expected_method for row in self.visuals["rows"]))
        self.assertTrue(all(row["reviewer"] == "Codex source audit" for row in self.visuals["rows"]))

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

    def test_big_sleep_preserves_primary_byline_organizations_and_contributors(self) -> None:
        expected_contributors = [
            "Miltos Allamanis",
            "Martin Arjovsky",
            "Charles Blundell",
            "Lars Buesing",
            "Mark Brand",
            "Sergei Glazunov",
            "Dominik Maier",
            "Petros Maniatis",
            "Guilherme Marinho",
            "Henryk Michalewski",
            "Koushik Sen",
            "Charles Sutton",
            "Vaibhav Tulsyan",
            "Marco Vanotti",
            "Theophane Weber",
            "Dan Zheng",
        ]
        reading = next(row for row in self.manifest["objects"] if row["id"] == "meeting-05-reading-02")
        self.assertEqual(reading["authors"], ["the Big Sleep team"])
        self.assertEqual(reading["source_byline"], "the Big Sleep team")
        self.assertEqual(reading["organizations"], ["Google Project Zero", "Google DeepMind"])
        self.assertEqual(reading["contributors"], expected_contributors)
        self.assertEqual(
            reading["canonical_project_url"],
            "https://projectzero.google/2024/10/from-naptime-to-big-sleep.html",
        )
        self.assertEqual(
            reading.get("source_metadata_evidence_url"),
            "https://projectzero.google/2024/10/from-naptime-to-big-sleep.html",
        )
        note = (COURSE_ROOT / reading["local_path"]).read_text("utf-8")
        self.assertIn("- Source byline: the Big Sleep team", note)
        self.assertIn("- Organizations: Google Project Zero; Google DeepMind", note)
        self.assertIn("- Source metadata evidence: https://projectzero.google/2024/10/from-naptime-to-big-sleep.html", note)
        self.assertIn(
            "- Contributors: " + "; ".join(expected_contributors),
            note,
        )

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

    def test_staged_editorial_overlay_decides_every_unit_visual_and_object(self) -> None:
        object_ids = {row["id"] for row in self.manifest["objects"]}
        unit_ids = {row["id"] for row in self.units["units"]}
        visual_ids = {row["id"] for row in self.visuals["rows"]}

        self.assertEqual(self.editorial["schema_version"], 1)
        self.assertEqual(len(self.editorial["reviewed_objects"]), len(set(self.editorial["reviewed_objects"])))
        self.assertEqual(set(self.editorial["reviewed_objects"]), object_ids)
        self.assertEqual(set(self.editorial["coverage"]), unit_ids)
        self.assertEqual(set(self.editorial["visuals"]), visual_ids)
        self.assertEqual(
            Counter(row["disposition"] for row in self.editorial["coverage"].values()),
            Counter({"integrated": 137, "covered-existing": 6, "source-only": 41, "excluded": 28}),
        )
        self.assertEqual(
            Counter(row["disposition"] for row in self.editorial["visuals"].values()),
            Counter({"integrated": 42, "source-only": 92, "excluded": 27}),
        )

    def test_staged_overlay_hash_and_counts_are_locked_without_relabelling_baseline(self) -> None:
        importer = self._load_importer()
        expected_hash = importer.editorial_overlay_sha256(self.editorial)
        expected_counts = importer.editorial_disposition_counts(self.editorial)
        self.assertEqual(
            self.lock["generated_audit"]["staged_editorial_overlay_sha256"],
            expected_hash,
        )
        self.assertEqual(self.lock["invariants"]["staged_editorial_counts"], expected_counts)
        self.assertEqual(self.inventory["staged_editorial_counts"], expected_counts)
        self.assertEqual(
            self.lock["invariants"]["editorial_overlay_status"],
            "staged-pending-destination-validation",
        )
        self.assertNotIn("editorial_overlay_sha256", self.coverage)
        self.assertNotIn("editorial_overlay_sha256", self.visuals)

    def test_staged_visual_selection_matches_the_full_audit_exactly(self) -> None:
        expected = {
            "meeting-01-slides-visual-self-consistency-sampling",
            "meeting-01-slides-visual-tree-of-thoughts-search",
            "meeting-01-slides-visual-reflection-and-self-debugging",
            "meeting-01-slides-visual-self-correction-failure-modes",
            "meeting-02-slides-visual-self-rewarding-setup",
            "meeting-02-slides-visual-irpo",
            "meeting-02-slides-visual-meta-rewarding",
            "meeting-02-slides-visual-evalplanner",
            "meeting-03-slides-visual-yu-su-hipporag-memory-sequence",
            "meeting-03-slides-visual-yu-su-world-model-planning-sequence",
            "meeting-04-slides-visual-sft-data-construction",
            "meeting-04-slides-visual-rlvr-method",
            "meeting-04-slides-visual-rlvr-results",
            "meeting-04-slides-visual-s1k-data",
            "meeting-04-slides-visual-test-time-scaling",
            "meeting-05-slides-visual-swe-agent-loop",
            "meeting-05-slides-visual-coding-agent-design-comparisons",
            "meeting-06-slides-visual-visualwebarena",
            "meeting-06-slides-visual-vision-language-web-agents",
            "meeting-06-slides-visual-web-agent-tree-search",
            "meeting-06-slides-visual-web-agent-tree-search-results",
            "meeting-06-slides-visual-insta-setup",
            "meeting-06-slides-visual-insta-verification-scaling",
            "meeting-07-slides-visual-osworld",
            "meeting-07-slides-visual-aguvis",
            "meeting-08-slides-visual-rl-and-alphazero",
            "meeting-08-slides-visual-alphaproof-methods",
            "meeting-08-slides-visual-test-time-rl",
            "meeting-09-slides-visual-kaiyu-yang-lean-theorem-proving-pipeline",
            "meeting-09-slides-visual-leaneuclid",
            "meeting-09-slides-visual-euclid-logical-gap",
            "meeting-10-slides-visual-lean-star",
            "meeting-10-slides-visual-draft-sketch-prove",
            "meeting-10-slides-visual-leanhammer",
            "meeting-10-slides-visual-minictx",
            "meeting-11-slides-visual-copra",
            "meeting-11-slides-visual-compiler-verification",
            "meeting-11-slides-visual-swarat-chaudhuri-lasr-concept-library-sequence",
            "meeting-12-slides-visual-dawn-song-agentic-threat-model-sequence",
            "meeting-12-slides-visual-prompt-injection-agentpoison",
            "meeting-12-slides-visual-dawn-song-privilege-control-sequence",
            "meeting-12-slides-visual-agent-monitoring-and-verification",
        }
        actual = {
            visual_id
            for visual_id, decision in self.editorial["visuals"].items()
            if decision["disposition"] == "integrated"
        }
        self.assertEqual(actual, expected)

    def test_staged_unit_deferrals_and_exclusions_match_the_full_audit_exactly(self) -> None:
        expected_covered = {
            "meeting-01-slides-self-consistency-sampling",
            "meeting-01-slides-tree-of-thoughts-search",
            "meeting-03-slides-memory-takeaway",
            "meeting-04-slides-preference-optimization",
            "meeting-04-slides-rlvr-method",
            "meeting-10-slides-advanced-prover-recap",
        }
        expected_source_only = {
            "official-syllabus-coursework-and-schedule",
            *(f"meeting-{number:02d}-recording-full-session" for number in range(1, 13)),
            "meeting-01-intro-course-identity-and-staff",
            "meeting-01-intro-prior-course-and-agent-frame",
            "meeting-01-intro-course-topic-map",
            "meeting-01-intro-official-coursework-contract",
            "meeting-01-slides-inference-time-reasoning-orientation",
            "meeting-02-slides-learning-to-reason-framing",
            "meeting-02-slides-reasoning-training-foundations",
            "meeting-02-slides-learning-to-reason-synthesis",
            "meeting-03-slides-memory-planning-orientation",
            "meeting-04-slides-open-posttraining-overview",
            "meeting-04-slides-olmo-pretraining",
            "meeting-04-slides-open-recipe-synthesis",
            "meeting-06-slides-web-agent-overview",
            "meeting-06-slides-plan-sequence-learn",
            "meeting-06-slides-proprietary-adjacent-demo",
            "meeting-07-slides-gui-agent-landscape",
            "meeting-07-slides-xgen-video",
            "meeting-07-slides-gens",
            "meeting-07-slides-gui-agent-summary",
            "meeting-08-slides-alphaproof-orientation",
            "meeting-08-slides-riemann-zeta-demo",
            "meeting-08-slides-formal-reasoning-future",
            "meeting-08-reading-03-catalogue-record",
            "meeting-08-reading-04-catalogue-record",
            "meeting-09-slides-autoformalization-capability-race",
            "meeting-10-slides-advanced-proving-frame",
            "meeting-11-slides-abstraction-discovery-frame",
            "meeting-11-slides-visual-concept-discovery",
        }
        expected_excluded = {
            "practice-precioux-discovery-link",
            "meeting-01-intro-course-website-pointer",
            "meeting-01-slides-outline-before-prompting",
            "meeting-01-slides-outline-before-self-consistency",
            "meeting-01-slides-outline-before-reflection",
            "meeting-01-slides-meeting-01-closing",
            "meeting-02-slides-meeting-02-closing",
            "meeting-03-slides-meeting-03-outline-a",
            "meeting-03-slides-meeting-03-outline-b",
            "meeting-03-slides-meeting-03-outline-c",
            "meeting-03-slides-meeting-03-closing",
            "meeting-04-slides-meeting-04-closing",
            "meeting-06-slides-nvidia-confidential-page",
            "meeting-06-slides-meeting-06-closing",
            "meeting-07-slides-meeting-07-outline-a",
            "meeting-07-slides-meeting-07-outline-b",
            "meeting-08-slides-meeting-08-demo-divider",
            "meeting-08-slides-meeting-08-closing",
            "meeting-10-slides-meeting-10-methods-divider",
            "meeting-10-slides-meeting-10-outline",
            "meeting-10-slides-meeting-10-research-divider",
            "meeting-10-slides-meeting-10-closing",
            "meeting-11-slides-meeting-11-divider",
            "meeting-11-slides-meeting-11-closing",
            "meeting-12-slides-meeting-12-outline-a",
            "meeting-12-slides-meeting-12-outline-b",
            "meeting-12-slides-meeting-12-outline-c",
            "meeting-12-slides-meeting-12-closing",
        }
        decisions = self.editorial["coverage"]
        for disposition, expected in (
            ("covered-existing", expected_covered),
            ("source-only", expected_source_only),
            ("excluded", expected_excluded),
        ):
            actual = {
                unit_id for unit_id, decision in decisions.items()
                if decision["disposition"] == disposition
            }
            self.assertEqual(actual, expected, disposition)

    def test_staged_overlay_is_structurally_applied_without_claiming_destination_completion(self) -> None:
        importer = self._load_importer()
        review = importer.semantic_review()
        manifest = importer.build_manifest()
        units = importer.build_units(review)
        baseline_coverage = importer.build_coverage(manifest, units, review)
        baseline_visuals = importer.build_visuals(review)
        coverage, visuals, overlay_sha = importer.apply_editorial_overlay(
            manifest,
            units,
            baseline_coverage,
            baseline_visuals,
            self.editorial,
            validate_destinations=False,
        )
        self.assertRegex(overlay_sha, r"^[a-f0-9]{64}$")
        self.assertEqual(Counter(row["disposition"] for row in coverage["rows"]), Counter({
            "integrated": 137,
            "covered-existing": 6,
            "source-only": 41,
            "excluded": 28,
        }))
        self.assertEqual(Counter(row["disposition"] for row in visuals["rows"]), Counter({
            "integrated": 42,
            "source-only": 92,
            "excluded": 27,
        }))
        generated = importer.build_documents()
        self.assertNotIn("editorial_overlay_sha256", generated["coverage.yml"])
        self.assertTrue(all(
            row["disposition"] in {"source-only", "excluded"}
            for row in generated["coverage.yml"]["rows"]
        ))

    def test_editorial_overlay_fails_closed_on_missing_or_unknown_decisions(self) -> None:
        importer = self._load_importer()
        review = importer.semantic_review()
        manifest = importer.build_manifest()
        units = importer.build_units(review)
        baseline_coverage = importer.build_coverage(manifest, units, review)
        baseline_visuals = importer.build_visuals(review)

        missing = copy.deepcopy(self.editorial)
        missing["coverage"].pop("meeting-01-slides-analogical-reasoning")
        with self.assertRaisesRegex(ValueError, "decide every source unit exactly once"):
            importer.apply_editorial_overlay(
                manifest, units, baseline_coverage, baseline_visuals, missing,
                validate_destinations=False,
            )

        unknown = copy.deepcopy(self.editorial)
        unknown["visuals"]["unknown-visual"] = {
            "disposition": "source-only", "reason": "test", "evidence": "test",
        }
        with self.assertRaisesRegex(ValueError, "decide every visual exactly once"):
            importer.apply_editorial_overlay(
                manifest, units, baseline_coverage, baseline_visuals, unknown,
                validate_destinations=False,
            )

    def test_editorial_overlay_rejects_duplicate_json_keys(self) -> None:
        importer = self._load_importer()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "editorial-map.yml"
            path.write_text('{"schema_version": 1, "coverage": {"x": {}, "x": {}}, "visuals": {}}', "utf-8")
            with self.assertRaisesRegex(RuntimeError, "duplicate JSON key: x"):
                importer.load_editorial_overlay(path)

    def test_confidential_and_proprietary_adjacent_pages_fail_closed(self) -> None:
        importer = self._load_importer()
        review = importer.semantic_review()
        manifest = importer.build_manifest()
        units = importer.build_units(review)
        baseline_coverage = importer.build_coverage(manifest, units, review)
        baseline_visuals = importer.build_visuals(review)

        confidential = copy.deepcopy(self.editorial)
        confidential["visuals"][importer.RESTRICTED_VISUAL]["disposition"] = "source-only"
        with self.assertRaisesRegex(ValueError, "confidential page must remain excluded"):
            importer.apply_editorial_overlay(
                manifest, units, baseline_coverage, baseline_visuals, confidential,
                validate_destinations=False,
            )

        adjacent = copy.deepcopy(self.editorial)
        adjacent["coverage"][importer.RIGHTS_REVIEW_UNIT] = {
            "disposition": "integrated",
            "destination": "future.md",
            "destination_anchor": "future",
        }
        with self.assertRaisesRegex(ValueError, "changed rights record"):
            importer.apply_editorial_overlay(
                manifest, units, baseline_coverage, baseline_visuals, adjacent,
                validate_destinations=False,
            )

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

    def test_reviewed_audit_contract_is_exact_and_hard_pinned(self) -> None:
        importer = self._load_importer()
        contract_path = COURSE_ROOT / "audit-contract.json"
        payload = contract_path.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), importer.AUDIT_CONTRACT_SHA256)
        contract = importer.load_reviewed_audit_contract()
        self.assertIs(contract["generated_by_importer"], False)
        self.assertEqual(contract["deck_count"], 13)
        self.assertEqual(contract["physical_page_count"], 1254)
        self.assertEqual(contract["semantic_section_count"], 161)
        self.assertEqual(
            contract["semantic_review_contract"],
            importer.audit_contract_projection(read_document("semantic-review.json")),
        )

        with tempfile.TemporaryDirectory() as temporary:
            mutated_path = Path(temporary) / "audit-contract.json"
            mutated_path.write_bytes(payload.replace(
                b'"learning-to-reason-framing"',
                b'"collapsed-meeting-02"',
                1,
            ))
            original_path = importer.AUDIT_CONTRACT_PATH
            importer.AUDIT_CONTRACT_PATH = mutated_path
            try:
                with self.assertRaisesRegex(ValueError, "audit contract SHA mismatch"):
                    importer.load_reviewed_audit_contract()
            finally:
                importer.AUDIT_CONTRACT_PATH = original_path

    def test_reviewed_audit_contract_rejects_collapse_after_full_regeneration(self) -> None:
        importer = self._load_importer()
        with tempfile.TemporaryDirectory() as temporary:
            isolated_course = Path(temporary) / "course"
            shutil.copytree(COURSE_ROOT, isolated_course)
            original_paths = {
                "COURSE_ROOT": importer.COURSE_ROOT,
                "LECTURES_ROOT": importer.LECTURES_ROOT,
                "READINGS_ROOT": importer.READINGS_ROOT,
                "METADATA_ROOT": importer.METADATA_ROOT,
                "SEMANTIC_REVIEW_PATH": importer.SEMANTIC_REVIEW_PATH,
                "SYLLABUS_PATH": importer.SYLLABUS_PATH,
            }
            replacement_paths = {
                "COURSE_ROOT": isolated_course,
                "LECTURES_ROOT": isolated_course / "Lectures",
                "READINGS_ROOT": isolated_course / "Readings",
                "METADATA_ROOT": isolated_course / "Metadata",
                "SEMANTIC_REVIEW_PATH": isolated_course / "semantic-review.json",
                "SYLLABUS_PATH": isolated_course / "Metadata/syllabus.html",
            }
            if hasattr(importer, "AUDIT_CONTRACT_PATH"):
                original_paths["AUDIT_CONTRACT_PATH"] = importer.AUDIT_CONTRACT_PATH
                replacement_paths["AUDIT_CONTRACT_PATH"] = isolated_course / "audit-contract.json"
            try:
                for name, path in replacement_paths.items():
                    setattr(importer, name, path)

                contract_before = importer.AUDIT_CONTRACT_PATH.read_bytes()
                importer.regenerate()
                self.assertEqual(importer.AUDIT_CONTRACT_PATH.read_bytes(), contract_before)
                self.assertEqual(hashlib.sha256(contract_before).hexdigest(), importer.AUDIT_CONTRACT_SHA256)

                review = importer.load_document(importer.SEMANTIC_REVIEW_PATH)
                meeting_two = next(deck for deck in review["decks"] if deck["id"] == "meeting-02-slides")
                meeting_two["sections"] = [{
                    "title": "All meeting 2 slides",
                    "semantic_id": "collapsed-meeting-02",
                    "kind": "section",
                    "page_start": 1,
                    "page_end": 106,
                    "visual_kind": "orientation",
                }]
                importer.write_document(importer.SEMANTIC_REVIEW_PATH, review)

                with self.assertRaisesRegex(ValueError, "audit contract"):
                    importer.regenerate()

                manifest = importer.build_manifest()
                units = importer.build_units(review)
                coverage = importer.build_coverage(manifest, units, review)
                visuals = importer.build_visuals(review)
                regenerated = {
                    "source-manifest.yml": manifest,
                    "source-units.yml": units,
                    "coverage.yml": coverage,
                    "visuals.yml": visuals,
                }
                for name, document in regenerated.items():
                    importer.write_document(isolated_course / name, document)
                importer.write_document(
                    isolated_course / "artifact-inventory.json",
                    importer.build_artifact_inventory(regenerated),
                )
                importer.write_document(
                    isolated_course / "snapshot-lock.json",
                    importer.build_snapshot_lock(regenerated),
                )

                self.assertEqual(
                    len([unit for unit in units["units"] if unit["source_object"] == "meeting-02-slides"]),
                    1,
                )
                failures = importer.check()
                self.assertTrue(any("audit contract" in failure.lower() for failure in failures), failures)
            finally:
                for name, path in original_paths.items():
                    setattr(importer, name, path)

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

    def test_shared_registry_verification_dates_cover_the_reviewed_berkeley_entry(self) -> None:
        registries = (
            REPOSITORY_ROOT / "05 Источники/Курсы.md",
            REPOSITORY_ROOT / "05 Источники/Source maps/Единый реестр покрытия источников.md",
        )
        for registry in registries:
            frontmatter = "\n".join(registry.read_text("utf-8").splitlines()[:12])
            self.assertIn("last_verified: 2026-09-04", frontmatter, registry)

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
