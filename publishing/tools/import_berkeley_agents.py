#!/usr/bin/env python3
"""Pin and audit Berkeley Advanced LLM Agents, Spring 2025.

The checked-in PDFs are unchanged official artifacts.  The adjacent page-index
files are deterministic ``pdftotext -layout`` search aids, never substitutes
for the originals.  All ledger rows are rebuilt from ``semantic-review.json``,
which records a direct page-by-page visual and semantic review at meaningful
section scope.  Its exact semantic IDs, titles, kinds, ranges, dispositions,
exclusion evidence, and parent PDF hashes must match the separately reviewed,
hard-pinned ``audit-contract.json``; no generator writes that contract.

``--check`` is deliberately offline.  It independently re-extracts every PDF,
validates the independent completeness contract and exact physical-page
closure, rebuilds all ledgers, and checks the deterministic snapshot lock.
Network access exists only in ``--refresh`` and ``--check-upstream-drift``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COURSE_RELATIVE = Path("05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025")
COURSE_ROOT = REPOSITORY_ROOT / COURSE_RELATIVE
LECTURES_ROOT = COURSE_ROOT / "Lectures"
READINGS_ROOT = COURSE_ROOT / "Readings"
METADATA_ROOT = COURSE_ROOT / "Metadata"
SEMANTIC_REVIEW_PATH = COURSE_ROOT / "semantic-review.json"
AUDIT_CONTRACT_PATH = COURSE_ROOT / "audit-contract.json"
EDITORIAL_MAP_PATH = COURSE_ROOT / "editorial-map.yml"
AUDIT_CONTRACT_SHA256 = "8b3a6f5d9e416b90f9681731a80d30fd069bc6e830a5053aa4acc7457fed4ba4"
SYLLABUS_URL = "https://rdi.berkeley.edu/adv-llm-agents/sp25"
SYLLABUS_PATH = METADATA_ROOT / "syllabus.html"
BASE_COMMIT = "fad4d91373fcd94c18906bfaf209125188eb20cf"
RETRIEVED_AT = "2026-09-04"
COURSE_NAME = "Berkeley Advanced LLM Agents"
OFFERING = "Spring 2025"
COURSE_HUB = f"{COURSE_RELATIVE.as_posix()}/_index.md"
RENDERED_ROUTE = "/sources/courses/berkeley-advanced-llm-agents-spring-2025/"
PRECI0UX_SHA = "16f7b26346ec4b8978199e6d35bcee96b128bd0a"
EXPECTED_READING_DISTRIBUTION = [3, 3, 3, 3, 2, 4, 2, 4, 3, 4, 2, 4]
GENERATED_LEDGER_NAMES = (
    "source-manifest.yml",
    "source-units.yml",
    "coverage.yml",
    "visuals.yml",
)
RIGHTS_EVIDENCE = (
    "User confirmed open educational reuse on 2026-09-04 for local educational "
    "preservation and later attributed textbook reuse; this is a permission record, "
    "not a named license. Page-specific restrictions remain controlling."
)
EXTRACTOR_REVISION = "berkeley-advanced-llm-agents-semantic-audit-v2"
PAGE_INDEX_TOOL = "pdftotext -layout (Poppler 26.04.0)"


@dataclass(frozen=True)
class Deck:
    object_id: str
    filename: str
    title: str
    lecturer: str
    meeting_date: str
    url: str
    sha256: str
    size: int
    pages: int
    artifact_role: str = "official lecture deck"


@dataclass(frozen=True)
class Recording:
    video_id: str
    title: str
    duration_seconds: int


@dataclass(frozen=True)
class Reading:
    key: str
    title: str
    authors: tuple[str, ...]
    year: int
    publication_date: str
    canonical_url: str
    syllabus_url: str
    source_type: str
    source_byline: str | None = None
    organizations: tuple[str, ...] = ()
    contributors: tuple[str, ...] = ()


@dataclass(frozen=True)
class Meeting:
    number: int
    date: str
    title: str
    lecturer: str
    deck_ids: tuple[str, ...]
    recording: Recording
    reading_keys: tuple[str, ...]


DECKS: tuple[Deck, ...] = (
    Deck(
        "meeting-01-intro", "meeting-01-intro.pdf", "Advanced LLM Agents: course introduction",
        "Dawn Song, Xinyun Chen, and Kaiyu Yang", "2025-01-27",
        "https://rdi.berkeley.edu/adv-llm-agents/slides/llm-agents-berkeley-intro-sp25.pdf",
        "23e4671905c456b879bd92b6ca98fcd5b836f9d875aa3b0c6813f5e909a85f47", 1730219, 16,
        "additional official intro deck in meeting 1",
    ),
    Deck(
        "meeting-01-slides", "meeting-01-slides.pdf", "Inference-Time Techniques for LLM Reasoning",
        "Xinyun Chen", "2025-01-27",
        "https://rdi.berkeley.edu/adv-llm-agents/slides/inference_time_techniques_lecture_sp25.pdf",
        "7c17e9d9baf8a3759a016872c17b1f5cce2e4d7691698b2799f6e7a2bd7ff576", 12364433, 74,
    ),
    Deck(
        "meeting-02-slides", "meeting-02-slides.pdf", "Learning to reason with LLMs",
        "Jason Weston", "2025-02-03",
        "https://rdi.berkeley.edu/adv-llm-agents/slides/Jason-Weston-Reasoning-Alignment-Berkeley-Talk.pdf",
        "84b5901763f7fc08d5622c824784d3ca8d304ceaf613a8d4b2ef8529b0ee0be3", 20453927, 106,
    ),
    Deck(
        "meeting-03-slides", "meeting-03-slides.pdf", "On Reasoning, Memory, and Planning of Language Agents",
        "Yu Su", "2025-02-10",
        "https://rdi.berkeley.edu/adv-llm-agents/slides/language_agents_YuSu_Berkeley.pdf",
        "748182d9be6e05589bf0664b4e8c08f55671c8d70e1a64ab031ebcf02fc3fd07", 10016389, 80,
    ),
    Deck(
        "meeting-04-slides", "meeting-04-slides.pdf", "Open Training Recipes for Reasoning in Language Models",
        "Hanna Hajishirzi", "2025-02-24",
        "https://rdi.berkeley.edu/adv-llm-agents/slides/OLMo-Tulu-Reasoning-Hanna.pdf",
        "6a4df80ff062cc6014b3c86d2d08d4256e059b5a9bf12a4926eda89327d37bf7", 22415095, 155,
    ),
    Deck(
        "meeting-05-slides", "meeting-05-slides.pdf", "Coding Agents and AI for Vulnerability Detection",
        "Charles Sutton", "2025-03-03",
        "https://rdi.berkeley.edu/adv-llm-agents/slides/Code%20Agents%20and%20AI%20for%20Vulnerability%20Detection.pdf",
        "c242e0298f5c46adf38af284637209a0986db7fca2b63c626dd29218b5dd809e", 3503777, 55,
    ),
    Deck(
        "meeting-06-slides", "meeting-06-slides.pdf", "Multimodal Autonomous AI Agents",
        "Ruslan Salakhutdinov", "2025-03-10",
        "https://rdi.berkeley.edu/adv-llm-agents/slides/ruslan-multimodal.pdf",
        "62ee4ac4383854062ef2e04242337ccbc34d52aef87f16c3b88b1c797a7220a0", 11619905, 126,
    ),
    Deck(
        "meeting-07-slides", "meeting-07-slides.pdf", "Multimodal Agents – From Perception to Action",
        "Caiming Xiong", "2025-03-17",
        "https://rdi.berkeley.edu/adv-llm-agents/slides/Multimodal_Agent_caiming.pdf",
        "3f12ea7b44c64d43d3c3804596700feacda59a26290ba188935fdd5cdde3847d", 21268799, 106,
    ),
    Deck(
        "meeting-08-slides", "meeting-08-slides.pdf", "AlphaProof: when reinforcement learning meets formal mathematics",
        "Thomas Hubert", "2025-03-31",
        "https://rdi.berkeley.edu/adv-llm-agents/slides/alphaproof.pdf",
        "94bd68c2fdb534d57459a8b96f2a80ada96fd14a58b36626016d4a9f601866ab", 10975911, 112,
    ),
    Deck(
        "meeting-09-slides", "meeting-09-slides.pdf", "Language models for autoformalization and theorem proving",
        "Kaiyu Yang", "2025-04-07",
        "https://rdi.berkeley.edu/adv-llm-agents/slides/mathverification.pdf",
        "2117a841c12dfc6850290f85c87dd9d15cc8d0327d3740fffa5d3277dfb01001", 3112537, 114,
    ),
    Deck(
        "meeting-10-slides", "meeting-10-slides.pdf", "Advanced topics in theorem proving",
        "Sean Welleck", "2025-04-14",
        "https://rdi.berkeley.edu/adv-llm-agents/slides/welleck2025_berkeley_bridging.pdf",
        "1c91d93b766b12ae1ddaaf3dc5575e8b8011b06505f3f7a9404ec13520fec78d", 18176794, 118,
    ),
    Deck(
        "meeting-11-slides", "meeting-11-slides.pdf", "Abstraction and Discovery with Large Language Model Agents",
        "Swarat Chaudhuri", "2025-04-21",
        "https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf",
        "ff4ecc5cacafe5d9fd3253ca28c968e8034466770c1e9d67481a7d51a7f70cc4", 3920144, 93,
    ),
    Deck(
        "meeting-12-slides", "meeting-12-slides.pdf", "Towards building safe and secure agentic AI",
        "Dawn Song", "2025-04-28",
        "https://rdi.berkeley.edu/adv-llm-agents/slides/dawn-agentic-ai.pdf",
        "395a0add9ee6328bc4206a042b90dbc70063fc6a8aa0420ad4483ffe0d2cab6b", 6920712, 99,
    ),
)


ARXIV_METADATA: dict[str, tuple[str, tuple[str, ...], str]] = {
    "1712.01815": ("Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm", ("David Silver", "Thomas Hubert", "Julian Schrittwieser", "Ioannis Antonoglou", "Matthew Lai", "Arthur Guez", "Marc Lanctot", "Laurent Sifre", "Dharshan Kumaran", "Thore Graepel", "Timothy Lillicrap", "Karen Simonyan", "Demis Hassabis"), "2017-12-05"),
    "2205.12615": ("Autoformalization with Large Language Models", ("Yuhuai Wu", "Albert Q. Jiang", "Wenda Li", "Markus N. Rabe", "Charles Staats", "Mateja Jamnik", "Christian Szegedy"), "2022-05-25"),
    "2210.12283": ("Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs", ("Albert Q. Jiang", "Sean Welleck", "Jin Peng Zhou", "Wenda Li", "Jiacheng Liu", "Mateja Jamnik", "Timothée Lacroix", "Yuhuai Wu", "Guillaume Lample"), "2022-10-21"),
    "2304.05128": ("Teaching Large Language Models to Self-Debug", ("Xinyun Chen", "Maxwell Lin", "Nathanael Schärli", "Denny Zhou"), "2023-04-11"),
    "2305.18290": ("Direct Preference Optimization: Your Language Model is Secretly a Reward Model", ("Rafael Rafailov", "Archit Sharma", "Eric Mitchell", "Stefano Ermon", "Christopher D. Manning", "Chelsea Finn"), "2023-05-29"),
    "2306.06070": ("Mind2Web: Towards a Generalist Agent for the Web", ("Xiang Deng", "Yu Gu", "Boyuan Zheng", "Shijie Chen", "Samuel Stevens", "Boshi Wang", "Huan Sun", "Yu Su"), "2023-06-09"),
    "2306.15626": ("LeanDojo: Theorem Proving with Retrieval-Augmented Language Models", ("Kaiyu Yang", "Aidan M. Swope", "Alex Gu", "Rahul Chalamala", "Peiyang Song", "Shixing Yu", "Saad Godil", "Ryan Prenger", "Anima Anandkumar"), "2023-06-27"),
    "2307.13854": ("WebArena: A Realistic Web Environment for Building Autonomous Agents", ("Shuyan Zhou", "Frank F. Xu", "Hao Zhu", "Xuhui Zhou", "Robert Lo", "Abishek Sridhar", "Xianyi Cheng", "Tianyue Ou", "Yonatan Bisk", "Daniel Fried", "Uri Alon", "Graham Neubig"), "2023-07-25"),
    "2309.03409": ("Large Language Models as Optimizers", ("Chengrun Yang", "Xuezhi Wang", "Yifeng Lu", "Hanxiao Liu", "Quoc V. Le", "Denny Zhou", "Xinyun Chen"), "2023-09-07"),
    "2309.11495": ("Chain-of-Verification Reduces Hallucination in Large Language Models", ("Shehzaad Dhuliawala", "Mojtaba Komeili", "Jing Xu", "Roberta Raileanu", "Xian Li", "Asli Celikyilmaz", "Jason Weston"), "2023-09-20"),
    "2310.01798": ("Large Language Models Cannot Self-Correct Reasoning Yet", ("Jie Huang", "Xinyun Chen", "Swaroop Mishra", "Huaixiu Steven Zheng", "Adams Wei Yu", "Xinying Song", "Denny Zhou"), "2023-10-03"),
    "2310.04353": ("An In-Context Learning Agent for Formal Theorem-Proving", ("Amitayush Thakur", "George Tsoukalas", "Yeming Wen", "Jimmy Xin", "Swarat Chaudhuri"), "2023-10-06"),
    "2404.07972": ("OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments", ("Tianbao Xie", "Danyang Zhang", "Jixuan Chen", "Xiaochuan Li", "Siheng Zhao", "Ruisheng Cao", "Toh Jing Hua", "Zhoujun Cheng", "Dongchan Shin", "Fangyu Lei", "Yitao Liu", "Yiheng Xu", "Shuyan Zhou", "Silvio Savarese", "Caiming Xiong", "Victor Zhong", "Tao Yu"), "2024-04-11"),
    "2404.19733": ("Iterative Reasoning Preference Optimization", ("Richard Yuanzhe Pang", "Weizhe Yuan", "Kyunghyun Cho", "He He", "Sainbayar Sukhbaatar", "Jason Weston"), "2024-04-30"),
    "2405.14831": ("HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models", ("Bernal Jiménez Gutiérrez", "Yiheng Shu", "Yu Gu", "Michihiro Yasunaga", "Yu Su"), "2024-05-23"),
    "2405.15071": ("Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization", ("Boshi Wang", "Xiang Yue", "Yu Su", "Huan Sun"), "2024-05-23"),
    "2405.17216": ("Autoformalizing Euclidean Geometry", ("Logan Murphy", "Kaiyu Yang", "Jialiang Sun", "Zhaoyu Li", "Anima Anandkumar", "Xujie Si"), "2024-05-27"),
    "2406.09279": ("Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback", ("Hamish Ivison", "Yizhong Wang", "Jiacheng Liu", "Zeqiu Wu", "Valentina Pyatkin", "Nathan Lambert", "Noah A. Smith", "Yejin Choi", "Hannaneh Hajishirzi"), "2024-06-13"),
    "2407.10040": ("Lean-STaR: Learning to Interleave Thinking and Proving", ("Haohan Lin", "Zhiqing Sun", "Sean Welleck", "Yiming Yang"), "2024-07-14"),
    "2407.12784": ("AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases", ("Zhaorun Chen", "Zhen Xiang", "Chaowei Xiao", "Dawn Song", "Bo Li"), "2024-07-17"),
    "2408.03350": ("miniCTX: Neural Theorem Proving with (Long-)Contexts", ("Jiewen Hu", "Thomas Zhu", "Sean Welleck"), "2024-08-05"),
    "2409.09359": ("Symbolic Regression with a Learned Concept Library", ("Arya Grayeli", "Atharva Sehgal", "Omar Costilla-Reyes", "Miles Cranmer", "Swarat Chaudhuri"), "2024-09-14"),
    "2409.16165": ("EnIGMA: Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities", ("Talor Abramovich", "Meet Udeshi", "Minghao Shao", "Kilian Lieret", "Haoran Xi", "Kimberly Milner", "Sofija Jancheska", "John Yang", "Carlos E. Jimenez", "Farshad Khorrami", "Prashanth Krishnamurthy", "Brendan Dolan-Gavitt", "Muhammad Shafique", "Karthik Narasimhan", "Ramesh Karri", "Ofir Press"), "2024-09-24"),
    "2410.04753": ("ImProver: Agent-Based Automated Proof Optimization", ("Riyaz Ahuja", "Jeremy Avigad", "Prasad Tetali", "Sean Welleck"), "2024-10-07"),
    "2411.06559": ("Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents", ("Yu Gu", "Kai Zhang", "Yuting Ning", "Boyuan Zheng", "Boyu Gou", "Tianci Xue", "Cheng Chang", "Sanjari Srivastava", "Yanan Xie", "Peng Qi", "Huan Sun", "Yu Su"), "2024-11-10"),
    "2411.14199": ("OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs", ("Akari Asai", "Jacqueline He", "Rulin Shao", "Weijia Shi", "Amanpreet Singh", "Joseph Chee Chang", "Kyle Lo", "Luca Soldaini", "Sergey Feldman", "Mike D'arcy", "David Wadden", "Matt Latzke", "Minyang Tian", "Pan Ji", "Shengyan Liu", "Hao Tong", "Bohao Wu", "Yanyu Xiong", "Luke Zettlemoyer", "Graham Neubig", "Dan Weld", "Doug Downey", "Wen-tau Yih", "Pang Wei Koh", "Hannaneh Hajishirzi"), "2024-11-21"),
    "2411.15124": ("Tulu 3: Pushing Frontiers in Open Language Model Post-Training", ("Nathan Lambert", "Jacob Morrison", "Valentina Pyatkin", "Shengyi Huang", "Hamish Ivison", "Faeze Brahman", "Lester James V. Miranda", "Alisa Liu", "Nouha Dziri", "Shane Lyu", "Yuling Gu", "Saumya Malik", "Victoria Graf", "Jena D. Hwang", "Jiangjiang Yang", "Ronan Le Bras", "Oyvind Tafjord", "Chris Wilhelm", "Luca Soldaini", "Noah A. Smith", "Yizhong Wang", "Pradeep Dasigi", "Hannaneh Hajishirzi"), "2024-11-22"),
    "2412.04454": ("Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction", ("Yiheng Xu", "Zekun Wang", "Junli Wang", "Dunjie Lu", "Tianbao Xie", "Amrita Saha", "Doyen Sahoo", "Tao Yu", "Caiming Xiong"), "2024-12-05"),
    "2504.11358": ("DataSentinel: A Game-Theoretic Detection of Prompt Injection Attacks", ("Yupei Liu", "Yuqi Jia", "Jinyuan Jia", "Dawn Song", "Neil Zhenqiang Gong"), "2025-04-15"),
    "2504.11703": ("Progent: Programmable Privilege Control for LLM Agents", ("Tianneng Shi", "Jingxuan He", "Zhun Wang", "Linyu Wu", "Hongwei Li", "Wenbo Guo", "Dawn Song"), "2025-04-16"),
}


def arxiv_reading(identifier: str, syllabus_url: str | None = None) -> Reading:
    title, authors, published = ARXIV_METADATA[identifier]
    return Reading(
        key=f"arxiv-{identifier.replace('.', '-')}",
        title=title,
        authors=authors,
        year=int(published[:4]),
        publication_date=published,
        canonical_url=f"https://arxiv.org/abs/{identifier}v1",
        syllabus_url=syllabus_url or f"https://arxiv.org/abs/{identifier}",
        source_type="arXiv paper metadata; v1 pinned",
    )


SPECIAL_READINGS: tuple[Reading, ...] = (
    Reading(
        "big-sleep",
        "From Naptime to Big Sleep: Using Large Language Models To Catch Vulnerabilities In Real-World Code",
        ("the Big Sleep team",),
        2024,
        "2024-11-01",
        "https://projectzero.google/2024/10/from-naptime-to-big-sleep.html",
        "https://googleprojectzero.blogspot.com/2024/10/from-naptime-to-big-sleep.html",
        "official project article",
        source_byline="the Big Sleep team",
        organizations=("Google Project Zero", "Google DeepMind"),
        contributors=(
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
        ),
    ),
    Reading("visualwebarena", "VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks", ("Jing Yu Koh", "Robert Lo", "Lawrence Jang", "Vikram Duvvur", "Ming Chong Lim", "Po-Yu Huang", "Graham Neubig", "Shuyan Zhou", "Ruslan Salakhutdinov", "Daniel Fried"), 2024, "2024-02-12", "https://jykoh.com/vwa", "https://jykoh.com/vwa", "official project page"),
    Reading("tree-search-agents", "Tree Search for Language Model Agents", ("Jing Yu Koh", "Stephen McAleer", "Daniel Fried", "Ruslan Salakhutdinov"), 2024, "2024-07-01", "https://arxiv.org/abs/2407.01476v1", "https://jykoh.com/search-agents", "official project page with pinned paper version"),
    Reading("deepmind-imo", "AI achieves silver-medal standard solving International Mathematical Olympiad problems", ("AlphaProof team", "AlphaGeometry team"), 2024, "2024-07-25", "https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/", "https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/", "official research article"),
    Reading("future-of-mathematics", "The Future of Mathematics?", ("Kevin Buzzard",), 2019, "2019-10-01", "https://www.youtube.com/watch?v=Dp-mQ3HxgDE", "https://www.youtube.com/watch?v=Dp-mQ3HxgDE", "official Microsoft Research talk recording"),
    Reading("mathematical-library-future", "Building the Mathematical Library of the Future", ("Kevin Hartnett",), 2020, "2020-10-01", "https://www.quantamagazine.org/building-the-mathematical-library-of-the-future-20201001/", "https://www.quantamagazine.org/building-the-mathematical-library-of-the-future-20201001/", "publisher article"),
    Reading("privtrans", "Privtrans: Automatically Partitioning Programs for Privilege Separation", ("David Brumley", "Dawn Song"), 2004, "2004-08-09", "https://dawnsong.io/papers/privtrans.pdf", "https://dawnsong.io/papers/privtrans.pdf", "author-hosted paper link"),
)


READINGS_BY_KEY: dict[str, Reading] = {
    **{reading.key: reading for reading in (arxiv_reading(identifier) for identifier in ARXIV_METADATA)},
    **{reading.key: reading for reading in SPECIAL_READINGS},
}

# Preserve the raw syllabus href where it differs from the canonical v1 locator.
for _identifier, _raw in {
    "2404.07972": "https://arxiv.org/pdf/2404.07972",
    "2412.04454": "https://arxiv.org/pdf/2412.04454",
    "1712.01815": "https://arxiv.org/pdf/1712.01815",
    "2408.03350": "https://www.arxiv.org/pdf/2408.03350",
    "2504.11703": "https://arxiv.org/html/2504.11703v1",
}.items():
    _old = READINGS_BY_KEY[f"arxiv-{_identifier.replace('.', '-')}"]
    READINGS_BY_KEY[_old.key] = Reading(
        _old.key, _old.title, _old.authors, _old.year, _old.publication_date,
        _old.canonical_url, _raw, _old.source_type,
    )


MEETINGS: tuple[Meeting, ...] = (
    Meeting(1, "2025-01-27", "Inference-Time Techniques for LLM Reasoning", "Xinyun Chen", ("meeting-01-intro", "meeting-01-slides"), Recording("g0Dwtf3BH-0", "Adv. LLM Agents MOOC | UC Berkeley Sp25 | Inference-Time Techniques for LLM Reasoning by Xinyun Chen", 4892), ("arxiv-2309-03409", "arxiv-2310-01798", "arxiv-2304-05128")),
    Meeting(2, "2025-02-03", "Learning to reason with LLMs", "Jason Weston", ("meeting-02-slides",), Recording("_MNlLhU33H0", "Adv. LLM Agents MOOC | UC Berkeley CS294-280 Sp25 | Learning to Reason with LLMs by Jason Weston", 4607), ("arxiv-2305-18290", "arxiv-2404-19733", "arxiv-2309-11495")),
    Meeting(3, "2025-02-10", "On Reasoning, Memory, and Planning of Language Agents", "Yu Su", ("meeting-03-slides",), Recording("zvI4UN2_i-w", "Adv. LLM Agents MOOC | UC Berkeley Sp25 | Reasoning, Memory & Planning of Language Agents by Yu Su", 5559), ("arxiv-2405-15071", "arxiv-2405-14831", "arxiv-2411-06559")),
    Meeting(4, "2025-02-24", "Open Training Recipes for Reasoning in Language Models", "Hanna Hajishirzi", ("meeting-04-slides",), Recording("cMiu3A7YBks", "Adv. LLM Agents MOOC | UC Berkeley Sp25 | Open Training Recipes: LLM Reasoning by Hanna Hajishirzi", 4853), ("arxiv-2411-15124", "arxiv-2406-09279", "arxiv-2411-14199")),
    Meeting(5, "2025-03-03", "Coding Agents and AI for Vulnerability Detection", "Charles Sutton", ("meeting-05-slides",), Recording("JCk6qJtaCSU", "Adv. LLM Agents MOOC | UC Berkeley Sp25 | Code Agents & AI Vulnerability Detection by Charles Sutton", 5222), ("arxiv-2409-16165", "big-sleep")),
    Meeting(6, "2025-03-10", "Multimodal Autonomous AI Agents", "Ruslan Salakhutdinov", ("meeting-06-slides",), Recording("RPINOYM12RU", "Adv. LLM Agents MOOC | UC Berkeley Sp25 | Multimodal Autonomous AI Agents by Ruslan Salakhutdinov", 4662), ("arxiv-2306-06070", "arxiv-2307-13854", "visualwebarena", "tree-search-agents")),
    Meeting(7, "2025-03-17", "Multimodal Agents – From Perception to Action", "Caiming Xiong", ("meeting-07-slides",), Recording("n__Tim8K2IY", "Adv. LLM Agents MOOC | UC Berkeley Sp25 | Multimodal Agents – Perception to Action by Caiming Xiong", 5301), ("arxiv-2404-07972", "arxiv-2412-04454")),
    Meeting(8, "2025-03-31", "AlphaProof: when reinforcement learning meets formal mathematics", "Thomas Hubert", ("meeting-08-slides",), Recording("3gaEMscOMAU", "Adv. LLM Agents MOOC | UC Berkeley CS294-280 Sp25 | AlphaProof RL Meets Formal Math by Thomas Hubert", 4448), ("deepmind-imo", "arxiv-1712-01815", "future-of-mathematics", "mathematical-library-future")),
    Meeting(9, "2025-04-07", "Language models for autoformalization and theorem proving", "Kaiyu Yang", ("meeting-09-slides",), Recording("cLhWEyMQ4mQ", "Adv. LLM Agents MOOC | UC Berkeley Sp25 | LMs for Autoformalization+Theorem Proving by Kaiyu Yang", 3127), ("arxiv-2306-15626", "arxiv-2205-12615", "arxiv-2405-17216")),
    Meeting(10, "2025-04-14", "Advanced topics in theorem proving", "Sean Welleck", ("meeting-10-slides",), Recording("Gy5Nm17l9oo", "Adv. LLM Agents MOOC | UC Berkeley CS294-280 Sp25 | Informal+Formal MathReasoning by Sean Welleck", 4332), ("arxiv-2210-12283", "arxiv-2408-03350", "arxiv-2407-10040", "arxiv-2410-04753")),
    Meeting(11, "2025-04-21", "Abstraction and Discovery with Large Language Model Agents", "Swarat Chaudhuri", ("meeting-11-slides",), Recording("IHc0TEMrEdY", "Adv. LLM Agents MOOC | UC Berkeley Sp25 | Abstraction, Discovery w/ LLM Agents by Swarat Chaudhuri", 5258), ("arxiv-2310-04353", "arxiv-2409-09359")),
    Meeting(12, "2025-04-28", "Towards building safe and secure agentic AI", "Dawn Song", ("meeting-12-slides",), Recording("ti6yPE2VPZc", "Adv. LLM Agents MOOC | UC Berkeley CS294-280 Sp25 | Towards Safe & Secure Agentic AI by Dawn Song", 6644), ("privtrans", "arxiv-2504-11358", "arxiv-2407-12784", "arxiv-2504-11703")),
)


DECKS_BY_ID = {deck.object_id: deck for deck in DECKS}


# Implementation helpers follow below.


class SyllabusRows(HTMLParser):
    """Collect hrefs by table row without depending on an HTML framework."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag == "a" and self._row is not None:
            href = dict(attrs).get("href")
            if href:
                self._row.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_bytes(document: dict[str, Any]) -> bytes:
    """Canonical checked-in representation used by deterministic locks."""

    return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_document(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = document_bytes(document)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    temporary.replace(path)


def load_document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON mapping")
    return value


def normalized_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(urllib.parse.unquote(url))
    host = parsed.netloc.lower()
    if host == "www.youtube.com":
        host = "youtube.com"
    return urllib.parse.urlunsplit((parsed.scheme.lower(), host, parsed.path, parsed.query, ""))


def raw_recording_url(meeting: Meeting) -> str:
    return f"https://www.youtube.com/live/{meeting.recording.video_id}"


def parse_syllabus_rows(payload: bytes) -> list[list[str]]:
    parser = SyllabusRows()
    parser.feed(payload.decode("utf-8"))
    return [[normalized_url(href) for href in row] for row in parser.rows]


def is_recording_link(href: str) -> bool:
    parsed = urllib.parse.urlsplit(href)
    return parsed.netloc == "youtube.com" and parsed.path.startswith("/live/")


def is_official_slide_link(href: str) -> bool:
    parsed = urllib.parse.urlsplit(href)
    return (
        parsed.netloc == "rdi.berkeley.edu"
        and parsed.path.startswith("/adv-llm-agents/slides/")
    )


def is_teaching_row(row: list[str]) -> bool:
    """Identify every course row independently of the pinned manifest."""

    return any(is_recording_link(href) or is_official_slide_link(href) for href in row)


def validate_syllabus_inventory(payload: bytes, failures: list[str]) -> None:
    """Assert every bundle member occurs together in its locked official row."""

    try:
        rows = parse_syllabus_rows(payload)
    except Exception as error:  # pragma: no cover - exercised by command failures
        failures.append(f"locked syllabus parse failed: {error}")
        return

    course_rows = [row for row in rows if is_teaching_row(row)]
    if len(course_rows) != len(MEETINGS):
        failures.append(
            f"locked syllabus: expected exactly {len(MEETINGS)} teaching rows, found {len(course_rows)}"
        )

    expected_recordings = {
        normalized_url(raw_recording_url(meeting)): meeting.number
        for meeting in MEETINGS
    }
    for row_number, row in enumerate(course_rows, start=1):
        recordings = [href for href in row if is_recording_link(href)]
        if len(recordings) != 1:
            failures.append(
                f"locked syllabus: unexpected teaching row {row_number}: expected one recording link, found {recordings}"
            )
        elif recordings[0] not in expected_recordings:
            failures.append(
                f"locked syllabus: unexpected teaching row {row_number} absent from manifest: {recordings[0]}"
            )

    for meeting in MEETINGS:
        recording = normalized_url(raw_recording_url(meeting))
        matching = [row for row in rows if recording in row]
        if len(matching) != 1:
            failures.append(
                f"locked syllabus meeting {meeting.number}: expected one row for recording {recording}, found {len(matching)}"
            )
            continue
        row = matching[0]
        expected = [
            recording,
            *(normalized_url(DECKS_BY_ID[deck_id].url) for deck_id in meeting.deck_ids),
            *(normalized_url(READINGS_BY_KEY[key].syllabus_url) for key in meeting.reading_keys),
        ]
        expected_set = set(expected)
        for href in expected:
            if href not in row:
                failures.append(
                    f"locked syllabus meeting {meeting.number}: manifest member absent from official row: {href}"
                )
        for href in row:
            if href not in expected_set:
                failures.append(
                    f"locked syllabus meeting {meeting.number}: unexpected artifact link absent from manifest: {href}"
                )
        if len(row) != len(expected):
            failures.append(
                f"locked syllabus meeting {meeting.number}: expected exactly {len(expected)} artifact links, found {len(row)}"
            )

    if len(course_rows) == len(MEETINGS):
        recording_links = [href for row in course_rows for href in row if "youtube.com/live/" in href]
        deck_links = [href for row in course_rows for href in row if "/adv-llm-agents/slides/" in href]
        expected_readings = {
            normalized_url(READINGS_BY_KEY[key].syllabus_url)
            for meeting in MEETINGS
            for key in meeting.reading_keys
        }
        reading_links = [href for row in course_rows for href in row if href in expected_readings]
        if len(recording_links) != 12:
            failures.append(f"locked syllabus: expected 12 recording links, found {len(recording_links)}")
        if len(deck_links) != 13:
            failures.append(f"locked syllabus: expected 13 official PDF links, found {len(deck_links)}")
        if len(reading_links) != 37:
            failures.append(f"locked syllabus: expected 37 individual reading links, found {len(reading_links)}")


def audit_contract_projection(review: dict[str, Any]) -> dict[str, Any]:
    """Return the exact human-reviewed semantics protected by the audit contract."""

    decks: list[dict[str, Any]] = []
    for reviewed_deck in review.get("decks", []):
        deck_id = reviewed_deck.get("id")
        pinned_deck = DECKS_BY_ID.get(deck_id)
        sections: list[dict[str, Any]] = []
        for section in reviewed_deck.get("sections", []):
            disposition = section.get("disposition", "source-only")
            projected_section = {
                "semantic_id": section.get("semantic_id"),
                "title": section.get("title"),
                "kind": section.get("kind"),
                "visual_kind": section.get("visual_kind"),
                "page_start": section.get("page_start"),
                "page_end": section.get("page_end"),
                "disposition": disposition,
            }
            if disposition == "excluded":
                projected_section["reason"] = section.get("reason")
                projected_section["evidence"] = section.get("evidence")
            sections.append(projected_section)
        decks.append({
            "id": deck_id,
            "source_pdf_sha256": pinned_deck.sha256 if pinned_deck else None,
            "page_count": reviewed_deck.get("page_count"),
            "sections": sections,
        })
    return {
        "reviewed_at": review.get("reviewed_at"),
        "reviewer": review.get("reviewer"),
        "method": review.get("method"),
        "decks": decks,
    }


def load_reviewed_audit_contract() -> dict[str, Any]:
    """Load the manually reviewed completeness anchor; generators never write it."""

    if not AUDIT_CONTRACT_PATH.exists():
        raise ValueError("immutable reviewed audit contract is missing: audit-contract.json")
    actual_sha = sha256_file(AUDIT_CONTRACT_PATH)
    if actual_sha != AUDIT_CONTRACT_SHA256:
        raise ValueError(
            "immutable reviewed audit contract SHA mismatch; re-review every changed range and "
            "explicitly update audit-contract.json plus AUDIT_CONTRACT_SHA256"
        )
    contract = load_document(AUDIT_CONTRACT_PATH)
    failures: list[str] = []
    if contract.get("schema_version") != 1:
        failures.append("schema_version must be 1")
    if contract.get("course") != COURSE_NAME or contract.get("offering") != OFFERING:
        failures.append("course/offering identity mismatch")
    if contract.get("generated_by_importer") is not False:
        failures.append("generated_by_importer must be false")
    if contract.get("deck_count") != len(DECKS):
        failures.append(f"deck_count must be {len(DECKS)}")
    if contract.get("physical_page_count") != sum(deck.pages for deck in DECKS):
        failures.append(f"physical_page_count must be {sum(deck.pages for deck in DECKS)}")
    protected = contract.get("semantic_review_contract")
    if not isinstance(protected, dict):
        failures.append("semantic_review_contract must be a mapping")
    else:
        sections = [
            section
            for deck in protected.get("decks", [])
            for section in deck.get("sections", [])
        ]
        if contract.get("semantic_section_count") != len(sections):
            failures.append("semantic_section_count does not match the protected section inventory")
    if not contract.get("update_workflow"):
        failures.append("update_workflow must document the manual re-review process")
    if failures:
        raise ValueError("immutable reviewed audit contract invalid: " + "; ".join(failures))
    return contract


def semantic_review() -> dict[str, Any]:
    review = load_document(SEMANTIC_REVIEW_PATH)
    failures: list[str] = []
    if review.get("schema_version") != 1:
        failures.append("semantic-review.json: schema_version must be 1")
    decks = review.get("decks")
    if not isinstance(decks, list):
        failures.append("semantic-review.json: decks must be a list")
        decks = []
    ids = [entry.get("id") for entry in decks if isinstance(entry, dict)]
    if ids != [deck.object_id for deck in DECKS]:
        failures.append("semantic-review.json: deck order/inventory differs from the 13 pinned official PDFs")
    seen_semantic_ids: set[str] = set()
    for entry in decks:
        if not isinstance(entry, dict) or entry.get("id") not in DECKS_BY_ID:
            failures.append(f"semantic-review.json: invalid deck entry {entry!r}")
            continue
        deck = DECKS_BY_ID[entry["id"]]
        if entry.get("page_count") != deck.pages:
            failures.append(f"{deck.object_id}: semantic review page_count differs from pinned PDF")
        pages: list[int] = []
        for index, section in enumerate(entry.get("sections", []), start=1):
            if not isinstance(section, dict):
                failures.append(f"{deck.object_id}: section {index} is not a mapping")
                continue
            semantic_id = section.get("semantic_id")
            if not isinstance(semantic_id, str) or not semantic_id:
                failures.append(f"{deck.object_id}: section {index} has no semantic_id")
            elif semantic_id in seen_semantic_ids:
                failures.append(f"semantic-review.json: duplicate semantic_id {semantic_id}")
            else:
                seen_semantic_ids.add(semantic_id)
            if not section.get("title") or section.get("kind") not in {
                "section", "mechanism", "derivation", "experiment", "worked-example",
                "failure-mode", "figure", "table", "code-trace", "visual-sequence", "administrative",
            }:
                failures.append(f"{deck.object_id}: invalid semantic section {semantic_id}")
            start, end = section.get("page_start"), section.get("page_end")
            if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
                failures.append(f"{deck.object_id}: invalid page range for {semantic_id}")
                continue
            pages.extend(range(start, end + 1))
            if section.get("kind") == "administrative":
                if section.get("disposition") != "excluded" or not section.get("reason") or not section.get("evidence"):
                    failures.append(f"{deck.object_id}: administrative section {semantic_id} lacks a reasoned exclusion")
        expected = list(range(1, deck.pages + 1))
        if pages != expected:
            missing = sorted(set(expected) - set(pages))
            duplicate = sorted(page for page in set(pages) if pages.count(page) > 1)
            failures.append(
                f"{deck.object_id}: semantic review must close exact physical pages; missing={missing}, duplicate={duplicate}"
            )
    if failures:
        raise ValueError("\n".join(failures))
    contract = load_reviewed_audit_contract()
    protected = contract["semantic_review_contract"]
    actual = audit_contract_projection(review)
    if actual != protected:
        raise ValueError(
            "semantic-review.json differs from the immutable reviewed audit contract; "
            "regeneration cannot redefine reviewed semantic completeness"
        )
    return review


def reading_rows() -> Iterable[tuple[Meeting, int, str, Reading]]:
    for meeting in MEETINGS:
        for order, key in enumerate(meeting.reading_keys, start=1):
            yield meeting, order, f"meeting-{meeting.number:02d}-reading-{order:02d}", READINGS_BY_KEY[key]


def reading_note(meeting: Meeting, order: int, reading: Reading) -> bytes:
    authors = "; ".join(reading.authors)
    source_metadata = ""
    if reading.source_byline:
        source_metadata += f"- Source byline: {reading.source_byline}\n"
    if reading.organizations:
        source_metadata += f"- Organizations: {'; '.join(reading.organizations)}\n"
    if reading.contributors:
        source_metadata += f"- Contributors: {'; '.join(reading.contributors)}\n"
    if reading.source_byline or reading.organizations or reading.contributors:
        source_metadata += f"- Source metadata evidence: {reading.canonical_url}\n"
    text = (
        f"# {reading.title}\n\n"
        f"Metadata-only catalogue record for meeting {meeting.number}: {meeting.title}. "
        "No copyrighted paper body is mirrored.\n\n"
        f"- Authors: {authors}\n"
        f"{source_metadata}"
        f"- Publication date: {reading.publication_date}\n"
        f"- Canonical pinned/project URL: {reading.canonical_url}\n"
        f"- Original syllabus URL: {reading.syllabus_url}\n"
        f"- Source type: {reading.source_type}\n"
        f"- Syllabus order: {order}\n"
        f"- Relationship: primary source for technical claims; the lecture deck is a secondary explanation.\n"
        f"- Retrieved: {RETRIEVED_AT}\n"
    )
    return text.encode("utf-8")


def reading_path(object_id: str) -> Path:
    return READINGS_ROOT / f"{object_id}.md"


def generate_reading_notes(write: bool, failures: list[str] | None = None) -> None:
    for meeting, order, object_id, reading in reading_rows():
        expected = reading_note(meeting, order, reading)
        path = reading_path(object_id)
        if write:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_bytes() != expected:
                path.write_bytes(expected)
        elif not path.exists():
            assert failures is not None
            failures.append(f"missing generated reading catalogue note: {path.relative_to(COURSE_ROOT)}")
        elif path.read_bytes() != expected:
            assert failures is not None
            failures.append(f"deterministic reading metadata mismatch: {path.relative_to(COURSE_ROOT)}")


def pdf_page_count(path: Path) -> int:
    completed = subprocess.run(
        ["pdfinfo", str(path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    match = re.search(rb"^Pages:\s+(\d+)\s*$", completed.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError(f"pdfinfo did not report Pages for {path}")
    return int(match.group(1))


def page_index_bytes(path: Path, expected_pages: int) -> bytes:
    completed = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    decoded = completed.stdout.decode("utf-8", errors="replace")
    chunks = decoded.split("\f")
    while chunks and chunks[-1].strip() == "":
        chunks.pop()
    if len(chunks) != expected_pages:
        raise RuntimeError(
            f"pdftotext physical page split mismatch for {path.name}: expected {expected_pages}, got {len(chunks)}"
        )
    blocks: list[str] = []
    for page, chunk in enumerate(chunks, start=1):
        content = chunk.rstrip()
        blocks.append(f"=== PHYSICAL PAGE {page} OF {expected_pages} ===\n{content}".rstrip())
    return ("\n\n".join(blocks) + "\n").encode("utf-8")


def page_index_path(deck: Deck) -> Path:
    return LECTURES_ROOT / f"{deck.object_id}.pages.txt"


def generate_page_indexes(write: bool, failures: list[str] | None = None) -> None:
    for deck in DECKS:
        pdf_path = LECTURES_ROOT / deck.filename
        if not pdf_path.exists():
            if failures is not None:
                failures.append(f"missing official PDF: Lectures/{deck.filename}")
            continue
        try:
            expected = page_index_bytes(pdf_path, deck.pages)
        except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
            if failures is None:
                raise
            failures.append(f"{deck.object_id}: page-index extraction failed: {error}")
            continue
        target = page_index_path(deck)
        if write:
            if not target.exists() or target.read_bytes() != expected:
                target.write_bytes(expected)
        elif not target.exists():
            assert failures is not None
            failures.append(f"missing page-index extraction: Lectures/{target.name}")
        elif target.read_bytes() != expected:
            assert failures is not None
            failures.append(f"{deck.object_id}: deterministic pdftotext page-index mismatch")


def object_common(
    *, object_id: str, kind: str, authority: str, author: str, meeting_date: str,
    title: str, canonical_url: str, revision: str, local_path: str | None,
    mime: str, size: int, page_count: int | None, rights_status: str,
    rights_evidence: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": object_id,
        "kind": kind,
        "source_authority": authority,
        "verification_status": "verified",
        "lecturer_or_author": author,
        "meeting_date": meeting_date,
        "title": title,
        "canonical_url": canonical_url,
        "revision_or_checksum": revision,
    }
    if local_path is not None:
        result["local_path"] = local_path
    result.update({"mime": mime, "bytes": size})
    if page_count is not None:
        result["page_count"] = page_count
    result.update({
        "language": "en",
        "rights_status": rights_status,
        "rights_evidence": rights_evidence,
        "retrieved_at": RETRIEVED_AT,
    })
    return result


def build_manifest() -> dict[str, Any]:
    syllabus_payload = SYLLABUS_PATH.read_bytes()
    objects: list[dict[str, Any]] = []
    syllabus = object_common(
        object_id="official-syllabus",
        kind="assignment",
        authority="official-course",
        author="Berkeley RDI course staff",
        meeting_date=MEETINGS[0].date,
        title="Advanced LLM Agents, Spring 2025 official syllabus",
        canonical_url=SYLLABUS_URL,
        revision=sha256_bytes(syllabus_payload),
        local_path="Metadata/syllabus.html",
        mime="text/html",
        size=len(syllabus_payload),
        page_count=1,
        rights_status="permission-recorded",
        rights_evidence=RIGHTS_EVIDENCE,
    )
    syllabus["artifact_role"] = "authoritative schedule, bundle membership, and coursework provenance"
    objects.append(syllabus)

    reading_ids_by_meeting: dict[int, list[str]] = {meeting.number: [] for meeting in MEETINGS}
    reading_records = list(reading_rows())
    readings_for_meeting: dict[int, list[tuple[int, str, Reading]]] = {meeting.number: [] for meeting in MEETINGS}
    for meeting, order, object_id, reading in reading_records:
        readings_for_meeting[meeting.number].append((order, object_id, reading))

    bundles: list[dict[str, Any]] = []
    for meeting in MEETINGS:
        recording_revision = sha256_bytes(
            f"{meeting.recording.video_id}\n{meeting.recording.title}\n{meeting.recording.duration_seconds}\n".encode("utf-8")
        )
        recording_id = f"meeting-{meeting.number:02d}-recording"
        recording = object_common(
            object_id=recording_id,
            kind="video",
            authority="official-course",
            author=meeting.lecturer,
            meeting_date=meeting.date,
            title=meeting.recording.title,
            canonical_url=f"https://www.youtube.com/watch?v={meeting.recording.video_id}",
            revision=recording_revision,
            local_path=None,
            mime="video/youtube",
            size=0,
            page_count=None,
            rights_status="permission-recorded",
            rights_evidence=RIGHTS_EVIDENCE,
        )
        recording.update({
            "syllabus_url": raw_recording_url(meeting),
            "channel": "Berkeley RDI",
            "channel_id": "UCB67PxhB5LAWEbI4etQS7aw",
            "channel_handle": "@BerkeleyRDI",
            "channel_url": "https://www.youtube.com/@BerkeleyRDI",
            "playlist_id": "PLS01nW3RtgorL3AW8REU9nGkzhvtn6Egn",
            "playlist_url": "https://www.youtube.com/playlist?list=PLS01nW3RtgorL3AW8REU9nGkzhvtn6Egn",
            "video_id": meeting.recording.video_id,
            "video_metadata": {
                "duration_seconds": meeting.recording.duration_seconds,
                "frame_rate": 1,
                "frame_count": meeting.recording.duration_seconds,
                "frame_rate_basis": "one-second metadata audit index; source encoding is not mirrored",
            },
            "metadata_retrieved_at": RETRIEVED_AT,
            "metadata_evidence_url": f"https://www.youtube.com/watch?v={meeting.recording.video_id}",
            "metadata_evidence": "Official syllabus recording link plus official YouTube videoDetails title/channel/duration metadata.",
        })
        objects.append(recording)

        for deck_id in meeting.deck_ids:
            deck = DECKS_BY_ID[deck_id]
            index_path = page_index_path(deck)
            deck_object = object_common(
                object_id=deck.object_id,
                kind="pdf",
                authority="official-course",
                author=deck.lecturer,
                meeting_date=deck.meeting_date,
                title=deck.title,
                canonical_url=deck.url,
                revision=deck.sha256,
                local_path=f"Lectures/{deck.filename}",
                mime="application/pdf",
                size=deck.size,
                page_count=deck.pages,
                rights_status="permission-recorded",
                rights_evidence=RIGHTS_EVIDENCE,
            )
            deck_object.update({
                "artifact_role": deck.artifact_role,
                "text_index_path": f"Lectures/{index_path.name}",
                "text_index_sha256": sha256_file(index_path),
                "text_index_tool": PAGE_INDEX_TOOL,
                "physical_page_closure": "complete in semantic-review.json and visuals.yml",
            })
            if deck.object_id == "meeting-06-slides":
                deck_object["rights_overrides"] = [{
                    "physical_page": 117,
                    "marking": "NVIDIA CONFIDENTIAL. DO NOT DISTRIBUTE.",
                    "disposition": "excluded",
                    "rights_scope": "do-not-reuse; preserve unchanged official archive only",
                    "precedence": "overrides the general permission record for this page",
                }]
            objects.append(deck_object)

        meeting_reading_ids: list[str] = []
        for order, object_id, reading in readings_for_meeting[meeting.number]:
            note_path = reading_path(object_id)
            note_payload = note_path.read_bytes()
            reading_object = object_common(
                object_id=object_id,
                kind="assignment",
                authority="primary-paper",
                author="; ".join(reading.authors),
                meeting_date=meeting.date,
                title=reading.title,
                canonical_url=reading.canonical_url,
                revision=sha256_bytes(note_payload),
                local_path=f"Readings/{note_path.name}",
                mime="text/markdown",
                size=len(note_payload),
                page_count=1,
                rights_status="link-only",
                rights_evidence="Metadata and a canonical link are preserved; the copyrighted work body is not mirrored.",
            )
            reading_object.update({
                "catalogue_kind": "reading",
                "authors": list(reading.authors),
                "year": reading.year,
                "publication_date": reading.publication_date,
                "canonical_project_url": reading.canonical_url,
                "syllabus_url": reading.syllabus_url,
                "syllabus_order": order,
                "source_type": reading.source_type,
                "relationship_to_deck": "primary source for technical claims",
            })
            if reading.source_byline:
                reading_object["source_byline"] = reading.source_byline
            if reading.organizations:
                reading_object["organizations"] = list(reading.organizations)
            if reading.contributors:
                reading_object["contributors"] = list(reading.contributors)
            if reading.source_byline or reading.organizations or reading.contributors:
                reading_object["source_metadata_evidence_url"] = reading.canonical_url
            objects.append(reading_object)
            meeting_reading_ids.append(object_id)
        reading_ids_by_meeting[meeting.number] = meeting_reading_ids

        bundles.append({
            "id": f"meeting-{meeting.number:02d}",
            "meeting_number": meeting.number,
            "meeting_date": meeting.date,
            "title": meeting.title,
            "lecturer": meeting.lecturer,
            "recording": recording_id,
            "decks": list(meeting.deck_ids),
            "readings": meeting_reading_ids,
        })

    mirror = object_common(
        object_id="practice-precioux-discovery",
        kind="assignment",
        authority="third-party-mirror",
        author="Precioux repository contributors",
        meeting_date=MEETINGS[0].date,
        title="Advanced Large Language Model Agents student mirror",
        canonical_url=f"https://github.com/Precioux/Advanced-Large-Language-Model-Agents/tree/{PRECI0UX_SHA}",
        revision=PRECI0UX_SHA,
        local_path=None,
        mime="text/html",
        size=0,
        page_count=1,
        rights_status="link-only",
        rights_evidence="Third-party discovery link only; no content is mirrored and it is not evidence of an official lab contract.",
    )
    mirror["verification_status"] = "unverified"
    mirror["artifact_role"] = "discovery lead only"
    objects.append(mirror)

    return {
        "schema_version": 1,
        "course": COURSE_NAME,
        "offering": OFFERING,
        "integration_base_commit": BASE_COMMIT,
        "retrieved_at": RETRIEVED_AT,
        "syllabus_url": SYLLABUS_URL,
        "editorial_status": "integration active; destinations validated",
        "bundle_count": 12,
        "official_pdf_count": 13,
        "reading_count": 37,
        "meeting_bundles": bundles,
        "practice_provenance": {
            "official_public_status": "lab and project confirmed; no verified official lab artifact exposed",
            "official_evidence": "The locked official syllabus and the 16-page Intro deck describe a lab, grading, final project, and project timeline, but the public syllabus row set contains no verified official lab artifact link.",
            "drive_artifact_status": "not pinned; unavailable for use",
            "drive_pin_requirements": "Before use, record the stable Google Drive file ID, immutable export checksum, and retrieval date.",
            "third_party_mirror": "practice-precioux-discovery",
            "later_exercise_label": "adaptation inspired by the course",
        },
        "rights_boundary": "permission-recorded is a user permission record, not a named license; page-specific restrictions override it",
        "objects": objects,
    }


def build_units(review: dict[str, Any]) -> dict[str, Any]:
    units: list[dict[str, Any]] = [
        {
            "id": "official-syllabus-coursework-and-schedule",
            "source_object": "official-syllabus",
            "order": 1,
            "source_location": "Metadata/syllabus.html#course-schedule-and-coursework",
            "kind": "section",
            "title": "Official schedule, reading lists, lab, and project contract",
            "semantic_id": "official-syllabus-coursework-and-schedule",
            "page": 1,
        }
    ]
    review_by_id = {entry["id"]: entry for entry in review["decks"]}
    for meeting in MEETINGS:
        recording_id = f"meeting-{meeting.number:02d}-recording"
        units.append({
            "id": f"{recording_id}-full-session",
            "source_object": recording_id,
            "order": 1,
            "source_location": f"https://www.youtube.com/watch?v={meeting.recording.video_id}#t=0s",
            "kind": "segment",
            "title": f"Full official recording: {meeting.title}",
            "start_seconds": 0,
            "end_seconds": meeting.recording.duration_seconds,
        })
        for deck_id in meeting.deck_ids:
            for order, section in enumerate(review_by_id[deck_id]["sections"], start=1):
                start, end = section["page_start"], section["page_end"]
                units.append({
                    "id": f"{deck_id}-{section['semantic_id']}",
                    "source_object": deck_id,
                    "order": order,
                    "source_location": f"Lectures/{deck_id}.pages.txt#physical-pages={start}-{end}",
                    "kind": section["kind"],
                    "title": section["title"],
                    "semantic_id": section["semantic_id"],
                    "page_start": start,
                    "page_end": end,
                })
        for order, object_id, reading in (
            (order, object_id, reading)
            for row_meeting, order, object_id, reading in reading_rows()
            if row_meeting.number == meeting.number
        ):
            units.append({
                "id": f"{object_id}-catalogue-record",
                "source_object": object_id,
                "order": 1,
                "source_location": f"Readings/{object_id}.md#metadata",
                "kind": "section",
                "title": reading.title,
                "semantic_id": f"{object_id}-catalogue-record",
                "page": 1,
            })
    units.append({
        "id": "practice-precioux-discovery-link",
        "source_object": "practice-precioux-discovery",
        "order": 1,
        "source_location": f"https://github.com/Precioux/Advanced-Large-Language-Model-Agents/tree/{PRECI0UX_SHA}#discovery-only",
        "kind": "section",
        "title": "Third-party practice discovery lead",
        "semantic_id": "practice-precioux-discovery-link",
        "page": 1,
    })
    return {
        "schema_version": 1,
        "review_method": review["method"],
        "semantic_review_sha256": sha256_file(SEMANTIC_REVIEW_PATH),
        "units": units,
    }


def unit_to_section(review: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for deck in review["decks"]:
        for section in deck["sections"]:
            result[f"{deck['id']}-{section['semantic_id']}"] = section
    return result


def build_coverage(manifest: dict[str, Any], units_document: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    sections = unit_to_section(review)
    object_to_bundle: dict[str, dict[str, Any]] = {}
    for bundle in manifest["meeting_bundles"]:
        for object_id in [bundle["recording"], *bundle["decks"], *bundle["readings"]]:
            object_to_bundle[object_id] = bundle

    rows: list[dict[str, Any]] = []
    for unit in units_document["units"]:
        object_id = unit["source_object"]
        row: dict[str, Any] = {
            "id": f"coverage-{unit['id']}",
            "source_object": object_id,
            "source_unit": unit["id"],
            "source_location": unit["source_location"],
            "kind": unit["kind"],
            "title": unit["title"],
            "primary_sources": [object_id],
        }
        if object_id == "practice-precioux-discovery":
            row.update({
                "disposition": "excluded",
                "reason": "discovery lead only",
                "evidence": "A student-maintained third-party mirror cannot substantiate an official lab contract.",
            })
        elif unit["id"] in sections and sections[unit["id"]].get("disposition") == "excluded":
            section = sections[unit["id"]]
            row.update({
                "disposition": "excluded",
                "reason": section["reason"],
                "evidence": section["evidence"],
            })
        else:
            if object_id == "official-syllabus":
                anchor = "practice-provenance"
                reason = "Authoritative source-only evidence for schedule, bundle membership, and the public practice boundary."
            else:
                bundle = object_to_bundle[object_id]
                anchor = bundle["id"]
                if object_id in bundle["readings"]:
                    row["secondary_sources"] = bundle["decks"]
                    reason = "Primary reading is catalogued individually; the official deck remains a secondary explanation."
                elif object_id == bundle["recording"]:
                    reason = "Official recording metadata is pinned for the source baseline; no media body is mirrored."
                else:
                    reason = "Reviewed semantic unit is preserved at exact physical-page scope; editorial integration is pending."
            row.update({
                "disposition": "source-only",
                "destination": COURSE_HUB,
                "destination_anchor": anchor,
                "reason": reason,
            })
        rows.append(row)
    return {
        "schema_version": 1,
        "semantic_review_sha256": sha256_file(SEMANTIC_REVIEW_PATH),
        "rows": rows,
    }


def visual_question(section: dict[str, Any]) -> str:
    if section.get("disposition") == "excluded":
        return f"Why is the reviewed page range for {section['title']} excluded from reuse?"
    return f"What does the reviewed sequence establish about {section['title']}?"


def build_visuals(review: dict[str, Any]) -> dict[str, Any]:
    review_hash = sha256_file(SEMANTIC_REVIEW_PATH)
    rows: list[dict[str, Any]] = []
    for reviewed_deck in review["decks"]:
        deck = DECKS_BY_ID[reviewed_deck["id"]]
        for section in reviewed_deck["sections"]:
            start, end = section["page_start"], section["page_end"]
            pages = list(range(start, end + 1))
            unit_id = f"{deck.object_id}-{section['semantic_id']}"
            row: dict[str, Any] = {
                "id": f"{deck.object_id}-visual-{section['semantic_id']}",
                "semantic_id": section["semantic_id"],
                "visual_kind": section["visual_kind"],
                "source_object": deck.object_id,
                "source_units": [unit_id],
                "source_location": f"Lectures/{deck.object_id}.pages.txt#physical-pages={start}-{end}",
                "source_pages": pages,
                "question": visual_question(section),
                "sequence_members": [
                    {
                        "order": order,
                        "source_location": f"Lectures/{deck.filename}#page={page}",
                    }
                    for order, page in enumerate(pages, start=1)
                ],
                "attribution": f"{deck.lecturer}, {deck.title}, Berkeley Advanced LLM Agents, Spring 2025",
                "rights_status": "permission-recorded",
                "rights_holder": "Berkeley course source authors and identified upstream asset owners",
                "rights_scope": "Local educational preservation and later attributed textbook reuse, subject to page-specific restrictions",
                "license_identifier": "User permission record (not a license)",
                "license_url": SYLLABUS_URL,
                "rights_evidence": RIGHTS_EVIDENCE,
                "parent_sha256": deck.sha256,
                "extractor_sha256": review_hash,
                "extractor_revision": EXTRACTOR_REVISION,
                "extraction_tool": f"Codex-assisted ordered contact-sheet review plus {PAGE_INDEX_TOOL}",
                "rendered_route": RENDERED_ROUTE,
                "desktop_evidence": "pending editorial integration; exact source pages visually inspected",
                "narrow_evidence": "pending editorial integration; exact source pages visually inspected",
                "reviewer": review["reviewer"],
                "checked_at": review["reviewed_at"],
            }
            if section.get("disposition") == "excluded":
                row.update({
                    "disposition": "excluded",
                    "reason": section["reason"],
                    "evidence": section["evidence"],
                })
                if section["semantic_id"] == "nvidia-confidential-page":
                    row.update({
                        "rights_status": "unknown",
                        "rights_holder": "NVIDIA and any identified upstream asset owners",
                        "rights_scope": "do-not-reuse; preserve unchanged official archive only",
                        "rights_evidence": section["evidence"],
                    })
                    row.pop("license_identifier")
                    row.pop("license_url")
            else:
                row.update({
                    "disposition": "source-only",
                    "destination": COURSE_HUB,
                    "destination_anchor": f"meeting-{int(deck.object_id.split('-')[1]):02d}",
                    "reason": "Reviewed visual remains in the source layer until an attributed editorial destination is approved.",
                })
            rows.append(row)
    return {
        "schema_version": 1,
        "audit_method": review["method"],
        "extraction_provenance": {
            "semantic_review": "semantic-review.json",
            "semantic_review_sha256": review_hash,
            "extractor_revision": EXTRACTOR_REVISION,
            "page_index_tool": PAGE_INDEX_TOOL,
            "parent_artifacts": "13 unchanged official PDFs",
        },
        "raster_evidence_summary": "All 1,254 physical pages were inspected in ordered contact sheets; every row retains exact page membership and parent provenance.",
        "rows": rows,
    }


COVERAGE_EDITORIAL_DISPOSITIONS = {
    "integrated",
    "covered-existing",
    "source-only",
    "excluded",
}
VISUAL_EDITORIAL_DISPOSITIONS = COVERAGE_EDITORIAL_DISPOSITIONS
EDITORIAL_TOP_LEVEL_FIELDS = {"schema_version", "reviewed_objects", "coverage", "visuals"}
COVERAGE_EDITORIAL_FIELDS = {
    "disposition",
    "destination",
    "destination_anchor",
    "reason",
    "evidence",
}
VISUAL_EDITORIAL_FIELDS = COVERAGE_EDITORIAL_FIELDS | {
    "local_files",
    "transformation",
    "caption",
    "rendered_route",
    "desktop_evidence",
    "narrow_evidence",
    "reviewer",
    "checked_at",
}
RESTRICTED_UNIT = "meeting-06-slides-nvidia-confidential-page"
RESTRICTED_VISUAL = "meeting-06-slides-visual-nvidia-confidential-page"
RIGHTS_REVIEW_UNIT = "meeting-06-slides-proprietary-adjacent-demo"
RIGHTS_REVIEW_VISUAL = "meeting-06-slides-visual-proprietary-adjacent-demo"


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_editorial_overlay(path: Path = EDITORIAL_MAP_PATH) -> dict[str, Any]:
    try:
        label = path.relative_to(REPOSITORY_ROOT)
    except ValueError:
        label = path
    try:
        value = json.loads(path.read_text("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"{label}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}: expected mapping")
    return value


def editorial_overlay_sha256(overlay: dict[str, Any]) -> str:
    return sha256_bytes(document_bytes(overlay))


def editorial_disposition_counts(overlay: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {
        section: dict(sorted(Counter(
            decision["disposition"] for decision in overlay[section].values()
        ).items()))
        for section in ("coverage", "visuals")
    }


def markdown_anchor(text: str) -> str:
    value = text.casefold().strip()
    value = re.sub(r"[^\w\- ]", "", value, flags=re.UNICODE)
    return re.sub(r"[ _]+", "-", value).strip("-")


def destination_source_unit_ids(text: str) -> set[str]:
    if not text.startswith("---\n"):
        return set()
    end = text.find("\n---", 4)
    if end == -1:
        return set()
    frontmatter = text[4:end]
    lines = frontmatter.splitlines()
    values: set[str] = set()
    for index, line in enumerate(lines):
        match = re.fullmatch(r"source_unit_id:\s*(.*)", line)
        if match is None:
            continue
        inline = match.group(1).strip()
        if inline:
            if inline.startswith("[") and inline.endswith("]"):
                inline = inline[1:-1]
            values.update(
                item.strip().strip("'\"")
                for item in inline.split(",")
                if item.strip()
            )
            continue
        for following in lines[index + 1 :]:
            item = re.fullmatch(r"\s+-\s+(.+?)\s*", following)
            if item is not None:
                values.add(item.group(1).strip().strip("'\""))
                continue
            if following.startswith((" ", "\t")) or not following.strip():
                continue
            break
    return values


def destination_has_anchor(text: str, anchor: str) -> bool:
    quoted = re.escape(anchor)
    if re.search(rf"<(?:a|span)\b[^>]*(?:id|name)=[\"']{quoted}[\"']", text):
        return True
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", text, flags=re.MULTILINE):
        if markdown_anchor(re.sub(r"\s+#+\s*$", "", heading)) == anchor:
            return True
    return False


def resolve_repository_file(repository_root: Path, relative_path: str, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    root = repository_root.resolve()
    destination = (root / relative_path).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"{label} escapes repository root: {relative_path}")
    if not destination.is_file():
        raise ValueError(f"{label} does not exist: {relative_path}")
    return destination


def validate_destination(
    repository_root: Path,
    decision_id: str,
    decision: dict[str, Any],
    reciprocal_ids: set[str],
) -> None:
    destination_value = decision.get("destination")
    anchor = decision.get("destination_anchor")
    if not isinstance(destination_value, str) or not destination_value.strip():
        raise ValueError(f"{decision_id}: integrated decision lacks destination")
    if not isinstance(anchor, str) or not anchor.strip():
        raise ValueError(f"{decision_id}: integrated decision lacks destination_anchor")
    destination = resolve_repository_file(
        repository_root,
        destination_value,
        f"{decision_id}: destination",
    )
    text = destination.read_text("utf-8")
    if not destination_has_anchor(text, anchor):
        raise ValueError(
            f"{decision_id}: destination anchor does not exist: {destination_value}#{anchor}"
        )
    actual_source_ids = destination_source_unit_ids(text)
    missing = sorted(reciprocal_ids - actual_source_ids)
    if missing:
        raise ValueError(
            f"{decision_id}: destination lacks reciprocal source_unit_id values: {missing}"
        )


def destination_image_targets(text: str) -> list[str]:
    targets = [
        target.split("|", 1)[0].split("#", 1)[0].strip()
        for target in re.findall(r"!\[\[([^\]\n]+)\]\]", text)
    ]
    targets.extend(
        target.strip().removeprefix("<").removesuffix(">")
        for target in re.findall(r"!\[[^\]\n]*\]\(([^)\n]+)\)", text)
    )
    return targets


def destination_embeds_file(
    repository_root: Path,
    destination: Path,
    text: str,
    relative_path: str,
) -> bool:
    repository_root = repository_root.resolve()
    asset = (repository_root / relative_path).resolve()
    normalized_relative = Path(relative_path).as_posix().lstrip("./")
    for raw_target in destination_image_targets(text):
        target = urllib.parse.unquote(raw_target).strip()
        normalized_target = target.replace("\\", "/").lstrip("./")
        if normalized_target == normalized_relative or normalized_target.endswith(f"/{normalized_relative}"):
            return True
        if target.startswith(("http://", "https://", "data:")):
            continue
        if (destination.parent / target).resolve() == asset:
            return True
    return False


def validate_visual_files(
    repository_root: Path,
    visual_id: str,
    decision: dict[str, Any],
) -> None:
    destination = resolve_repository_file(
        repository_root,
        decision["destination"],
        f"{visual_id}: destination",
    )
    text = destination.read_text("utf-8")
    for index, relative_path in enumerate(decision["local_files"]):
        resolve_repository_file(
            repository_root,
            relative_path,
            f"{visual_id}: local_files[{index}]",
        )
        if not destination_embeds_file(repository_root, destination, text, relative_path):
            raise ValueError(
                f"{visual_id}: local_files[{index}] is not embedded in destination: "
                f"{decision['destination']}"
            )


def _validate_restricted_decisions(
    coverage_overlay: dict[str, Any],
    visual_overlay: dict[str, Any],
) -> None:
    restricted_unit = coverage_overlay.get(RESTRICTED_UNIT, {})
    restricted_visual = visual_overlay.get(RESTRICTED_VISUAL, {})
    for decision_id, decision in (
        (RESTRICTED_UNIT, restricted_unit),
        (RESTRICTED_VISUAL, restricted_visual),
    ):
        if decision.get("disposition") != "excluded":
            raise ValueError(f"{decision_id}: source-marked confidential page must remain excluded")
        evidence = f"{decision.get('reason', '')} {decision.get('evidence', '')}".casefold()
        if "do-not-reuse" not in evidence or "confidential" not in evidence:
            raise ValueError(f"{decision_id}: exclusion must retain confidential do-not-reuse evidence")

    for decision_id, decision in (
        (RIGHTS_REVIEW_UNIT, coverage_overlay.get(RIGHTS_REVIEW_UNIT, {})),
        (RIGHTS_REVIEW_VISUAL, visual_overlay.get(RIGHTS_REVIEW_VISUAL, {})),
    ):
        if decision.get("disposition") != "source-only":
            raise ValueError(f"{decision_id}: pages 118-121 require a changed rights record before integration")
        evidence = f"{decision.get('reason', '')} {decision.get('evidence', '')}".casefold()
        if "rights review" not in evidence:
            raise ValueError(f"{decision_id}: source-only decision must retain the pending rights review")


def apply_editorial_overlay(
    manifest: dict[str, Any],
    source_units: dict[str, Any],
    baseline_coverage: dict[str, Any],
    baseline_visuals: dict[str, Any],
    overlay: dict[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    validate_destinations: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    unknown_top = sorted(set(overlay) - EDITORIAL_TOP_LEVEL_FIELDS)
    if unknown_top:
        raise ValueError(f"unsupported editorial overlay field: {unknown_top[0]}")
    if overlay.get("schema_version") != 1:
        raise ValueError("editorial overlay schema_version must be 1")

    object_ids = {
        item.get("id")
        for item in manifest.get("objects", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    reviewed_objects = overlay.get("reviewed_objects")
    if not isinstance(reviewed_objects, list) or not all(isinstance(item, str) for item in reviewed_objects):
        raise ValueError("editorial overlay reviewed_objects must be a list of ids")
    if len(reviewed_objects) != len(set(reviewed_objects)):
        raise ValueError("editorial overlay reviewed_objects contains duplicates")
    if set(reviewed_objects) != object_ids:
        missing = sorted(object_ids - set(reviewed_objects))
        unknown = sorted(set(reviewed_objects) - object_ids)
        detail = f"missing={missing[:1]}, unknown={unknown[:1]}"
        raise ValueError(f"editorial overlay must review every source object: {detail}")

    unit_rows = source_units.get("units")
    coverage_rows = baseline_coverage.get("rows")
    visual_rows = baseline_visuals.get("rows")
    if not isinstance(unit_rows, list) or not isinstance(coverage_rows, list):
        raise ValueError("source units and baseline coverage must contain row lists")
    if not isinstance(visual_rows, list):
        raise ValueError("baseline visuals must contain a row list")
    units_by_id = {row["id"]: row for row in unit_rows}
    visuals_by_id = {row["id"]: row for row in visual_rows}
    coverage_overlay = overlay.get("coverage")
    visual_overlay = overlay.get("visuals")
    if not isinstance(coverage_overlay, dict):
        raise ValueError("editorial overlay coverage must be a mapping keyed by source unit id")
    if not isinstance(visual_overlay, dict):
        raise ValueError("editorial overlay visuals must be a mapping keyed by visual id")
    if set(coverage_overlay) != set(units_by_id):
        missing = sorted(set(units_by_id) - set(coverage_overlay))
        unknown = sorted(set(coverage_overlay) - set(units_by_id))
        raise ValueError(f"editorial overlay must decide every source unit exactly once: missing={missing[:1]}, unknown={unknown[:1]}")
    if set(visual_overlay) != set(visuals_by_id):
        missing = sorted(set(visuals_by_id) - set(visual_overlay))
        unknown = sorted(set(visual_overlay) - set(visuals_by_id))
        raise ValueError(f"editorial overlay must decide every visual exactly once: missing={missing[:1]}, unknown={unknown[:1]}")
    _validate_restricted_decisions(coverage_overlay, visual_overlay)

    coverage = copy.deepcopy(baseline_coverage)
    visuals = copy.deepcopy(baseline_visuals)
    output_coverage = {row["source_unit"]: row for row in coverage["rows"]}
    output_visuals = {row["id"]: row for row in visuals["rows"]}

    for unit_id, decision in coverage_overlay.items():
        if not isinstance(decision, dict):
            raise ValueError(f"coverage decision must be a mapping: {unit_id}")
        unknown = sorted(set(decision) - COVERAGE_EDITORIAL_FIELDS)
        if unknown:
            raise ValueError(f"unsupported coverage editorial field: {unknown[0]}")
        disposition = decision.get("disposition")
        if disposition not in COVERAGE_EDITORIAL_DISPOSITIONS:
            raise ValueError(f"{unit_id}: invalid disposition: {disposition!r}")
        if units_by_id[unit_id].get("kind") == "administrative" and disposition not in {"source-only", "excluded"}:
            raise ValueError(f"administrative source semantics may only be source-only or excluded: {unit_id}")
        row = output_coverage[unit_id]
        row.pop("reason", None)
        row.pop("evidence", None)
        if disposition == "excluded":
            if "destination" in decision or "destination_anchor" in decision:
                raise ValueError(f"{unit_id}: excluded decision cannot claim a destination")
            row.pop("destination", None)
            row.pop("destination_anchor", None)
        row.update(decision)
        if disposition in {"source-only", "excluded"}:
            for field in ("reason", "evidence"):
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{unit_id}: {disposition} decision lacks {field}")
            if disposition == "source-only" and row.get("destination") != COURSE_HUB:
                raise ValueError(f"{unit_id}: source-only decision must remain routed to the course hub")
        else:
            if validate_destinations:
                validate_destination(repository_root, unit_id, row, {unit_id})
            else:
                for field in ("destination", "destination_anchor"):
                    value = row.get(field)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"{unit_id}: integrated decision lacks {field}")

    for visual_id, decision in visual_overlay.items():
        if not isinstance(decision, dict):
            raise ValueError(f"visual decision must be a mapping: {visual_id}")
        unknown = sorted(set(decision) - VISUAL_EDITORIAL_FIELDS)
        if unknown:
            raise ValueError(f"unsupported visual editorial field: {unknown[0]}")
        disposition = decision.get("disposition")
        if disposition not in VISUAL_EDITORIAL_DISPOSITIONS:
            raise ValueError(f"{visual_id}: invalid disposition: {disposition!r}")
        row = output_visuals[visual_id]
        row.pop("reason", None)
        row.pop("evidence", None)
        if disposition == "excluded":
            if "destination" in decision or "destination_anchor" in decision:
                raise ValueError(f"{visual_id}: excluded decision cannot claim a destination")
            row.pop("destination", None)
            row.pop("destination_anchor", None)
        row.update(decision)
        if disposition in {"source-only", "excluded"}:
            for field in ("reason", "evidence"):
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{visual_id}: {disposition} decision lacks {field}")
            if disposition == "source-only" and row.get("destination") != COURSE_HUB:
                raise ValueError(f"{visual_id}: source-only decision must remain routed to the course hub")
        else:
            reciprocal = {item for item in row.get("source_units", []) if isinstance(item, str)}
            linked_decisions = [coverage_overlay[unit_id] for unit_id in sorted(reciprocal)]
            if not linked_decisions or any(
                item.get("disposition") not in {"integrated", "covered-existing"}
                for item in linked_decisions
            ):
                raise ValueError(f"{visual_id}: integrated visual must link integrated source units")
            destinations = {
                (item.get("destination"), item.get("destination_anchor"))
                for item in linked_decisions
            }
            if destinations != {(row.get("destination"), row.get("destination_anchor"))}:
                raise ValueError(f"{visual_id}: visual and source-unit destinations disagree")
            required_fields = (
                "transformation",
                "caption",
                "rendered_route",
                "desktop_evidence",
                "narrow_evidence",
                "reviewer",
                "checked_at",
            )
            for field in required_fields:
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{visual_id}: integrated visual lacks {field}")
            local_files = row.get("local_files")
            if (
                not isinstance(local_files, list)
                or not local_files
                or not all(isinstance(path, str) and path.strip() for path in local_files)
            ):
                raise ValueError(f"{visual_id}: integrated visual lacks non-empty local_files")
            if validate_destinations:
                for field in ("transformation", "rendered_route", "desktop_evidence", "narrow_evidence", "reviewer", "checked_at"):
                    if "pending" in row[field].casefold() or "planned" in row[field].casefold():
                        raise ValueError(f"{visual_id}: final visual evidence is still pending in {field}")
                if not row["rendered_route"].startswith("/"):
                    raise ValueError(f"{visual_id}: rendered_route must be an absolute site route")
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["checked_at"]) is None:
                    raise ValueError(f"{visual_id}: checked_at must be an ISO date")
                validate_destination(repository_root, visual_id, row, reciprocal)
                validate_visual_files(repository_root, visual_id, row)

    integrated_local_files = [
        local_file
        for row in output_visuals.values()
        if row.get("disposition") in {"integrated", "covered-existing"}
        for local_file in row["local_files"]
    ]
    if len(integrated_local_files) != len(set(integrated_local_files)):
        raise ValueError("integrated visuals must not share a local_files entry")

    overlay_sha = editorial_overlay_sha256(overlay)
    coverage["editorial_overlay_sha256"] = overlay_sha
    visuals["editorial_overlay_sha256"] = overlay_sha
    return coverage, visuals, overlay_sha


def build_documents() -> dict[str, dict[str, Any]]:
    review = semantic_review()
    manifest = build_manifest()
    units = build_units(review)
    baseline_coverage = build_coverage(manifest, units, review)
    baseline_visuals = build_visuals(review)

    closure_failures: list[str] = []
    validate_pdf_page_closure(manifest, units, baseline_coverage, closure_failures)
    validate_visual_page_closure(manifest, units, baseline_visuals, closure_failures)
    if closure_failures:
        raise RuntimeError(
            "Berkeley source-semantic closure failed before editorial overlay:\n- "
            + "\n- ".join(closure_failures)
        )

    coverage, visuals, _ = apply_editorial_overlay(
        manifest,
        units,
        baseline_coverage,
        baseline_visuals,
        load_editorial_overlay(),
    )
    return {
        "source-manifest.yml": manifest,
        "source-units.yml": units,
        "coverage.yml": coverage,
        "visuals.yml": visuals,
    }


def validate_manifest_inventory(manifest: dict[str, Any], failures: list[str]) -> None:
    bundles = manifest.get("meeting_bundles", [])
    objects = manifest.get("objects", [])
    if len(bundles) != 12:
        failures.append(f"manifest: expected 12 meeting bundles, found {len(bundles)}")
        return
    distribution = [len(bundle.get("readings", [])) for bundle in bundles]
    if distribution != EXPECTED_READING_DISTRIBUTION:
        failures.append(
            f"manifest: reading distribution mismatch: expected {EXPECTED_READING_DISTRIBUTION}, found {distribution}"
        )
    if sum(len(bundle.get("decks", [])) for bundle in bundles) != 13:
        failures.append("manifest: expected 13 official PDFs across the 12 bundles")
    first_decks = bundles[0].get("decks", []) if bundles else []
    if first_decks != ["meeting-01-intro", "meeting-01-slides"]:
        failures.append("manifest: Jan 27 must retain the additional 16-page Intro deck")
    object_ids = [row.get("id") for row in objects]
    if len(object_ids) != len(set(object_ids)):
        failures.append("manifest: duplicate source object IDs")
    expected_reading_ids = [object_id for _, _, object_id, _ in reading_rows()]
    actual_reading_ids = [row["id"] for row in objects if row.get("catalogue_kind") == "reading"]
    if actual_reading_ids != expected_reading_ids:
        failures.append("manifest: individual reading object inventory/order differs from the 37-item syllabus contract")
    by_id = {row.get("id"): row for row in objects}
    for bundle in bundles:
        for object_id in [bundle.get("recording"), *bundle.get("decks", []), *bundle.get("readings", [])]:
            if object_id not in by_id:
                failures.append(f"manifest: bundle refers to missing object {object_id}")
    for deck in DECKS:
        row = by_id.get(deck.object_id)
        if not row:
            failures.append(f"manifest: missing official deck object {deck.object_id}")
            continue
        for field, expected in (("page_count", deck.pages), ("bytes", deck.size), ("revision_or_checksum", deck.sha256)):
            if row.get(field) != expected:
                failures.append(f"manifest: {deck.object_id}.{field} differs from pinned artifact")
    for meeting in MEETINGS:
        row = by_id.get(f"meeting-{meeting.number:02d}-recording")
        if not row:
            failures.append(f"manifest: missing recording metadata for meeting {meeting.number}")
            continue
        metadata = row.get("video_metadata", {})
        if row.get("video_id") != meeting.recording.video_id or row.get("title") != meeting.recording.title:
            failures.append(f"manifest: meeting {meeting.number} recording identity mismatch")
        if metadata.get("duration_seconds") != meeting.recording.duration_seconds:
            failures.append(f"manifest: meeting {meeting.number} recording duration mismatch")
        if not row.get("channel") or not row.get("channel_handle") or not row.get("video_id"):
            failures.append(f"manifest: meeting {meeting.number} recording lacks verified channel/video metadata")


def validate_pdf_page_closure(
    manifest: dict[str, Any], units_document: dict[str, Any], coverage_document: dict[str, Any], failures: list[str]
) -> None:
    """Validate semantic (not merely syntactic) closure for every official PDF."""

    objects = {row["id"]: row for row in manifest.get("objects", [])}
    bundles = manifest.get("meeting_bundles", [])
    deck_ids = [deck_id for bundle in bundles for deck_id in bundle.get("decks", [])]
    units_by_deck: dict[str, list[dict[str, Any]]] = {deck_id: [] for deck_id in deck_ids}
    for unit in units_document.get("units", []):
        if unit.get("source_object") in units_by_deck:
            units_by_deck[unit["source_object"]].append(unit)
    coverage_by_unit = {row.get("source_unit"): row for row in coverage_document.get("rows", [])}
    forbidden = {"page", "heading", "rendered-text", "rendered-link"}
    for deck_id in deck_ids:
        page_count = objects.get(deck_id, {}).get("page_count")
        if not isinstance(page_count, int):
            failures.append(f"{deck_id}: no pinned physical page_count")
            continue
        seen: list[int] = []
        expected_order = 1
        for unit in units_by_deck[deck_id]:
            if unit.get("order") != expected_order:
                failures.append(f"{deck_id}: semantic unit order is not contiguous at {unit.get('id')}")
            expected_order += 1
            if unit.get("kind") in forbidden:
                failures.append(f"{deck_id}: non-semantic proxy unit kind {unit.get('kind')} is forbidden")
            start = unit.get("page", unit.get("page_start"))
            end = unit.get("page", unit.get("page_end"))
            if not isinstance(start, int) or not isinstance(end, int) or end < start:
                failures.append(f"{deck_id}: invalid semantic range for {unit.get('id')}")
                continue
            seen.extend(range(start, end + 1))
            coverage = coverage_by_unit.get(unit.get("id"))
            if not coverage:
                failures.append(f"{deck_id}: missing coverage for semantic unit {unit.get('id')}")
            elif unit.get("kind") == "administrative":
                if coverage.get("disposition") != "excluded" or not coverage.get("reason") or not coverage.get("evidence"):
                    failures.append(f"{deck_id}: administrative unit lacks explicit reasoned exclusion: {unit.get('id')}")
        expected_pages = list(range(1, page_count + 1))
        missing = sorted(set(expected_pages) - set(seen))
        duplicates = sorted(page for page in set(seen) if seen.count(page) > 1)
        outside = sorted(set(seen) - set(expected_pages))
        if missing:
            failures.append(f"{deck_id}: semantic page closure missing pages {missing}")
        if duplicates:
            failures.append(f"{deck_id}: semantic page closure overlaps pages {duplicates}")
        if outside:
            failures.append(f"{deck_id}: semantic page closure exceeds physical pages {outside}")
        if not missing and not duplicates and not outside and seen != expected_pages:
            failures.append(f"{deck_id}: semantic page closure is not in physical order")


def validate_visual_page_closure(
    manifest: dict[str, Any], units_document: dict[str, Any], visuals_document: dict[str, Any], failures: list[str]
) -> None:
    objects = {row["id"]: row for row in manifest.get("objects", [])}
    unit_by_id = {row["id"]: row for row in units_document.get("units", [])}
    deck_ids = [deck for bundle in manifest.get("meeting_bundles", []) for deck in bundle.get("decks", [])]
    rows_by_deck: dict[str, list[dict[str, Any]]] = {deck: [] for deck in deck_ids}
    for row in visuals_document.get("rows", []):
        if row.get("source_object") in rows_by_deck:
            rows_by_deck[row["source_object"]].append(row)
    for deck_id in deck_ids:
        seen: list[int] = []
        for row in rows_by_deck[deck_id]:
            pages = row.get("source_pages", [])
            if not isinstance(pages, list) or not all(isinstance(page, int) for page in pages):
                failures.append(f"{deck_id}: visual row {row.get('id')} has invalid source_pages")
                continue
            seen.extend(pages)
            linked = row.get("source_units", [])
            if not linked:
                failures.append(f"{deck_id}: visual row {row.get('id')} has no source-unit provenance")
            for unit_id in linked:
                if unit_by_id.get(unit_id, {}).get("source_object") != deck_id:
                    failures.append(f"{deck_id}: visual row {row.get('id')} links a foreign/missing source unit")
            if row.get("parent_sha256") != objects[deck_id].get("revision_or_checksum"):
                failures.append(f"{deck_id}: visual row {row.get('id')} parent SHA mismatch")
            members = row.get("sequence_members", [])
            if len(members) != len(pages) or [member.get("order") for member in members] != list(range(1, len(pages) + 1)):
                failures.append(f"{deck_id}: visual row {row.get('id')} has incomplete ordered membership")
        expected = list(range(1, objects[deck_id]["page_count"] + 1))
        if seen != expected:
            missing = sorted(set(expected) - set(seen))
            duplicates = sorted(page for page in set(seen) if seen.count(page) > 1)
            failures.append(f"{deck_id}: visual page closure mismatch; missing={missing}, duplicate={duplicates}")

    required = {
        "yu-su-agent-first-vs-llm-first",
        "yu-su-hipporag-memory-sequence",
        "yu-su-world-model-planning-sequence",
        "charles-sutton-vulnerability-discovery-loop",
        "kaiyu-yang-lean-theorem-proving-pipeline",
        "swarat-chaudhuri-lasr-concept-library-sequence",
        "dawn-song-agentic-threat-model-sequence",
        "dawn-song-privilege-control-sequence",
    }
    rows_by_semantic = {row.get("semantic_id"): row for row in visuals_document.get("rows", [])}
    for semantic_id in sorted(required):
        row = rows_by_semantic.get(semantic_id)
        if not row or len(row.get("source_pages", [])) < 2:
            failures.append(f"visual ledger: required real multi-member sequence missing: {semantic_id}")


def source_file_entries() -> list[dict[str, Any]]:
    paths: list[tuple[str, Path]] = [
        ("Metadata/syllabus.html", SYLLABUS_PATH),
        ("semantic-review.json", SEMANTIC_REVIEW_PATH),
        ("audit-contract.json", AUDIT_CONTRACT_PATH),
        ("publishing/tools/import_berkeley_agents.py", Path(__file__).resolve()),
    ]
    for deck in DECKS:
        paths.extend([
            (f"Lectures/{deck.filename}", LECTURES_ROOT / deck.filename),
            (f"Lectures/{deck.object_id}.pages.txt", page_index_path(deck)),
        ])
    for _, _, object_id, _ in reading_rows():
        paths.append((f"Readings/{object_id}.md", reading_path(object_id)))
    return [
        {"path": label, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for label, path in paths
    ]


def ledger_record_count(name: str, document: dict[str, Any]) -> int:
    key = "objects" if name == "source-manifest.yml" else ("units" if name == "source-units.yml" else "rows")
    return len(document[key])


def build_snapshot_lock(documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "course": COURSE_NAME,
        "offering": OFFERING,
        "retrieved_at": RETRIEVED_AT,
        "integration_base_commit": BASE_COMMIT,
        "upstream": {
            "syllabus_url": SYLLABUS_URL,
            "syllabus_sha256": sha256_file(SYLLABUS_PATH),
            "official_pdf_count": 13,
            "official_pdf_pages": sum(deck.pages for deck in DECKS),
        },
        "generated_audit": {
            "semantic_review_sha256": sha256_file(SEMANTIC_REVIEW_PATH),
            "audit_contract_sha256": sha256_file(AUDIT_CONTRACT_PATH),
            "editorial_overlay_sha256": editorial_overlay_sha256(load_editorial_overlay()),
            "importer_sha256": sha256_file(Path(__file__).resolve()),
            "extractor_revision": EXTRACTOR_REVISION,
            "page_index_tool": PAGE_INDEX_TOOL,
            "ledgers": {
                name: {
                    "sha256": sha256_bytes(document_bytes(document)),
                    "records": ledger_record_count(name, document),
                }
                for name, document in documents.items()
            },
        },
        "source_files": source_file_entries(),
        "invariants": {
            "meeting_bundles": 12,
            "official_pdfs": 13,
            "physical_pdf_pages": 1254,
            "recordings": 12,
            "individual_readings": 37,
            "reading_distribution": EXPECTED_READING_DISTRIBUTION,
            "editorial_overlay_status": "active-destination-validated",
            "editorial_counts": editorial_disposition_counts(load_editorial_overlay()),
        },
    }


def build_artifact_inventory(documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    audit_contract = load_reviewed_audit_contract()
    manifest = documents["source-manifest.yml"]
    units = documents["source-units.yml"]
    coverage = documents["coverage.yml"]
    visuals = documents["visuals.yml"]
    return {
        "schema_version": 1,
        "course": COURSE_NAME,
        "offering": OFFERING,
        "status": "inventory complete; editorial overlay active",
        "audit_contract": {
            "local_path": "audit-contract.json",
            "sha256": AUDIT_CONTRACT_SHA256,
            "generated_by_importer": False,
            "semantic_section_count": audit_contract["semantic_section_count"],
        },
        "counts": {
            "meeting_bundles": len(manifest["meeting_bundles"]),
            "official_pdfs": sum(len(bundle["decks"]) for bundle in manifest["meeting_bundles"]),
            "physical_pdf_pages": sum(deck.pages for deck in DECKS),
            "recordings": len(manifest["meeting_bundles"]),
            "individual_readings": sum(len(bundle["readings"]) for bundle in manifest["meeting_bundles"]),
            "source_objects": len(manifest["objects"]),
            "semantic_units": len(units["units"]),
            "coverage_rows": len(coverage["rows"]),
            "visual_rows": len(visuals["rows"]),
            "coverage_by_disposition": dict(sorted(
                Counter(row["disposition"] for row in coverage["rows"]).items()
            )),
            "visuals_by_disposition": dict(sorted(
                Counter(row["disposition"] for row in visuals["rows"]).items()
            )),
        },
        "reading_distribution": [len(bundle["readings"]) for bundle in manifest["meeting_bundles"]],
        "editorial_counts": editorial_disposition_counts(load_editorial_overlay()),
        "decks": [
            {
                "id": deck.object_id,
                "meeting_date": deck.meeting_date,
                "pages": deck.pages,
                "bytes": deck.size,
                "sha256": deck.sha256,
                "role": deck.artifact_role,
            }
            for deck in DECKS
        ],
        "restriction_flags": [
            {
                "source_object": "meeting-06-slides",
                "physical_page": 117,
                "disposition": "excluded",
                "rights_scope": "do-not-reuse; preserve unchanged official archive only",
            }
        ],
    }


def validate_generated_documents(
    actual: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]],
    snapshot_lock: dict[str, Any],
    failures: list[str],
) -> None:
    audits = snapshot_lock.get("generated_audit", {}).get("ledgers", {})
    for name in GENERATED_LEDGER_NAMES:
        actual_document = actual.get(name)
        expected_document = expected.get(name)
        if actual_document is None or expected_document is None:
            failures.append(f"{name}: deterministic extraction mismatch (missing document)")
            continue
        if document_bytes(actual_document) != document_bytes(expected_document):
            failures.append(f"{name}: deterministic extraction mismatch against independent rebuild")
        audit = audits.get(name, {})
        actual_sha = sha256_bytes(document_bytes(actual_document))
        if audit.get("sha256") != actual_sha:
            failures.append(f"{name}: generated lock SHA mismatch")
        if audit.get("records") != ledger_record_count(name, actual_document):
            failures.append(f"{name}: generated lock record-count mismatch")
    expected_overlay_sha = editorial_overlay_sha256(load_editorial_overlay())
    locked_overlay_sha = snapshot_lock.get("generated_audit", {}).get("editorial_overlay_sha256")
    if locked_overlay_sha != expected_overlay_sha:
        failures.append("snapshot lock editorial overlay SHA differs from the reviewed overlay")


def validate_local_artifacts(manifest: dict[str, Any], failures: list[str]) -> None:
    if not SYLLABUS_PATH.exists():
        failures.append("missing locked official syllabus: Metadata/syllabus.html")
    else:
        validate_syllabus_inventory(SYLLABUS_PATH.read_bytes(), failures)
    for deck in DECKS:
        path = LECTURES_ROOT / deck.filename
        if not path.exists():
            failures.append(f"missing official PDF: Lectures/{deck.filename}")
            continue
        if path.stat().st_size != deck.size:
            failures.append(f"{deck.object_id}: PDF byte-size mismatch")
        if sha256_file(path) != deck.sha256:
            failures.append(f"{deck.object_id}: PDF SHA-256 mismatch")
        try:
            pages = pdf_page_count(path)
        except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
            failures.append(f"{deck.object_id}: pdfinfo failed: {error}")
        else:
            if pages != deck.pages:
                failures.append(f"{deck.object_id}: physical page count mismatch: expected {deck.pages}, found {pages}")
    generate_reading_notes(write=False, failures=failures)
    generate_page_indexes(write=False, failures=failures)
    validate_manifest_inventory(manifest, failures)


def regenerate() -> dict[str, dict[str, Any]]:
    semantic_review()
    generate_reading_notes(write=True)
    generate_page_indexes(write=True)
    documents = build_documents()
    for name, document in documents.items():
        write_document(COURSE_ROOT / name, document)
    write_document(COURSE_ROOT / "artifact-inventory.json", build_artifact_inventory(documents))
    write_document(COURSE_ROOT / "snapshot-lock.json", build_snapshot_lock(documents))
    return documents


def opener() -> urllib.request.OpenerDirector:
    # The workspace can carry unrelated proxy variables; official course fetches
    # are direct and intentionally isolated from them.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch(url: str) -> bytes:
    quoted = urllib.parse.quote(url, safe=":/?=&%")
    request = urllib.request.Request(
        quoted,
        headers={"User-Agent": "Berkeley-course-source-audit/1.0 (+local educational archive)"},
    )
    with opener().open(request, timeout=90) as response:
        return response.read()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    temporary.replace(path)


def refresh() -> None:
    syllabus = fetch(SYLLABUS_URL)
    failures: list[str] = []
    validate_syllabus_inventory(syllabus, failures)
    if failures:
        raise RuntimeError("refusing to pin an incomplete/drifted syllabus:\n" + "\n".join(failures))
    atomic_write(SYLLABUS_PATH, syllabus)
    for deck in DECKS:
        payload = fetch(deck.url)
        if len(payload) != deck.size or sha256_bytes(payload) != deck.sha256:
            raise RuntimeError(
                f"{deck.object_id}: live official PDF differs from reviewed pin; "
                "update the source audit explicitly instead of silently replacing it"
            )
        atomic_write(LECTURES_ROOT / deck.filename, payload)
    regenerate()


def check_upstream_drift() -> list[str]:
    failures: list[str] = []
    try:
        syllabus = fetch(SYLLABUS_URL)
    except (OSError, urllib.error.URLError) as error:
        return [f"live syllabus fetch failed: {error}"]
    validate_syllabus_inventory(syllabus, failures)
    if SYLLABUS_PATH.exists() and sha256_bytes(syllabus) != sha256_file(SYLLABUS_PATH):
        failures.append("live official syllabus SHA differs from the locked snapshot")
    for deck in DECKS:
        try:
            payload = fetch(deck.url)
        except (OSError, urllib.error.URLError) as error:
            failures.append(f"{deck.object_id}: live PDF fetch failed: {error}")
            continue
        if len(payload) != deck.size or sha256_bytes(payload) != deck.sha256:
            failures.append(f"{deck.object_id}: live official PDF differs from the reviewed pin")
    return failures


def check() -> list[str]:
    failures: list[str] = []
    if not EDITORIAL_MAP_PATH.exists():
        failures.append("missing non-generated editorial overlay: editorial-map.yml")
    try:
        review = semantic_review()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [f"semantic review validation failed: {error}"]

    actual: dict[str, dict[str, Any]] = {}
    for name in GENERATED_LEDGER_NAMES:
        path = COURSE_ROOT / name
        if not path.exists():
            failures.append(f"missing generated ledger: {name}")
            continue
        try:
            actual[name] = load_document(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"{name}: parse failed: {error}")
    lock_path = COURSE_ROOT / "snapshot-lock.json"
    if not lock_path.exists():
        failures.append("missing snapshot-lock.json")
        return failures
    try:
        snapshot_lock = load_document(lock_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        failures.append(f"snapshot-lock.json: parse failed: {error}")
        return failures

    if "source-manifest.yml" in actual:
        validate_local_artifacts(actual["source-manifest.yml"], failures)
    if len(actual) != len(GENERATED_LEDGER_NAMES):
        return failures
    try:
        expected = build_documents()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        failures.append(f"independent ledger rebuild failed: {error}")
        return failures
    validate_generated_documents(actual, expected, snapshot_lock, failures)
    validate_pdf_page_closure(actual["source-manifest.yml"], actual["source-units.yml"], actual["coverage.yml"], failures)
    validate_visual_page_closure(actual["source-manifest.yml"], actual["source-units.yml"], actual["visuals.yml"], failures)
    expected_lock = build_snapshot_lock(expected)
    if document_bytes(snapshot_lock) != document_bytes(expected_lock):
        failures.append("snapshot-lock.json: deterministic source/extraction lock mismatch")
    inventory_path = COURSE_ROOT / "artifact-inventory.json"
    if not inventory_path.exists():
        failures.append("missing artifact-inventory.json")
    else:
        expected_inventory = build_artifact_inventory(expected)
        try:
            actual_inventory = load_document(inventory_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"artifact-inventory.json: parse failed: {error}")
        else:
            if document_bytes(actual_inventory) != document_bytes(expected_inventory):
                failures.append("artifact-inventory.json: deterministic inventory mismatch")
    return failures


def print_failures(label: str, failures: list[str]) -> int:
    if not failures:
        print(f"{label}: OK")
        return 0
    print(f"{label}: FAILED", file=sys.stderr)
    for failure in failures:
        print(f"- {failure}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--refresh", action="store_true", help="download and pin official syllabus/PDFs, then rebuild ledgers")
    modes.add_argument("--regenerate-ledgers", action="store_true", help="offline deterministic rebuild from pinned artifacts")
    modes.add_argument("--check", action="store_true", help="offline independent artifact/extraction/closure check")
    modes.add_argument("--check-upstream-drift", action="store_true", help="networked comparison against official live sources")
    args = parser.parse_args(argv)
    try:
        if args.refresh:
            refresh()
            print("Berkeley Advanced LLM Agents refresh: OK (12 bundles, 13 PDFs, 37 readings)")
            return 0
        if args.regenerate_ledgers:
            regenerate()
            print("Berkeley Advanced LLM Agents ledger regeneration: OK")
            return 0
        if args.check_upstream_drift:
            return print_failures("Berkeley Advanced LLM Agents upstream drift check", check_upstream_drift())
        return print_failures("Berkeley Advanced LLM Agents offline source check", check())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, urllib.error.URLError) as error:
        print(f"Berkeley Advanced LLM Agents importer: FAILED\n- {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
