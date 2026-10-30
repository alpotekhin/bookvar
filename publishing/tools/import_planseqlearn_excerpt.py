#!/usr/bin/env python3
"""Reproduce selected CC-BY PSL sections, not a translation or a robotics run.

The pinned HTML paragraphs are copied completely, including source mistakes.
Only HTML-to-Markdown markup, formula delimiters, and absolute links change.
--write replaces only the delimited generated section; default checks it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import textwrap
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов.md"
SOURCE = "https://arxiv.org/html/2405.01534v1"
ARCHIVE = ROOT / "05 Источники/Originals/planseqlearn-2026-10-26"
START = "<!-- BEGIN SOURCE: Plan-Seq-Learn 2405.01534v1 -->"
END = "<!-- END SOURCE: Plan-Seq-Learn 2405.01534v1 -->"
ANCHOR = '<a id="plan-seq-learn"></a>'
INSERT_BEFORE = '<span id="когда-более-сильной-модели-нужен-менее-жёсткий-scaffold"></span>'
SOURCE_SHA = "a057b0dd9ba773f4b0594795428e2d0bcb57898bdbcde3ae0acd8146112bae13"
BASE_SHA = "15a26693fe0f017f3d59be94319205afe7b510051a81b63d362d504acec17562"
UNIT = "  - meeting-06-slides-plan-sequence-learn\n"
PAPER = "  - https://arxiv.org/abs/2405.01534v1\n"
FIGURES = "02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/planseqlearn-2026-10-26/"
SECTIONS = ("S3.SS2", "S3.SS3", "S3.SS4", "S3.SS5", "A4.SS3.SSS1", "A2.SS3")
EXPECTED_COUNTS = (1, 5, 4, 4, 2, 1)
ASSET_ROOT = "00 Учебник/Assets/Figures/curated/planseqlearn-2026-10-26/"
ORIGINAL_ROOT = "05 Источники/Originals/planseqlearn-2026-10-26/"
IMAGE_DATA = (
    ("method", "PSL_method.png", "S3.F2", "b54934dba901db0f5ab0170873a4f63150266db17049061bc91db57665b016b8", [1320, 669]),
    ("sequencing", "Sequence_Module.png", "A4.F1", "22c591d868c77a5f48dabde0fa1c11d45f8b18a5468cb2a60b63096bd7ce0e57", [1320, 463]),
    ("nut-assembly", "PSL_fig_1.png", "S1.F1", "dea2df88018452e3010b263d366ff2dfe551dda95472a327a7dd946cc5bd0e70", [1315, 501]),
)


def asset_entries() -> list[dict]:
    return [{
        "id": "curated-planseqlearn-" + name + "-" + image_sha[:12],
        "asset": ASSET_ROOT + name + ".png",
        "author": "Murtaza Dalal, Tarun Chiruvolu, Devendra Singh Chaplot, Ruslan Salakhutdinov",
        "source_url": SOURCE + "#" + anchor,
        "source_image_url": SOURCE + "/" + filename,
        "source_asset": ASSET_ROOT + name + ".png",
        "source_asset_sha256": image_sha,
        "sha256": image_sha,
        "source_page_snapshot": ORIGINAL_ROOT + "article.html",
        "source_page_sha256": SOURCE_SHA,
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "license_evidence": ORIGINAL_ROOT + "license-page.html",
        "license_evidence_sha256": "29dae3bfff2f9df8ca5c68c178d1dc2f717f372abe30ed0964e0796bb961b8d0",
        "public_license_copy": "site/public/licenses/planseqlearn-NOTICE.txt",
        "derivation": "source-raster-copy",
        "modifications": "Complete original PNG downloaded byte-for-byte from arXiv v1. No crop, redraw, relabeling, translation or recompression.",
        "metadata_status": "verified",
        "provenance_confirmation": "source-license-verified",
        "last_verified": "2026-10-26",
        "dimensions_px": dimensions,
        "used_in": [str(CHAPTER.relative_to(ROOT))],
    } for name, filename, anchor, image_sha, dimensions in IMAGE_DATA]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inline(node) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""
    if node.name == "math":
        return "$" + node["alttext"] + "$"
    body = "".join(inline(child) for child in node.children)
    if node.name == "a" and node.get("href"):
        return "[" + body + "](" + urljoin(SOURCE, node["href"]) + ")"
    return body


def excerpt(soup, section_id: str) -> str:
    section = soup.find(id=section_id)
    paragraphs = section.find_all("p", class_="ltx_p")
    expected = EXPECTED_COUNTS[SECTIONS.index(section_id)]
    assert len(paragraphs) == expected, (section_id, len(paragraphs))
    result = [f"<!-- source-psl:{section_id} -->"]
    for paragraph in paragraphs:
        text = re.sub(r"\s+", " ", inline(paragraph)).strip()
        result.append(textwrap.fill(text, width=100, initial_indent="> ",
                                    subsequent_indent="> ", break_long_words=False,
                                    break_on_hyphens=False))
    return "\n\n".join(result)


def figure(name: str, caption: str, source_anchor: str) -> str:
    caption = caption.replace("*", r"\*")
    alt = {"method": "PSL: языковой план, движение к объекту и локальное управление",
           "sequencing": "PSL: от RGB-D и масок объектов к пути манипулятора",
           "nut-assembly": "PSL: четыре этапа установки двух гаек на штифты"}[name]
    return (f"![[{FIGURES}{name}.png|{alt}]]\n\n*{caption} "
            f"Dalal et al., [Plan-Seq-Learn]({SOURCE}#{source_anchor}), ICLR 2024; "
            "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). "
            "Полный исходный PNG без crop, перевода или перерисовки.*")


def generate() -> str:
    payload = (ARCHIVE / "article.html").read_bytes()
    assert digest(payload) == SOURCE_SHA, "pinned article HTML changed"
    soup = BeautifulSoup(payload, "html.parser")
    assert "License: CC BY 4.0" in soup.get_text()
    introduction = """### Plan-Seq-Learn: от языковых подцелей к управлению роботом

Команда «надеть гайку на штифт» задаёт цель, но не движения манипулятора.
Нужно найти гайку на изображении, подвести захват, удержать предмет и совместить
его отверстие со штифтом. В Plan-Seq-Learn (PSL) эти действия разделены между
тремя компонентами: LLM составляет последовательность подцелей, зрение и
планировщик движений подводят робота к нужной области, а обучаемая политика
управляет взаимодействием с предметом. Здесь RL обучает моторное управление;
это другая задача, чем дообучение языковой модели по награде за ответ.

Ниже сохранены полные английские объяснения из §§3.3–3.5 статьи
[Murtaza Dalal, Tarun Chiruvolu, Devendra Singh Chaplot, Ruslan Salakhutdinov,
Plan-Seq-Learn, ICLR 2024](https://arxiv.org/html/2405.01534v1), предваряющий
абзац §3.2, §D.3.1 об условиях завершения и ограничения §B.3. Это цитаты
оригинала по CC BY 4.0, без перевода и стилистической правки. Русские связки
и разбор рисунков отделены от цитат. В Berkeley эта тема находится в лекции 6,
на страницах 107–116; рисунки взяты из первичной статьи, а не из уменьшенных
копий на слайдах."""
    introduction += "\n\n[[05 Источники/Papers/Plan-Seq-Learn|Карточка статьи и авторские материалы]]."
    overview = figure("method", "Figure 2. Верхний блок формирует пары «область, условие завершения». "
                      "Синий блок достигает исходной конфигурации; зелёный обучает взаимодействие "
                      "и передаёт управление следующему этапу.", "S3.F2")
    worked = figure("nut-assembly", "Figure 1. Четыре этапа NutAssembly. "
                    "Синие стрелки в нижнем ряду показывают подведение руки, зелёные — "
                    "локальное управление. Верхний ряд показывает состояния после этапов.", "S1.F1")
    worked_text = """В этом оригинале первой выбрана золотая гайка: `gold nut / grasp`,
затем `gold peg / place`, после них серебряная гайка и серебряный штифт.
После оценки положения золотой гайки планировщик приводит руку в её окрестность.
Политика по изображению с камеры у захвата учится захватывать и поднимать гайку.
Проверка `grasp` завершает первый этап. Планировщик переносит руку с удерживаемой
гайкой к золотому штифту, после чего та же политика управляет установкой.
Проверка `place` разрешает переход к серебряной гайке. Это четыре условия
завершения, а не четыре заранее готовых моторных навыка. На слайдах Berkeley
показан другой допустимый порядок — сначала серебряная гайка.

Состояние после удачного этапа становится началом следующего. Если вероятность
успеха каждого этапа при условии успешных предыдущих равна 0,9, то
вероятность завершить всю цепочку равна $0{,}9^4=0{,}6561$. Это учебный расчёт
по правилу произведения, не результат эксперимента PSL. Разделение задачи
помогает обучать короткие взаимодействия, но само по себе не устраняет
накопление ошибок."""
    sequencing = figure("sequencing", "Figure D.1. Глубина и калибровка камеры дают облако точек; "
                        "маски выделяют объект и робота. Обратная кинематика переводит целевое "
                        "положение захвата в углы суставов, AIT* ищет путь без столкновений.", "A4.F1")
    caveat = """В §3.5 оригинала после установки серебряной гайки названа область `gold peg`.
Это несогласованность текста с его же планом: перед установкой золотую гайку
сначала нужно захватить, поэтому соответствующий переход ведёт к `gold nut`.
Цитата сохранена без исправления; порядок на Figure 1 и в примере выше
следует списку подцелей из §3.3.

Выбор метки `grasp` ещё не задаёт надёжный датчик завершения. Следующий раздел
оригинала раскрывает, как авторы проверяли захват, положение и поворот предметов."""
    conclusion = """Опыты PSL проводились в симуляторах, а не на физическом роботе;
в них использовались заданная средой награда, калиброванные камеры и доступные
условия завершения. В §D.1 также описано удаление шагов, где LLM придумала
несуществующий объект или условие. Поэтому «без заранее обученной библиотеки
навыков» не означает «без настроек среды и проверки плана». В отличие от
WebDreamer, PSL не сравнивает воображаемые будущие страницы: он связывает
языковую декомпозицию с геометрическим планированием и реальными переходами
симулятора при обучении локальной политики.

[Видео выполнения и авторский код](https://mihdalal.github.io/planseqlearn/)
показывают движение, которое неподвижные кадры передают лишь по этапам."""
    parts = [START, ANCHOR, introduction, overview, excerpt(soup, "S3.SS2"),
             "#### Planning Module: Zero-Shot High-level Planning", excerpt(soup, "S3.SS3"),
             worked, worked_text,
             "#### Sequencing Module: Vision-based Plan Tracking", excerpt(soup, "S3.SS4"), sequencing,
             "#### Learning Module: Efficiently Learning Local Control", excerpt(soup, "S3.SS5"), caveat,
             "#### Stage Termination Details", excerpt(soup, "A4.SS3.SSS1"),
             "#### Limitations", excerpt(soup, "A2.SS3"), conclusion, END]
    return "\n\n".join(parts) + "\n\n"


def baseline(text: str) -> str:
    from topic_consolidation_baseline import before_topic_consolidation
    text = before_topic_consolidation(text, CHAPTER,
                                     '2026-10-30-agent-reconciliation-additions.json')
    if START in text:
        text = re.sub(re.escape(START) + r"[\s\S]*?" + re.escape(END) + r"\n\n", "", text)
    return (text.replace("last_updated: 2026-10-26", "last_updated: 2026-10-20", 1)
            .replace(UNIT, "", 1).replace(PAPER, "", 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    from topic_consolidation_baseline import before_topic_consolidation
    current_raw = CHAPTER.read_text()
    current = before_topic_consolidation(current_raw, CHAPTER,
                                        '2026-10-30-agent-reconciliation-additions.json')
    # baseline() performs its own exact journal reversal from the raw revision.
    before = baseline(current_raw)
    assert digest(before.encode()) == BASE_SHA, "existing chapter changed outside approved addition"
    generated = generate()
    expected = before.replace(INSERT_BEFORE, generated + INSERT_BEFORE, 1)
    expected = expected.replace("last_updated: 2026-10-20", "last_updated: 2026-10-26", 1)
    expected = expected.replace("source_unit_id:\n", "source_unit_id:\n" + UNIT, 1)
    expected = expected.replace("primary_sources:\n", "primary_sources:\n" + PAPER, 1)
    if args.write:
        assert current_raw == current, 'refusing to overwrite later reviewed additions'
        CHAPTER.write_text(expected)
        (ROOT / ASSET_ROOT / "assets.json").write_text(
            json.dumps(asset_entries(), ensure_ascii=False, indent=2) + "\n")
    else:
        assert current == expected, "chapter PSL block differs from pinned source conversion"
        assert json.loads((ROOT / ASSET_ROOT / "assets.json").read_text()) == asset_entries()
    for row in asset_entries():
        for field, checksum in [("asset", "sha256"), ("source_page_snapshot", "source_page_sha256"),
                                ("license_evidence", "license_evidence_sha256")]:
            assert digest((ROOT / row[field]).read_bytes()) == row[checksum], row[field]
    print("PASS: 17 complete original paragraphs / 6 source sections; 3 original figures;")
    print("      previous chapter restored byte-for-byte after removing the bounded addition.")
    print("No translation, robot simulation, RL training or benchmark executed.")


if __name__ == "__main__":
    main()
