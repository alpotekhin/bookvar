"""Check the hand-worked examples in the corpus introduction (CPU, stdlib)."""

from math import isclose
from pathlib import Path
import hashlib
import re

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/11 Pre-training и Scaling/01 Данные и pre-training.md'
ARCHIVE = ROOT / '05 Источники/Originals/olmo-pretraining-2026-10-26/article.html'
ARTICLE_SHA = '68e5fd8ffb899793b8c517770f9e8611946bb12a2a23c84e3641615c3a4d67d0'
PARAGRAPHS = (
    'S2.SS3.SSS0.Px1.p1.1',
    'S2.SS3.SSS0.Px2.p1.1',
    'S2.SS3.SSS0.Px2.p2.1',
)


def tokens(text):
    """Ignore layout whitespace, but not words, punctuation or numbers."""
    return re.findall(r'\w+|[^\w\s]', text)


def check_original_recipe(text):
    payload = ARCHIVE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == ARTICLE_SHA
    article = BeautifulSoup(payload, 'html.parser')
    assert 'License: CC BY 4.0' in article.get_text()
    block = text.split('<!-- BEGIN SOURCE: OLMo 2 2501.00656v1 recipe -->', 1)[1]
    block = block.split('<!-- END SOURCE: OLMo 2 2501.00656v1 recipe -->', 1)[0]
    for paragraph_id in PARAGRAPHS:
        excerpt = block.split('<!-- source-olmo:' + paragraph_id + ' -->', 1)[1]
        excerpt = excerpt.split('<!--', 1)[0]
        excerpt = re.sub(r'^> ?', '', excerpt, flags=re.M)
        plain = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', excerpt)
        original = article.find(id=paragraph_id)
        assert original is not None
        assert tokens(plain) == tokens(original.get_text()), paragraph_id
        actual_links = re.findall(r'\]\(([^)]+)\)', excerpt)
        expected_links = ['https://arxiv.org/html/2501.00656v1' + a['href']
                          for a in original.find_all('a', href=True)]
        assert actual_links == expected_links, (paragraph_id, actual_links, expected_links)

    # Compare all 40 reported scores with the actual HTML table, preserving
    # checkpoint order across the two more readable Markdown tables.
    rows = []
    for row in article.find(id='S4.T9').find_all('tr'):
        cells = [c.get_text(' ', strip=True) for c in row.find_all(['td', 'th'], recursive=False)]
        if cells and cells[0] in {'Pretraining', 'Pretraining & mid-training'}:
            rows.append([float(cell) for cell in cells[1:]])
    assert len(rows) == 4 and all(len(row) == 10 for row in rows)
    chapter_rows = []
    for line in text.splitlines():
        if re.match(r'\| (?:7B|13B) · pretraining', line):
            chapter_rows.append([float(cell.strip()) for cell in line.strip('|').split('|')[1:]])
    assert len(chapter_rows) == 8
    assert [left + right for left, right in zip(chapter_rows[:4], chapter_rows[4:])] == rows
    inventory_rows = []
    for table_id in ('S2.T1', 'S2.T2'):
        for row in article.find(id=table_id).find_all('tr'):
            cells = [c.get_text(' ', strip=True) for c in row.find_all(['td', 'th'], recursive=False)]
            if len(cells) == 6 and re.fullmatch(r'\d+(?:\.\d+)?[KMBT]', cells[2]):
                inventory_rows.append(cells[2:])
    card = (ROOT / '05 Источники/Papers/OLMo 2 Dolmino.md').read_text('utf-8')
    card_rows = []
    for line in card.splitlines():
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        if line.startswith('|') and len(cells) == 6 and re.fullmatch(r'\d+(?:\.\d+)?[KMBT]', cells[2]):
            card_rows.append(cells[2:])
    assert len(inventory_rows) == 22
    assert card_rows == inventory_rows, 'Table 1/2 row order or source-reported inventory quantities changed'
    for link in re.findall(r'https://arxiv.org/html/2501\.00656v1#([^\s)]+)', text):
        assert article.find(id=link) is not None, link
    assert 2000 * 1024 * 4096 == 8_388_608_000
    assert 2000 * 2048 * 4096 == 16_777_216_000
    assert 3 * 50 == 150 and 3 * 100 + 300 == 600  # billions, across branches
    assert isclose((0.8 + 1.0 + 1.2) / 3, 1.0)
    assert isclose(rows[1][8] - rows[0][8], 43.4)
    assert isclose(rows[1][1] - rows[0][1], 3.9)
    assert not isclose(sum(rows[0][1:7]) / 6, rows[0][0])
    print('PASS: three complete English recipe paragraphs and original reference targets')
    print('PASS: all 40 Table 9 scores, warmup budgets and independent-branch arithmetic')
    print('PASS: all 22 Table 1/2 rows and 88 source-reported inventory quantities')


def main():
    before = 10 / 100
    after = 1 / (100 - 9)
    assert isclose(before, 0.1)
    assert isclose(after * 100, 1.098901098901099)
    assert isclose(before / after, 9.1)

    budget, web_tokens, code_tokens = 500, 900, 100  # billions
    assert isclose(budget * 0.3, 150)
    assert isclose(budget * 0.3 / code_tokens, 1.5)
    assert isclose(budget * 0.7, 350)
    assert budget * 0.7 / web_tokens < 1

    olmo = [3710, 83, 58.6, 20.8, 12.2, 11.8, 3.7]
    assert isclose(sum(olmo), 3900.1)
    assert olmo[0] / sum(olmo) > 0.95
    dolmino = [47.2, 16.6, 2.45, 5.85, 7.11, 20.8]
    assert isclose(sum(dolmino), 100.01)
    # The microanneal example is a counterfactual teaching calculation,
    # not a newly executed OLMo training run.
    assert 200e6 + 200e6 == 400e6
    for consumed, remaining_lr in ((100e6, 0.75), (200e6, 0.5), (400e6, 0)):
        assert 1 - consumed / 400e6 == remaining_lr
    assert 130e9 < 3 * 50e9
    print('PASS: duplicate weight, mixture budget, OLMo corpus and Dolmino shares')
    print('PASS: microanneal token budget and learning-rate schedule')
    check_original_recipe(CHAPTER.read_text('utf-8'))
    print('Chapter SHA-256:', hashlib.sha256(CHAPTER.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
