"""Non-executing projection of the pinned Harvard Beamer dialect.

This is not a TeX engine. Known presentation wrappers become Markdown; math is
kept as TeX, notes are separated, and unsupported constructs remain visible.
Every generated page must also retain its hash-verified complete original.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Callable


def strip_comments(text: str) -> str:
    result = []
    i = 0
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            result.append(text[i:i + 2])
            i += 2
        elif text[i] == '%':
            end = text.find('\n', i)
            i = len(text) if end < 0 else end
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def group(text: str, position: int, opener: str = '{') -> tuple[str, int]:
    """Read balanced TeX groups without evaluating macros or comments."""
    while position < len(text) and text[position].isspace():
        position += 1
    closer = '}' if opener == '{' else ']'
    if position >= len(text) or text[position] != opener:
        raise ValueError(f'Expected {opener!r} at {position}: {text[position:position + 80]!r}')
    start = position + 1
    depth = 1
    position += 1
    while position < len(text):
        char = text[position]
        if char == '\\':
            position += 2
            continue
        if char == '%':
            end = text.find('\n', position)
            position = len(text) if end < 0 else end
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start:position], position + 1
        position += 1
    raise ValueError(f'Unclosed {opener!r}: {text[start:start + 80]!r}')


def optional(text: str, position: int) -> tuple[str, int]:
    position0 = position
    while position < len(text) and text[position].isspace():
        position += 1
    if position < len(text) and text[position] == '[':
        return group(text, position, '[')
    return '', position0


def environment(text: str, position: int, name: str) -> tuple[str, int]:
    start = position
    depth = 1
    pattern = re.compile(r'\\(begin|end)\{' + re.escape(name) + r'\}')
    for match in pattern.finditer(text, position):
        line_start = text.rfind('\n', 0, match.start()) + 1
        if not strip_comments(text[line_start:match.start()] + 'SENTINEL').endswith('SENTINEL'):
            continue
        depth += 1 if match[1] == 'begin' else -1
        if depth == 0:
            return text[start:match.start()], match.end()
    raise ValueError(f'Unclosed environment {name!r}')


def code(text: str) -> str:
    fence = '`' * (max([len(m[0]) for m in re.finditer(r'`+', text)] or [0]) + 1)
    return f'{fence} {text} {fence}'


def code_block(text: str) -> str:
    fence = '`' * max(3, max([len(m[0]) + 1 for m in re.finditer(r'`+', text)] or [0]))
    return f'\n\n{fence}tex\n{text.strip()}\n{fence}\n\n'


MATH_ALIASES = {
    'Dvol': r'D_{\text{vol}}', 'Rpeak': r'R_{\text{peak}}',
    'Llat': r'L_{\text{lat}}', 'Ttotal': r'T_{\text{total}}',
}


def aliases_from_source(source: str) -> dict[str, str]:
    aliases = dict(MATH_ALIASES)
    clean = strip_comments(source)
    for match in re.finditer(r'\\(?:newcommand|renewcommand)\s*\{\\([A-Za-z]+)\}', clean):
        start = match.end()
        if clean[start:].lstrip().startswith('['):
            continue  # Parameterised macros are not safe text substitutions.
        if clean[start:].lstrip().startswith('{'):
            body, _ = group(clean, start)
            if '#' not in body and match[1] not in {'familydefault'}:
                aliases[match[1]] = body
    return aliases


def expand_math(text: str, aliases: dict[str, str]) -> str:
    for _ in range(8):
        expanded = re.sub(r'\\([A-Za-z]+)\b', lambda m: aliases.get(m[1], m[0]), text)
        if expanded == text:
            return text
        text = expanded
    raise ValueError('Cyclic or excessively nested math macro')


def math_segment(text: str, position: int) -> tuple[str, int] | None:
    if text.startswith('$$', position):
        opening, closing = '$$', '$$'
    elif text[position] == '$':
        opening, closing = '$', '$'
    elif text.startswith(r'\[', position):
        opening, closing = r'\[', r'\]'
    elif text.startswith(r'\(', position):
        opening, closing = r'\(', r'\)'
    else:
        return None
    end = position + len(opening)
    while end < len(text):
        if text.startswith(closing, end):
            return text[position + len(opening):end], end + len(closing)
        end += 2 if text[end] == '\\' else 1
    raise ValueError(f'Unclosed math at {position}: {text[position:position + 80]!r}')


def split_top(text: str, separator: str) -> list[str]:
    """Split table/list tokens only outside groups, math and environments."""
    pieces = []
    start = i = 0
    while i < len(text):
        math = math_segment(text, i)
        if math:
            i = math[1]
        elif text[i] == '{':
            _, i = group(text, i)
        elif text.startswith(r'\begin{', i):
            name, pos = group(text, i + len(r'\begin'))
            _, i = environment(text, pos, name)
        elif text.startswith(separator, i) and (
            not separator[-1].isalpha() or i + len(separator) == len(text)
            or not text[i + len(separator)].isalpha()
        ):
            pieces.append(text[start:i])
            i += len(separator)
            start = i
        elif text[i] == '\\':
            i += 2
        else:
            i += 1
    pieces.append(text[start:])
    return pieces


def project_tex(text: str, *, aliases: dict[str, str] | None = None,
                image_resolver: Callable[[str], str | None] | None = None,
                warnings: list[str] | None = None, depth: int = 0) -> str:
    aliases = aliases if aliases is not None else dict(MATH_ALIASES)
    warnings = warnings if warnings is not None else []
    text = strip_comments(text)
    # TeX indentation is layout source, not Markdown's four-space code block.
    # Lists are indented only after their child fragments have been projected.
    text = re.sub(r'(?m)^[ \t]+', '', text)
    out = []
    i = 0

    def project(value: str) -> str:
        return project_tex(value, aliases=aliases, image_resolver=image_resolver,
                           warnings=warnings, depth=depth + 1)

    while i < len(text):
        math = math_segment(text, i)
        if math:
            formula, end = math
            formula = expand_math(formula, aliases).replace(r'\$', r'\textdollar{}')
            display = text.startswith(('$$', r'\['), i)
            if display:
                out.append('\n\n$$\n' + formula.strip() + '\n$$\n\n')
            else:
                formula = re.sub(r'\s+', ' ', formula.strip())
                if out and out[-1].startswith('$') and out[-1].endswith('$'):
                    # TeX permits adjacent $a$$b$ spans; Markdown interprets
                    # the central $$ as a different delimiter.
                    out[-1] = out[-1][:-1] + ' ' + formula + '$'
                else:
                    out.append('$' + formula + '$')
            i = end
            continue
        if text[i] == '{':
            body, i = group(text, i)
            out.append(project(body))
            continue
        if text[i] != '\\':
            if text.startswith(('``', "''"), i):
                out.append('“' if text[i] == '`' else '”')
                i += 2
                continue
            if text.startswith('--', i):
                count = 3 if text.startswith('---', i) else 2
                out.append('—' if count == 3 else '–')
                i += count
                continue
            if text[i] == '~':
                out.append(' ')
            elif text[i] in '#_':
                out.append('\\' + text[i])
            else:
                out.append(text[i])
            i += 1
            continue
        match = re.match(r'\\([A-Za-z@]+\*?|.)', text[i:], re.S)
        if not match:
            out.append('\\')
            i += 1
            continue
        command = match[1]
        i += len(match[0])
        if command in {'%', '&', '_', '#', '{', '}', '$'}:
            out.append('&#36;' if command == '$' else command)
        elif command == '\\':
            _, i = optional(text, i)
            out.append('\n\n')
        elif command in {' ', ',', ';', ':', '!', '/', 'quad', 'qquad'}:
            out.append(' ')
        elif command in {'textbar', 'textless', 'textgreater', 'checkmark', 'ldots'}:
            out.append({'textbar': '|', 'textless': '<', 'textgreater': '>',
                        'checkmark': '✓', 'ldots': '…'}[command])
        elif command in {'textsubscript', 'textsuperscript'}:
            body, i = group(text, i)
            tag = 'sub' if command == 'textsubscript' else 'sup'
            out.append('<' + tag + '>' + project(body) + '</' + tag + '>')
        elif command == 'raisebox':
            _, i = group(text, i)
            _, i = optional(text, i)
            _, i = optional(text, i)
            body, i = group(text, i)
            out.append(project(body))
        elif command in {'"', "'", '`'}:
            body, i = group(text, i) if text[i:i + 1] == '{' else (text[i:i + 1], i + 1)
            if len(body) == 1 and body.isalpha():
                accent = {'"': '\u0308', "'": '\u0301', '`': '\u0300'}[command]
                out.append(unicodedata.normalize('NFC', body + accent))
            else:
                warnings.append('Accent preserved as original TeX: \\' + command)
                out.append(code('\\' + command + '{' + body + '}'))
        elif command in aliases:
            out.append(project(aliases[command]))
        elif command in {'textbf', 'alert', 'emph', 'textit', 'texttt', 'textsc', 'textrm',
                         'textsf', 'textnormal', 'mbox', 'MakeUppercase', 'underline'}:
            body, i = group(text, i)
            if command == 'texttt' and re.search(r'\\(?:[A-Za-z]+|\\)', body):
                warnings.append('Typeset code retained as a TeX fragment, not executable code')
            body = (re.sub(r'\\([_%&#${}])', r'\1', body) if command == 'texttt' else project(body))
            if command in {'textbf', 'alert'}:
                out.append('**' + body.strip() + '**')
            elif command in {'emph', 'textit'}:
                out.append('*' + body.strip() + '*')
            elif command == 'texttt':
                out.append(code(body))
            else:
                out.append(body.upper() if command == 'MakeUppercase' else body)
        elif command in {'textcolor', 'colorbox'}:
            _, i = group(text, i)
            body, i = group(text, i)
            out.append(project(body))
        elif command in {'href', 'url'}:
            url, i = group(text, i)
            label, i = group(text, i) if command == 'href' else (url, i)
            out.append('[' + project(label) + '](<' + url + '>)')
        elif command in {'safeimg', 'includegraphics'}:
            _, i = optional(text, i)
            name, i = group(text, i)
            destination = image_resolver(name) if image_resolver else None
            if destination:
                caption = ''
                if isinstance(destination, tuple):
                    destination, caption = destination
                out.append('\n\n![[' + destination + ']]\n\n')
                if caption:
                    out.append(caption + '\n\n')
            else:
                warnings.append('Unresolved image: ' + name)
                out.append('\n\nOriginal figure reference: ' + code(name) + '.\n\n')
        elif command in {'mlsysconcept', 'mlsysalert', 'mlsysinsight', 'mlsysref'}:
            heading, i = group(text, i)
            body, i = group(text, i)
            out.append('\n\n**' + project(heading).strip() + '** ' + project(body) + '\n\n')
        elif command == 'mlsyscite':
            body, i = group(text, i)
            out.append('\n\nSource on slide: ' + project(body) + '\n\n')
        elif command in {'vspace', 'vspace*', 'hspace', 'hspace*', 'color', 'label'}:
            _, i = group(text, i)
        elif command in {'setlength', 'addtolength'}:
            while i < len(text) and text[i].isspace():
                i += 1
            if text.startswith('\\', i):
                variable = re.match(r'\\[A-Za-z]+', text[i:])
                i += len(variable[0])
            else:
                _, i = group(text, i)
            _, i = group(text, i)
        elif command in {'renewcommand', 'newcommand'}:
            _, i = group(text, i)
            _, i = optional(text, i)
            _, i = group(text, i)
        elif command in {'small', 'footnotesize', 'scriptsize', 'tiny', 'normalsize', 'large',
                         'Large', 'LARGE', 'huge', 'Huge', 'bfseries', 'itshape', 'ttfamily',
                         'rmfamily', 'sffamily', 'centering', 'raggedright', 'pause', 'hfill',
                         'vfill', 'medskip', 'smallskip', 'bigskip', 'strut', 'selectfont'}:
            pass
        elif command in {'par', 'newline'}:
            out.append('\n\n')
        elif command == 'begin':
            name, i = group(text, i)
            _, i = optional(text, i)
            if name in {'column', 'mlsyscard', 'minipage', 'tabular'}:
                _, i = group(text, i)
            body, i = environment(text, i, name)
            if name in {'columns', 'column', 'mlsyscard', 'minipage', 'center'}:
                out.append('\n\n' + project(body) + '\n\n')
            elif name in {'block', 'exampleblock', 'alertblock'}:
                title, pos = group(body, 0)
                out.append('\n\n**' + project(title) + '**\n\n' + project(body[pos:]) + '\n\n')
            elif name in {'itemize', 'enumerate'}:
                items = split_top(body, r'\item')
                out.append('\n\n' + project(items[0]).strip() + '\n\n')
                for number, item in enumerate(items[1:], 1):
                    item = re.sub(r'^\s*<[^>]*>\s*', '', item)
                    label, pos = optional(item, 0)
                    if label:
                        item = item[pos:]
                    prefix = '- ' if name == 'itemize' else f'{number}. '
                    lines = project(item).strip().splitlines()
                    if lines:
                        out.append(prefix + lines[0] + '\n')
                        out.extend(' ' * len(prefix) + line + '\n' for line in lines[1:])
                out.append('\n')
            elif name == 'tabular':
                original = r'\begin{tabular}{' + _ + '}' + body + r'\end{tabular}'
                table_body = re.sub(r'\\(?:toprule|midrule|bottomrule)\b', '', body)
                raw_rows = [re.sub(r'^\s*\[[^\]]*\]', '', row).strip()
                            for row in split_top(table_body, r'\\')]
                rows = [split_top(row, '&') for row in raw_rows if row]
                rendered = []
                spans = False
                for row in rows:
                    cells = []
                    for cell in row:
                        cell = cell.strip()
                        count = 1
                        if cell.startswith(r'\multicolumn'):
                            count_text, pos = group(cell, len(r'\multicolumn'))
                            _, pos = group(cell, pos)
                            contents, pos = group(cell, pos)
                            if not count_text.isdigit() or cell[pos:].strip():
                                warnings.append('Unsupported column span preserved in original source')
                            else:
                                count = int(count_text)
                                cell = contents
                        # Markdown table pipes are structural. Use equivalent
                        # TeX bar commands inside math, and escape text pipes.
                        parts = []
                        k = 0
                        while k < len(cell):
                            segment = math_segment(cell, k)
                            if segment:
                                formula, end = segment
                                formula = formula.replace(r'\|', r'\Vert')
                                formula = re.sub(r'(?<!\\)\|', lambda m: r'\vert ', formula)
                                parts.append('$' + formula + '$')
                                k = end
                            elif cell[k] == '\\':
                                parts.append(cell[k:k + 2])
                                k += 2
                            else:
                                parts.append(r'\|' if cell[k] == '|' else cell[k])
                                k += 1
                        value = re.sub(r'\s+', ' ', project(''.join(parts))).strip()
                        if count > 1:
                            first = len(cells) + 1
                            value += f' (source columns {first}–{first + count - 1})'
                            spans = True
                        cells.extend([value] + [''] * (count - 1))
                    rendered.append(cells)
                if (not rendered or len({len(row) for row in rendered}) != 1
                        or re.search(r'\\(?:multirow|cmidrule|cline)', body)):
                    warnings.append('Complex table preserved as original TeX')
                    out.append(code_block(original))
                else:
                    out.append('\n\n| ' + ' | '.join(rendered[0]) + ' |\n')
                    out.append('| ' + ' | '.join(['---'] * len(rendered[0])) + ' |\n')
                    out.extend('| ' + ' | '.join(row) + ' |\n' for row in rendered[1:])
                    out.append('\n')
                    if spans:
                        warnings.append('Column spans are labelled and expanded into separate Markdown cells')
            elif name in {'equation', 'equation*', 'align', 'align*', 'gather', 'gather*'}:
                wrapper = {'align': 'aligned', 'gather': 'gathered'}.get(name.rstrip('*'))
                formula = expand_math(body.strip(), aliases)
                if wrapper:
                    formula = rf'\begin{{{wrapper}}}' + '\n' + formula + '\n' + rf'\end{{{wrapper}}}'
                out.append('\n\n$$\n' + formula + '\n$$\n\n')
            else:
                warnings.append('Environment preserved as original TeX: ' + name)
                out.append(code_block(r'\begin{' + name + '}' + body + r'\end{' + name + '}'))
        else:
            warnings.append('Command preserved as original TeX: \\' + command)
            out.append(code('\\' + command))
    result = ''.join(out)
    return re.sub(r'\n{3,}', '\n\n', result).strip()


@dataclass
class Slide:
    title: str
    body: str
    original: str
    kind: str
    line: int
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def without_notes(body: str) -> tuple[str, list[str]]:
    out = []
    notes = []
    position = 0
    for match in re.finditer(r'\\note(?:<[^>]*>)?(?![A-Za-z])', body):
        if match.start() < position:
            continue
        note, end = group(body, match.end())
        out.append(body[position:match.start()])
        notes.append(note)
        position = end
    out.append(body[position:])
    return ''.join(out), notes


def parse_deck(source: str, image_resolver: Callable[[str], str | None] | None = None) -> list[Slide]:
    # Use original positions for exact notes, line links and per-slide source.
    begin = source.find(r'\begin{document}')
    if begin < 0:
        raise ValueError('No document environment')
    position = begin + len(r'\begin{document}')
    aliases = aliases_from_source(source[:begin])
    pattern = re.compile(r'\\begin\{frame\}|\\mlsysfocus\b|\\mlsystitle\b')
    slides = []
    while match := pattern.search(source, position):
        # Commented-out frames/macros must not become slides.
        line_start = source.rfind('\n', 0, match.start()) + 1
        if not strip_comments(source[line_start:match.start()] + 'SENTINEL').endswith('SENTINEL'):
            position = match.end()
            continue
        kind = 'frame' if match[0].startswith(r'\begin') else match[0][1:]
        pos = match.end()
        if kind == 'frame':
            _, pos = optional(source, pos)
            while pos < len(source) and source[pos].isspace():
                pos += 1
            title, pos = group(source, pos) if source[pos:pos + 1] == '{' else (f'Frame {len(slides) + 1}', pos)
            body, end = environment(source, pos, 'frame')
        else:
            title, pos = group(source, pos)
            body, end = group(source, pos)
            if kind == 'mlsystitle':
                # Decorative cover art is not a teaching figure. Preserve its
                # exact reference in the complete source, without downloading it.
                _, end = group(source, end)
        body, notes = without_notes(body)
        warnings = []
        rendered = project_tex(body, aliases=aliases, image_resolver=image_resolver, warnings=warnings)
        slides.append(Slide(project_tex(title, aliases=aliases, warnings=warnings), rendered,
                            source[match.start():end], kind, source.count('\n', 0, match.start()) + 1,
                            notes, sorted(set(warnings))))
        position = end
    return slides
