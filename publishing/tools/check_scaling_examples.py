#!/usr/bin/env python3
"""Check numbers read from chapter 43; no model training or performance test.

Rounded displays use half a unit in their last printed decimal place. Published
loss tables are checked as transcriptions, not as reproduced experiments.
"""

import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/43 Scaling laws.md"
ASSETS = ROOT / "00 Учебник/Assets/Figures/curated/scaling-review-2026-10"
TEXT = CHAPTER.read_text(encoding="utf-8")
RESULTS = {}
FENCE = chr(96) * 3


def need(pattern, text=TEXT):
    match = re.search(pattern, text, re.S | re.M)
    assert match, f"Expected actual chapter expression: {pattern}"
    return match


def number(value):
    value = value.replace("{,}", ".").replace(",", ".").replace("*", "")
    return float(value.strip().strip("$"))


def approx(actual, displayed, tolerance):
    assert abs(actual - displayed) <= tolerance + 1e-12, (
        actual, displayed, tolerance
    )


def table(header):
    lines = TEXT.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    assert rows, header
    return rows


def mean_example():
    sigma = number(need(r"При \$\\sigma=(\d+)\$").group(1))
    n1, n2, e1, e2 = need(
        r"выборки со (\d+) до (\d+) наблюдений.*?с \$(0\{,\}\d+)\$\s*до \$(0\{,\}\d+)\$"
    ).groups()
    errors = [sigma**2 / int(n1), sigma**2 / int(n2)]
    approx(errors[0], number(e1), 0.005)
    approx(errors[1], number(e2), 0.005)
    approx(math.sqrt(errors[0] / errors[1]), 2, 1e-12)
    RESULTS["mean_mse"] = errors


def isoflop_example():
    coefficient, power = need(r"C_1=(\d+)\\cdot10\^\{(\d+)\}").groups()
    compute = int(coefficient) * 10**int(power)
    rows = table("| $N$ | $D=C_1/(6N)$")
    token_counts = []
    for n_cell, d_cell, ratio_cell, _loss in rows:
        n = number(n_cell.split()[0]) * 1e9
        d = compute / (6 * n)
        approx(d / 1e9, number(d_cell.split()[0]), 0.5)
        approx(d / n, number(ratio_cell), 0.05)
        token_counts.append(d / 1e9)
    losses = [number(row[3]) for row in rows]
    assert losses.index(min(losses)) == 1
    blocks = re.findall(FENCE + r"python\n(.*?)\n" + FENCE, TEXT, re.S)
    assert len(blocks) == 2
    block = next(block for block in blocks if "def project_budget(" in block)
    tree = ast.parse(block)
    call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "project_budget"
    )
    args = [ast.literal_eval(arg) for arg in call.args]
    anchor, denominator, den_power, exponent = need(
        r"N\^\*\(C\)=(\d+\{,\}\d+)\\text\{B\}\s*"
        r"\\left\(\\frac\{C\}\{(\d+)\\cdot10\^\{(\d+)\}\}\\right\)\^\{([\d.]+)\}"
    ).groups()
    target_coefficient, target_power = need(
        r"C_\{\\mathrm\{target\}\}=(\d+)\\cdot10\^\{(\d+)\}"
    ).groups()
    assert args == [
        int(target_coefficient) * 10**int(target_power),
        int(denominator) * 10**int(den_power),
        number(anchor) * 1e9, number(exponent),
    ]
    namespace = {}
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(tree, str(CHAPTER), "exec"), namespace)
    expected_stdout = need(r"^# (N=[^\n]*)$", block).group(1)
    assert output.getvalue().strip() == expected_stdout
    n, d = namespace["n"], namespace["d"]
    displayed_n = number(need(r"N\^\*_\{\\mathrm\{target\}\}=(\d+)\\text\{B\}").group(1))
    approx(n / 1e9, displayed_n, 1e-12)
    d_coefficient, d_power = need(r"\\approx(\d+\{,\}\d+)\\cdot10\^\{(\d+)\}").groups()
    approx(d / 10**int(d_power), number(d_coefficient), 0.005)
    ratio = number(need(r"Отношение равно примерно \$(\d+\{,\}\d+)\$").group(1))
    approx(d / n, ratio, 0.05)
    lower, upper, d_lower, d_upper = need(
        r"интервал (\d+)–(\d+) млрд.*?от (\d+) до (\d+) млрд"
    ).groups()
    for parameters, tokens in [(lower, d_lower), (upper, d_upper)]:
        approx(args[0] / (6 * int(parameters) * 1e18), int(tokens), 0.5)
    invalid = 0
    for position in range(4):
        candidates = [0, -1, math.nan, math.inf]
        if position == 3:
            candidates.append(1)
        for value in candidates:
            bad_args = args.copy()
            bad_args[position] = value
            try:
                namespace["project_budget"](*bad_args)
            except ValueError:
                invalid += 1
            else:
                raise AssertionError(("accepted invalid projection input", bad_args))
    RESULTS["isoflop_tokens_billions"] = token_counts
    RESULTS["projection"] = {"N_B": n / 1e9, "D_B": d / 1e9, "D_per_N": d / n,
                             "invalid_inputs_rejected": invalid}


def nonparametric_example():
    cases = re.findall(r"При \$n=(\d+)\\,(\d+)\$ и \$h=(\d+\{,\}\d+)\$", TEXT)
    assert len(cases) == 2
    results = []
    for thousands, remainder, h_text in cases:
        n, h = int(thousands + remainder), number(h_text)
        cells = h**-2
        count = n*h*h
        results.append((round(cells), round(count), 1/count, h*h))
    assert [x[:2] for x in results] == [(100, 100), (400, 400)]
    approx(results[0][2], .01, 1e-12)
    approx(results[1][2], .0025, 1e-12)
    approx(results[0][3]/results[1][3], 4, 1e-12)
    assert 'n^{-2/(d+2)}' in TEXT and 'функция\nлипшицева' in TEXT
    RESULTS["nonparametric_cells"] = results


def fitted_examples():
    blocks = re.findall(FENCE + r"python\n(.*?)\n" + FENCE, TEXT, re.S)
    block = next(block for block in blocks if "def three_point_minimum(" in block)
    namespace, output = {}, io.StringIO()
    with redirect_stdout(output):
        exec(compile(block, str(CHAPTER) + ":quadratic", "exec"), namespace)
    assert output.getvalue().strip() == need(r"^# (N=[^\n]+)$", block).group(1)
    fn = namespace["three_point_minimum"]
    approx(namespace["n_min"], 6e9 * 2**(1 / 18), .001)
    approx(namespace["d_min"] / 1e9, 80.18531974117876, 1e-10)
    # Check the stated polynomial against all three actual toy rows.
    rows = table("| $N$ | $D=C_1/(6N)$")
    a, b, c = map(number, need(r"q\(s\)=(\d+\{,\}\d+)s\^2-(\d+\{,\}\d+)s\+(\d+\{,\}\d+)").groups())
    call = next(node for node in ast.walk(ast.parse(block)) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "three_point_minimum")
    sizes, losses = [ast.literal_eval(arg) for arg in call.args]
    assert sizes == [number(row[0].split()[0])*1e9 for row in rows]
    assert losses == [number(row[-1]) for row in rows]
    for size, row in zip(sizes, rows):
        s = math.log2(size / 6e9)
        approx(a * s*s - b * s + c, number(row[-1]), 1e-12)
    approx(fn([3, 6, 12], [2.47, 2.42, 2.46]), namespace["n_min"] / 1e9, 1e-10)
    invalid = [([1, 2], [2, 1]), ([1, 2, 4], [1, 1]),
               ([1, 0, 4], [2, 1, 2]), ([1, math.nan, 4], [2, 1, 2]),
               ([1, 3, 4], [2, 1, 2]), ([4, 2, 1], [2, 1, 2]),
               ([1, 2, 4], [1, 1, 1]), ([1, 2, 4], [1, 2, 1]),
               ([1, 2, 4], [4, 2, 1]), ([1, 2, 4], [2, math.inf, 2])]
    for sizes, losses in invalid:
        try:
            fn(sizes, losses)
        except ValueError:
            pass
        else:
            raise AssertionError(("invalid quadratic accepted", sizes, losses))
    def huber(r, delta):
        return r*r/2 if abs(r) <= delta else delta*(abs(r)-delta/2)
    observed, pred1, pred2 = map(number, need(
        r"при наблюдаемом \$L_i=(\d+)\$ прогнозы \$(\d+\{,\}\d+)\$ и \$(\d+\{,\}\d+)\$"
    ).groups())
    residuals = [math.log(predicted / observed) for predicted in [pred1, pred2]]
    shown1, shown2 = map(number, need(r"остатки примерно \$(0\{,\}\d+)\$ и \$(0\{,\}\d+)\$").groups())
    approx(residuals[0], shown1, .00000005)
    approx(residuals[1], shown2, .00000005)
    coefficient1, exponent1, coefficient2, exponent2 = need(
        r"их вклады равны примерно \$(\d+\{,\}\d+)\\cdot10\^\{(-\d+)\}\$ и\s*\$(\d+\{,\}\d+)\\cdot10\^\{(-\d+)\}\$"
    ).groups()
    approx(huber(residuals[0], .001), number(coefficient1)*10**int(exponent1), .0005e-7)
    approx(huber(residuals[1], .001), number(coefficient2)*10**int(exponent2), .00005e-5)
    assert 'H_\\delta(r)' in TEXT and 'L-BFGS' in TEXT
    # The optimum derived in the prose balances the two marginal terms.
    A, B, alpha, beta, compute = 3.0, 7.0, .34, .28, 6000.0
    optimum = (alpha*A/(beta*B*6**beta))**(1/(alpha+beta)) * compute**(beta/(alpha+beta))
    approx(alpha*A*optimum**-alpha, beta*B*(compute/(6*optimum))**-beta, 1e-12)
    loss = lambda n: A*n**-alpha + B*(compute/(6*n))**-beta
    assert loss(optimum) < loss(optimum*.8) and loss(optimum) < loss(optimum*1.2)
    steepness, center = map(number, need(r"\\exp\((\d+)\(\\ell-(\d+\{,\}\d+)\)\)").groups())
    sigmoid = lambda ell: 1/(1+math.exp(steepness*(ell-center)))
    ell = number(need(r"\\ell_\{\\mathrm\{task\}\}=(\d+\{,\}\d+)").group(1))
    acc = number(need(r"ожидаемая точность равна примерно \$(\d+\{,\}\d+)\$").group(1))
    low, high = map(number, need(r"лежит между \$(\d+\{,\}\d+)\$ и \$(\d+\{,\}\d+)\$").groups())
    acc_low, acc_high = map(number, need(r"интервал от \$(\d+\{,\}\d+)\$ до \$(\d+\{,\}\d+)\$").groups())
    for ell, displayed in [(ell, acc), (low, acc_high), (high, acc_low)]:
        approx(sigmoid(ell), displayed, .0005)
    for width, norm in [(64, 8), (256, 16)]:
        assert math.sqrt(width) == norm and sum([1]*width)/width == 1
    assert 10e6 + 256*50000 == 22.8e6
    approx(3000*2**19/1e9, 1.57, .005)
    RESULTS["fitted_toy_examples"] = {"quadratic_N_B": namespace["n_min"]/1e9,
        "quadratic_D_B": namespace["d_min"]/1e9, "invalid_inputs_rejected": len(invalid),
        "huber_residuals": residuals, "sigmoid_accuracy": sigmoid(.9)}


def batch_and_wsd_examples():
    values = [[number(cell) for cell in row]
              for row in table("| $B/B_{\\mathrm{crit}}$")]
    for ratio, steps, samples in values:
        approx(1 + 1 / ratio, steps, 1e-12)
        approx(1 + ratio, samples, 1e-12)
        approx((steps - 1) * (samples - 1), 1, 1e-12)
    wsd_rows = [[number(cell) for cell in row]
                for row in table("| Всего | До ветвления |")]
    fraction = number(need(r"снижение занимает (\d+)% каждого запуска").group(1)) / 100
    for total, trunk, tail in wsd_rows:
        approx(total, trunk + tail, 1e-12)
        approx(tail, fraction * total, 1e-12)
    shared = max(row[1] for row in wsd_rows) + sum(row[2] for row in wsd_rows)
    independent = sum(row[0] for row in wsd_rows)
    approx(shared, number(need(r"обрабатывают суммарно (\d+) млрд").group(1)), 1e-12)
    approx(independent, number(need(r"запуска обработали бы (\d+) млрд").group(1)), 1e-12)
    RESULTS["critical_batch_ratios"] = values
    RESULTS["wsd_tokens_billions"] = {"shared": shared, "independent": independent}


def deepseek_example():
    lr_coefficient, lr_power = map(float, need(
        r"\\eta_\{\\mathrm\{opt\}\}=([\d.]+)C\^\{(-[\d.]+)\}"
    ).groups())
    batch_coefficient, batch_power = map(float, need(
        r"B_\{\\mathrm\{opt\}\}=([\d.]+)C\^\{([\d.]+)\}"
    ).groups())
    compute_power = int(need(r"Например, \$C=10\^\{(\d+)\}\$").group(1))
    batch = batch_coefficient * (10**compute_power)**batch_power
    lr = lr_coefficient * (10**compute_power)**lr_power
    display_batch, display_batch_power = need(
        r"B_\{\\mathrm\{opt\}\}\\approx([\d.]+)\\cdot10\^\{?(\d+)\}?"
    ).groups()
    display_lr, display_lr_power = need(
        r"\\eta_\{\\mathrm\{opt\}\}\\approx([\d.]+)\\cdot10\^\{(-\d+)\}"
    ).groups()
    approx(batch / 10**int(display_batch_power), float(display_batch), 0.0005)
    approx(lr / 10**int(display_lr_power), float(display_lr), 0.005)
    sequence, displayed_sequences = need(
        r"по (\d+) токенов это около (\d+) последовательностей"
    ).groups()
    approx(batch / int(sequence), int(displayed_sequences), 0.5)
    RESULTS["deepseek_at_actual_compute"] = {
        "compute": 10**compute_power, "batch_tokens": batch,
        "lr": lr, "sequences": batch / int(sequence)
    }


def mup_examples():
    base_width = number(need(r"\$P=(\d+)\$ — базовая ширина").group(1))
    width, exponent, displayed_base, displayed_lr1 = need(
        r"Например, при \$M=(\d+)\$ и \$\\eta_0=2\^\{(-\d+)\}=(0\{,\}\d+)\$.*?"
        r"равна \$(0\{,\}\d+)\$"
    ).groups()
    width2, displayed_lr2 = need(
        r"При \$M=(\d+)\$ первая\s*уменьшается до \$(0\{,\}\d+)\$"
    ).groups()
    base = 2**int(exponent)
    approx(base, number(displayed_base), 1e-12)
    lrs = [base * base_width / int(width), base * base_width / int(width2)]
    approx(lrs[0], number(displayed_lr1), 1e-12)
    approx(lrs[1], number(displayed_lr2), 1e-12)
    baseline_rows = table("| $\\eta_0$ |")
    assert [int(need(r"2\^\{(-\d+)\}", row[0]).group(1))
            for row in baseline_rows] == [-10, -8, -6, -4, -2]
    baseline_losses = [list(column) for column in zip(*[
        [number(x) for x in row[1:]] for row in baseline_rows
    ])]
    large_rows = table("| Параметры; $M$ |")
    assert [row[0] for row in large_rows] == [
        "2 млн; 128", "40 млн; 512", "600 млн; 2048", "10 млрд; 8192"
    ]
    large_losses = [[number(x) for x in row[1:]] for row in large_rows]
    assert all(row.index(min(row)) == 2 for row in baseline_losses)
    assert all(row.index(min(row)) == 1 for row in large_losses)
    # Exact published Table 1/3 transcription, not measurements made by this tool.
    assert baseline_losses == [
        [3.846, 3.743, 3.695, 3.884, 4.143],
        [3.114, 2.993, 2.953, 3.221, 3.506],
        [2.711, 2.553, 2.511, 2.563, 3.244],
    ]
    assert large_losses == [
        [3.791, 3.766, 3.814], [3.016, 2.983, 3.004],
        [2.513, 2.459, 2.466], [2.238, 2.167, 2.169],
    ]
    failure_exponents = [
        [int(need(r"2\^\{(-\d+)\}", cell).group(1)) for cell in row[1:]]
        for row in table("| Изменение |")
    ]
    assert failure_exponents == [[-4, -6, -8], [-4, -4, -6],
                                 [-10, -8, -8], [-8, -6, -6]]
    RESULTS["mup_actual_group_lr"] = {"base": base, "matrix_lrs": lrs,
                                     "source_values_checked": 39}


def inference_example():
    n1, d1 = map(number, need(
        r"модель на (\d+) млрд параметров после (\d+) млрд токенов"
    ).groups())
    n2, d2 = map(number, need(
        r"модель на (\d+) млрд параметров после (\d+,\d+) трлн токенов"
    ).groups())
    rows = table("| Затраты |")
    assert len(rows) == 3 and rows[0][0] == "Обучение"
    assert "$Q=10^{12}$" in rows[1][0]
    q_multiplier = int(need(r"Q=(\d+)\\cdot10\^\{12\}", rows[2][0]).group(1))
    header = need(r"^\| Затраты \|[^\n]*").group(0)
    assert [float(x) for x in re.findall(r"N=(\d+)", header)] == [n1, n2]
    assert "Все численные значения затрат выражены в $10^{22}$ FLOPs." in TEXT
    costs = list(zip(*[[number(x) for x in row[1:]] for row in rows]))
    values = [(n1*1e9, d1*1e9, *costs[0]), (n2*1e9, d2*1e12, *costs[1])]
    for n, d, training, total1, total2 in values:
        approx(6 * n * d / 1e22, training, 0.05)
        approx((6 * n * d + 2 * n * 1e12) / 1e22, total1, 0.05)
        approx((6 * n * d + 2 * n * q_multiplier * 1e12) / 1e22, total2, 0.05)
    n1, d1 = values[0][:2]
    n2, d2 = values[1][:2]
    breakeven = (6 * n2 * d2 - 6 * n1 * d1) / (2 * (n1 - n2))
    coefficient = number(need(r"Q=(\d+\{,\}\d+)\\cdot10\^\{12\}").group(1))
    approx(breakeven / 1e12, coefficient, 0.05)
    RESULTS["inference_breakeven_tokens"] = breakeven


def structure_and_assets():
    source_ids = re.findall(r"^  - (lecture-(?:09|11)-[a-z-]+)$", TEXT, re.M)
    assert len(source_ids) == len(set(source_ids)) == 16
    required_anchors = [
        "scaling-campaign-начинается-до-регрессии",
        "batch-size--часть-экспериментального-режима",
        "learning-rate-нельзя-переносить-молча",
        "как-удешевить-кампанию-wsd",
        "как-читать-опубликованный-scaling-plot",
    ]
    assert all(f'<a id="{anchor}"></a>' in TEXT for anchor in required_anchors)
    assert not re.search(r"^\$$", TEXT, re.M), "Multiline inline math remains"
    assert len(re.findall(r"^\$\$$", TEXT, re.M)) % 2 == 0
    assert len(re.findall("^" + FENCE, TEXT, re.M)) % 2 == 0
    embeds = re.findall(r"!\[\[([^]|]+)", TEXT)
    assert len(embeds) == 26, len(embeds)
    for path in embeds:
        assert (ROOT / path.removeprefix("02 Areas/ML & DL/")).is_file(), path
    records = json.loads((ASSETS / "assets.json").read_text())
    assert len(records) == 21
    for record in records:
        asset = ROOT / record["asset"]
        source = ROOT / record["source_asset"]
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == record["sha256"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == record["source_asset_sha256"]
        assert list(struct.unpack(">II", asset.read_bytes()[16:24])) == record["dimensions"]
        assert str(asset.relative_to(ROOT)) in "\n".join(embeds)
        left, top, right, bottom = record["crop_pixels"]
        assert record["dimensions"] == [right - left, bottom - top]
        assert 0 <= left < right <= record["source_render_dimensions"][0]
        assert 0 <= top < bottom <= record["source_render_dimensions"][1]
    RESULTS["structure"] = {"source_ids": len(source_ids), "images": len(embeds),
                             "asset_hashes_and_dimensions": len(records)}
    new = ROOT / "00 Учебник/Assets/Figures/curated/scaling-2026-10-23"
    new_records = json.loads((new / "assets.json").read_text())
    assert len(new_records) == 3
    for record in new_records:
        for path, digest in [(record["asset"], record["sha256"]),
                             (record["source_asset"], record["source_asset_sha256"]),
                             (record["source_page_snapshot"], record["source_page_sha256"]),
                             (record["license_evidence"], record["license_evidence_sha256"])]:
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
        assert (ROOT / record["asset"]).read_bytes() == (ROOT / record["source_asset"]).read_bytes()
        assert record["asset"] in "\n".join(embeds)
    archive = ROOT / "05 Источники/Originals/scaling-2026-10-23"
    assert "License: CC BY 4.0" in (archive / "porian-v1.html").read_text()
    assert (archive / "mup-LICENSE.txt").read_bytes() == (ROOT / "site/public/licenses/mup-19814971-MIT.txt").read_bytes()
    # Quotes are unchanged source excerpts, not translated or generated text.
    porian = (archive / "porian-v1.html").read_text()
    quote = need(r"^> (However, this led.*?)$").group(1)
    assert quote in re.sub(r"<[^>]*>", "", porian)
    readme = (archive / "mup-README.md").read_text()
    quote = need(r"^> (If instead you see.*?)$").group(1)
    assert quote in " ".join(readme.split())
    svg = (new / "porian-figure1.svg").read_text()
    assert not re.search(r'<(?:script|foreignObject)\b|\bon\w+=', svg)
    assert all(ref.startswith('#') for ref in re.findall(r'(?:xlink:)?href="([^"]+)"', svg))
    RESULTS["structure"]["new_originals"] = len(new_records)


if __name__ == "__main__":
    for check in [mean_example, nonparametric_example, isoflop_example, fitted_examples, batch_and_wsd_examples,
                  deepseek_example, mup_examples, inference_example, structure_and_assets]:
        check()
    print(json.dumps(RESULTS, ensure_ascii=False, indent=2))
    print("PASS: chapter code, displayed arithmetic, source tables and local assets.")
    print("Not run: model training, real-run scaling fit, Chinchilla L-BFGS reproduction, mup training, GPU performance, website rendering.")
    print("Chapter SHA-256:", hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
