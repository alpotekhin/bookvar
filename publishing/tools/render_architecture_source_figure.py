#!/usr/bin/env python3
"""Render the original Lab 06 normalized-budget plot; no replacement chart."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import plotly
import plotly.graph_objects as go

from check_architecture_lab import ROOT, SOURCES, original_engine, original_variants, plotting_nodes, plotting_source

OUTPUT = ROOT / '00 Учебник/Assets/Figures/curated/architecture-lab-2026-10-26'


def main():
    engine = original_engine()
    values = original_variants()['iphone']
    profile = engine.architecture_track_profile(
        SimpleNamespace(track_id='iphone', label='iphone'),
        SimpleNamespace(**values, hardware_ref='unresolved-registry', system_ref=None),
        SimpleNamespace(), SimpleNamespace())
    signature = engine.architecture_signature(profile)
    style_tree = ast.parse((SOURCES / 'style-45ecc8d.py').read_text())
    style_nodes = [n for n in style_tree.body
                   if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'COLORS' for t in n.targets))
                   or (isinstance(n, ast.FunctionDef) and n.name == 'apply_plotly_theme')]
    assert len(style_nodes) == 2
    namespace = {'go': go, 'v1_06_architecture': profile, 'v1_06_signature': signature}
    exec(compile(ast.Module(body=style_nodes, type_ignores=[]), 'original-style', 'exec'), namespace)
    nodes = plotting_nodes()
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'original-budget-plot', 'exec'), namespace)
    figure = namespace['_signature_fig']
    assert len(figure.data) == 3 and len(figure.layout.shapes) == 1
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Export dimensions only; source artists, text, values and theme are unchanged.
    figure.write_image(OUTPUT / 'harvard-architecture-budgets.svg', width=1000, height=480)
    figure.write_image('/var/tmp/bookvar-architecture-budgets.png', width=1000, height=480, scale=3)
    result = ROOT / '06 Практика/Results/architecture-lab-2026-10-26'
    result.mkdir(parents=True, exist_ok=True)
    figure.write_json(result / 'source-plot.json')
    print(json.dumps({'plotly': plotly.__version__, 'original_plotting_statements': len(nodes),
                      'trace_names': [t.name for t in figure.data],
                      'trace_values': [list(t.y) for t in figure.data],
                      'export_canvas': [1000, 480], 'png_scale': 3,
                      'source_plot_source_sha256': hashlib.sha256(plotting_source().encode()).hexdigest()},
                     indent=2))


if __name__ == '__main__':
    main()
