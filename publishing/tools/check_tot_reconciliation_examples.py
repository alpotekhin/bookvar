"""Replay pinned ToT selection/state code with explicit responses; no LLM."""
import ast
import contextlib
import io
import json
import math
import os
from pathlib import Path
import re

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / '05 Источники/Originals/tot-reconciliation-2026-10-30'
CHAPTER = ROOT / '00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute.md'


def definitions(path, names, scope):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)


def main():
    scope = {}
    blocks = re.findall(r'^' + chr(96)*3 + r'python\n(.*?)^' + chr(96)*3 + r'\s*$', CHAPTER.read_text(), re.M | re.S)
    displayed = [block for block in blocks if 'def crossword_priorities(' in block]
    assert len(displayed) == 1
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(displayed[0], 'displayed-crossword-example', 'exec'), scope)

    # Source vote parser, rather than a reimplementation.
    text_scope = {'Task': object, 're': re}
    definitions(PACKET / 'tot-text.py', {'TextTask'}, text_scope)
    vote = text_scope['TextTask'].vote_outputs_unwrap
    with contextlib.redirect_stdout(io.StringIO()):
        assert vote(['The best choice is 2']*3 +
                    ['The best choice is 1', 'unparseable'], 5) == [1, 3, 0, 0, 0]
        assert vote(['The best choice is 1', 'The best choice is 2'], 5) == [1, 1, 0, 0, 0]
        assert vote(['The best choice is 9'], 5) == [0]*5

    # Extract only native definitions, never execute notebook setup/experiment loops.
    notebook = json.loads((PACKET / 'tot-crosswords-dfs.ipynb').read_text())
    cell_text = '\n'.join(''.join(cell['source']) for cell in notebook['cells']
                          if cell['cell_type'] == 'code')
    functions = [node for cell in notebook['cells'] if cell['cell_type'] == 'code'
                 and not ''.join(cell['source']).startswith('cd ')
                 for node in ast.parse(''.join(cell['source'])).body
                 if isinstance(node, ast.FunctionDef)]
    native = {'re': re, 'confidence_to_value':
              {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'native-dfs-functions', 'exec'), native)
    parsed = native['parse_response']('h4. salon (certain)\nh3. grand (high)\ninvalid')
    assert parsed == [('h4. salon', 1), ('h3. grand', 0.5)]
    assert native['parse_response']('h6. apple (certain)') is None
    assert "n=8" in cell_text and "max_per_state=3" in cell_text

    def status_reply(prompt):
        # Deliberately imposed label, not a linguistic or dictionary judgment.
        pattern = prompt.splitlines()[-1].split(': ')[-1].replace(' ', '').lower()
        return ['impossible' if pattern == 'tm_s_' else 'maybe']

    prompts = {}
    exec(compile((PACKET / 'tot-crosswords-prompts.py').read_text(), 'native-prompts', 'exec'), prompts)
    env_scope = {'os': os, 'json': json, 'DATA_PATH': str(PACKET), 'gpt': status_reply,
                 'value_prompt': prompts['value_prompt']}
    definitions(PACKET / 'tot-crosswords.py', {'MiniCrosswordsEnv'}, env_scope)
    env = env_scope['MiniCrosswordsEnv'](str(PACKET / 'tot-crosswords-data.json'))
    board = list('TASKSMOTOR_______________')
    assert len(board) == 25
    status = [1, 1] + [0]*8
    env.reset(0, board=board.copy(), status=status.copy(), steps=2)
    assert env.ans[5] == 'TM___'
    env.step('h4. salon')
    assert env.ans[5] == 'TM_S_' and 2 not in env.status
    env.reset(0, board=board.copy(), status=status.copy(), steps=2)
    env.step('v1. abcde')
    assert 2 in env.status  # Environment allows overwrite; controller rejects it.

    env.reset(0, board=board.copy(), status=status.copy(), steps=2)
    expanded = []
    def candidates(current):
        expanded.append(current.render_board())
        if current.steps == 2:
            return {'v1. abcde': 4, 'h4. salon': 1.3, 'h3. grand': 1.0}
        return {}
    native['get_candidates_to_scores'] = candidates
    infos, actions = [], []
    with contextlib.redirect_stdout(io.StringIO()):
        native['dfs'](env, actions, infos, 100, True, 3)
    assert [row['actions'][-1] for row in infos] == ['h4. salon', 'h3. grand']
    assert infos[0]['count']['impossible'] >= 1
    assert len(expanded) == 2  # Root and GRAND; pruned SALON was not expanded.
    assert 'GRAND' in expanded[1] and 'SALON' not in expanded[1]
    assert actions == [] and env.board == board and env.status == status and env.steps == 2
    expanded_first_run = len(expanded)

    # Preserve the pinned notebook's early-exit behavior rather than silently
    # claiming it restores the root at the child-cap break as well.
    env.reset(0, board=board.copy(), status=status.copy(), steps=2)
    capped_infos, capped_actions = [], []
    with contextlib.redirect_stdout(io.StringIO()):
        native['dfs'](env, capped_actions, capped_infos, 100, True, 1)
    assert len(capped_infos) == 1 and capped_actions == []
    assert env.ans[2] == 'GRAND' and env.board != board

    figures = json.loads((ROOT / '00 Учебник/Assets/Figures/curated/tot-reconciliation-2026-10-30/assets.json').read_text())
    for figure in figures:
        with Image.open(ROOT / figure['source_page_snapshot']) as page, Image.open(ROOT / figure['asset']) as image:
            assert page.size == tuple(figure['render_page_size_px'])
            assert image.size == (figure['width'], figure['height'])
            assert ImageChops.difference(page.crop(figure['crop_box_px']), image).getbbox() is None
    print(json.dumps({'displayed_programs': 1, 'native_vote_cases': 3,
                      'native_parser_negative_cases': 2, 'native_logged_children': len(infos),
                      'native_expanded_states': expanded_first_run, 'parent_state_restored': True,
                      'child_cap_exit_gap_reproduced': True,
                      'exact_original_figures': len(figures), 'llm_run': False,
                      'crossword_benchmark': False}))


if __name__ == '__main__':
    main()
