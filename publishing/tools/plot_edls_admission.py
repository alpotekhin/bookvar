"""Plot recorded CPU phase intervals, not an illustrative/simulated timeline.

uv run --with matplotlib==3.10.7 python publishing/tools/plot_edls_admission.py DATA.json OUTPUT.png
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

data = json.loads(Path(sys.argv[1]).read_text())
policies = ['single', 'refill', 'wait_bound']
selected = [next(r for r in data['runs'] if r['policy'] == policy and r['slots'] == 2
                 and r['workload'] == 'decode_heavy' and r['repeat'] == 0)
            for policy in policies]
plt.rcParams.update({'font.size': 18, 'axes.titlesize': 20, 'axes.labelsize': 18,
                     'xtick.labelsize': 16, 'ytick.labelsize': 18,
                     'font.family': 'DejaVu Sans'})
fig, ax = plt.subplots(figsize=(12, 6.6), dpi=150)
styles = {'prefill': ('#2d718e', '///'), 'decode': ('#e5a34b', '..')}
for lane, run in enumerate(selected):
    for phase in run['phases']:
        assert phase['end'] >= phase['start'] and phase['error'] is None
        color, hatch = styles[phase['stage']]
        start = phase['start'] * 1000
        width = (phase['end'] - phase['start']) * 1000
        ax.barh(lane, width, left=start, height=.42, color=color, hatch=hatch,
                edgecolor='#303b43', linewidth=.45)
    ax.scatter([r['submitted']*1000 for r in run['receipts']],
               [lane+.31]*len(run['receipts']), marker='v', s=28, color='#303b43',
               zorder=3)
ax.set_yticks(range(3), policies)
ax.invert_yaxis()
ax.set_ylim(2.65, -.65)
ax.set_xlim(left=0)
ax.set_xlabel('Elapsed wall-clock time (ms)')
ax.set_title('Recorded CPU phase calls\nDecode-heavy workload · 2 slots · repeat 0', pad=20)
ax.grid(axis='x', color='#dce3e7', linewidth=.8)
ax.set_axisbelow(True)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.tick_params(axis='y', length=0, pad=12)
handles = [Patch(facecolor=styles[s][0], hatch=styles[s][1], edgecolor='#303b43',
                 label=s.capitalize()) for s in ['prefill', 'decode']]
handles.append(plt.Line2D([], [], color='#303b43', marker='v', linestyle='None',
                          label='Producer submission'))
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, .105),
           ncol=3, frameon=False, fontsize=15)
fig.text(.52, .055, 'Random 22,720-parameter Qwen3 · CPU FP32 · one PyTorch thread',
         ha='center', fontsize=14, color='#43535e')
fig.text(.52, .02, 'Intervals include synchronous engine work and wrapper validation; not GPU kernel times.',
         ha='center', fontsize=12, color='#43535e')
fig.subplots_adjust(left=.18, right=.98, top=.78, bottom=.30)
fig.savefig(sys.argv[2], facecolor='white', metadata={'Software': 'Bookvar recorded CPU phase plot'})
plt.close(fig)
