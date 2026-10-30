"""Execute original Marimo applications, separately from static source extraction.

Each application gets a subprocess and isolated HOME for its design ledger.
Default controls exercise initial rendering, not completion of the assignments.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main(directory, output):
    records = []
    for source in sorted((directory / 'labs').glob('vol*/lab_*.py')):
        with tempfile.TemporaryDirectory(prefix='bookvar-marimo-home-') as home:
            code = ('import json,runpy; app=runpy.run_path(' + repr(str(source)) + ')["app"]; '
                    'outputs,defs=app.run(); print("BOOKVAR_RESULT="+json.dumps('
                    '{"outputs":len(outputs),"definitions":len(defs),'
                    '"tabs":sum(type(v).__name__=="tabs" for v in defs.values()),'
                    '"controls":sum(hasattr(v,"value") for v in defs.values())}))')
            started = time.monotonic()
            process = subprocess.run([sys.executable, '-c', code], capture_output=True,
                text=True, timeout=90, cwd=home, env=dict(os.environ, HOME=home))
            marker = [s[len('BOOKVAR_RESULT='):] for s in process.stdout.splitlines()
                      if s.startswith('BOOKVAR_RESULT=')]
            row = dict(source=str(source.relative_to(directory)),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                exit_code=process.returncode, seconds=time.monotonic()-started,
                result=json.loads(marker[-1]) if marker else None,
                stderr=process.stderr[-6000:])
            records.append(row)
            print(row['source'], row['exit_code'], row['result'], flush=True)
    report = dict(scope='original Marimo default-state execution; not all prediction gates or UI interactions',
                  pinned_commit='45ecc8d82fcae70c149cdce550d3b3d3411df913',
                  passed=sum(r['exit_code'] == 0 for r in records), total=len(records), records=records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    if report['passed'] != report['total']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.directory.resolve(), args.output.resolve())
