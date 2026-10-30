"""Launch an original Harvard Marimo lab on loopback with its pinned modules."""
import argparse
from pathlib import Path
import subprocess
import sys

from prepare_harvard_runtime import main as prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lab', default='vol2/lab_10_inference.py')
    parser.add_argument('--directory', type=Path, default=Path('/tmp/bookvar-harvard-45ecc8d'))
    parser.add_argument('--port', type=int, default=4338)
    args = parser.parse_args()
    if len(Path(args.lab).parts) != 2 or Path(args.lab).parts[0] not in ('vol1', 'vol2'):
        parser.error('--lab must be vol1/lab_*.py or vol2/lab_*.py')
    directory = args.directory.resolve()
    prepare(directory)
    lab = directory / 'labs' / args.lab
    if not lab.is_file() or not lab.name.startswith('lab_') or lab.suffix != '.py':
        parser.error('No such lab in the pinned source')
    dependencies = ['marimo==0.23.1', 'numpy==2.3.5', 'plotly==6.5.0',
        'pandas==2.3.3', 'pydantic==2.12.4', 'pint==0.25.2',
        str(directory/'wheels/mlsysim-0.1.2-py3-none-any.whl'),
        str(directory/'wheels/mlsysbook_labs-0.1.0-py3-none-any.whl')]
    command = ['uv', 'run', '--no-project']
    for dependency in dependencies:
        command += ['--with', dependency]
    command += ['marimo', 'run', str(lab), '--host', '127.0.0.1',
                '--port', str(args.port), '--headless']
    raise SystemExit(subprocess.call(command))


if __name__ == '__main__':
    main()
