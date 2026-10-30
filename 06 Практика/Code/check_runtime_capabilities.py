"""Record local capabilities and verify unavailable-CUDA guards do no work."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import torch


def main(output):
    root = Path(__file__).parent
    report = dict(python=platform.python_version(), machine=platform.machine(),
        system=platform.system(), torch=torch.__version__, cuda=torch.cuda.is_available(),
        mps=torch.backends.mps.is_available(), gloo=torch.distributed.is_gloo_available())
    if not report['cuda']:
        with tempfile.TemporaryDirectory(prefix='bookvar-capability-') as temp:
            for name, args in [('text_checkpoint_lab.py', ['--device', 'cuda']),
                               ('ddp_step_lab.py', ['--backend', 'nccl'])]:
                target = Path(temp) / name
                run = subprocess.run([sys.executable, str(root/name), *args, '--output', str(target)],
                    capture_output=True, text=True, env=dict(os.environ, WORLD_SIZE='2'), timeout=30)
                assert run.returncode != 0 and not target.exists(), run
                message = (run.stderr+run.stdout).strip()
                assert 'No NCCL result' in message or 'CUDA unavailable' in message, message
                report[name] = dict(exit_code=run.returncode, message=message, result_created=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
