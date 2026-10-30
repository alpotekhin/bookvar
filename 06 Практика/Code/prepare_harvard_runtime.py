"""Fetch only the pinned Harvard labs and their two small runtime wheels.

No repository clone, model weights, or datasets. Original files are immutable;
the manifest records Git blob IDs and SHA256. Use --directory outside the vault.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = '45ecc8d82fcae70c149cdce550d3b3d3411df913'
REPO = 'https://raw.githubusercontent.com/harvard-edge/cs249r_book/' + COMMIT


def main(directory):
    tree_url = f'https://api.github.com/repos/harvard-edge/cs249r_book/git/trees/{COMMIT}?recursive=1'
    with urllib.request.urlopen(tree_url, timeout=30) as response:
        tree = json.load(response)['tree']
    files = [r for r in tree if r['type'] == 'blob' and (
        r['path'] in ('labs/bootstrap.py', 'LICENSE.md',
                      'wheels/mlsysim-0.1.2-py3-none-any.whl',
                      'wheels/mlsysbook_labs-0.1.0-py3-none-any.whl')
        or r['path'].startswith(('labs/mlsysbook_labs/', 'mlsysim/mlsysim/'))
        or (r['path'].startswith(('labs/vol1/lab_', 'labs/vol2/lab_'))
            and r['path'].endswith('.py')))]
    def fetch(item):
        path = directory / item['path']
        if path.exists():
            data = path.read_bytes()
        else:
            with urllib.request.urlopen(REPO + '/' + item['path'], timeout=30) as response:
                data = response.read()
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if blob != item['sha']:
            raise ValueError('Refusing to overwrite a changed source: ' + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return dict(path=item['path'], bytes=len(data), git_blob=blob,
                    sha256=hashlib.sha256(data).hexdigest())
    with ThreadPoolExecutor(max_workers=6) as executor:
        manifest = list(executor.map(fetch, files))
    result = dict(commit=COMMIT, files=manifest, bytes=sum(r['bytes'] for r in manifest))
    (directory / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'files': len(files), 'bytes': result['bytes'], 'directory': str(directory)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    main(parser.parse_args().directory.resolve())
