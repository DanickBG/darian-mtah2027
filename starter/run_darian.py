"""Reproduce the exact packaged DaRiAn submission from the provided tactical assets."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', help='bundle root containing tactical/; or set MTAH_DATA_DIR')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/submission/DaRiAn/DaRiAn.csv')
    parser.add_argument('--diagnostics-dir', type=Path, default=ROOT / 'data/derived/darian')
    args = parser.parse_args()
    config_path = ROOT / 'DARIAN_CONFIG.json'
    if not config_path.exists():
        parser.error('DARIAN_CONFIG.json is missing; restore it from this release')
    config = json.loads(config_path.read_text())
    model = ROOT / config['model_relative_path']
    if not model.is_file() or digest(model) != config['model_sha256']:
        parser.error('Packaged model is missing or its SHA-256 differs from DARIAN_CONFIG.json')
    for name, expected in config['source_sha256'].items():
        if digest(ROOT / name) != expected:
            parser.error('Packaged inference source changed: ' + name)
    if config['selected_version'] != 'v4':
        parser.error('This verified release runner supports the selected v4 model only')
    command = [sys.executable, str(ROOT / 'starter/possession_v4.py'), '--model', str(model),
               '--output', str(args.output.resolve()), '--diagnostics-dir', str(args.diagnostics_dir.resolve())]
    if args.data_dir:
        command += ['--data-dir', str(Path(args.data_dir).resolve())]
    subprocess.run(command, cwd=ROOT, check=True)
    # The recorded CSV uses LF. Normalise platform line endings without
    # changing any prediction so byte verification also works on Windows.
    content = args.output.read_bytes()
    normalised = content.replace(b'\r\n', b'\n')
    if content != normalised:
        args.output.write_bytes(normalised)
    subprocess.run([sys.executable, str(ROOT / 'evaluation/validate_submission.py'), str(args.output.resolve())],
                   cwd=ROOT, check=True)
    actual = digest(args.output)
    if actual != config['submission_sha256']:
        raise SystemExit('Valid format, but bytes differ from the recorded submission. Check assets and pinned dependencies.')
    print('Verified: output SHA-256 matches the packaged DaRiAn submission.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
