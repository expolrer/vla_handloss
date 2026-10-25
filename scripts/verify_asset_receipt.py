"""Run on the target after upload. A source manifest alone is not a readiness proof."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hand_pipeline.records import atomic


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--asset-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    args = p.parse_args()
    value = json.loads(args.manifest.read_text())
    if value.get('usd_transitive_check') != 'PASSED' or value.get('unresolved'):
        raise SystemExit('Source USD dependency audit has not passed')
    root = args.asset_root.resolve()
    for rel, info in value['files'].items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            raise ValueError(rel)
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(4 * 1024**2), b''):
                h.update(block)
        if path.stat().st_size != info['bytes'] or h.hexdigest() != info['sha256']:
            raise ValueError('Target asset SHA256 mismatch: ' + rel)
    receipt = dict(files=len(value['files']), bytes=sum(v['bytes'] for v in value['files'].values()),
        source_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        usd_transitive_check='PASSED', verified_epoch=time.time(), gpu_render_test='PENDING')
    atomic(args.runtime / 'ANCHOR_ASSETS_VERIFIED.json', receipt)
    atomic(args.runtime / 'ASSETS_VERIFIED.json', receipt)
    print(json.dumps(receipt))
