"""Install the captured model/robot sources into a pinned upstream checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, default=ROOT / 'workspaces/Bench2Dex')
    args = p.parse_args()
    destination = args.workspace.resolve()
    if not (destination / 'run_policy.py').is_file():
        raise SystemExit('Install the pinned Bench2Dex checkout first')
    sources = ROOT / 'workspaces/architectures/pi05'
    records = ROOT / 'records/model_overlay'
    records.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for tree, prefix in [('overlay', 'policy/pi05'), ('bench2dex_overlay', '')]:
        for source in (sources / tree).rglob('*'):
            if not source.is_file():
                continue
            relative = Path(prefix) / source.relative_to(sources / tree)
            target = destination / relative
            backup = records / 'upstream_originals' / relative
            if target.exists() and not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (records / 'installed_sha256.json').write_text(json.dumps(hashes, indent=2))
    print(json.dumps(dict(installed_files=len(hashes), records=str(records))))


if __name__ == '__main__':
    main()
