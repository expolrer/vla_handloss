"""Offline installer. Download wheels on the relay, never on the evaluation server."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hand_pipeline.config import load
from hand_pipeline.records import atomic


def verify_wheels(wheelhouse):
    root = Path(wheelhouse).resolve()
    manifest = root / 'wheel_manifest.json'
    value = json.loads(manifest.read_text())
    for relative, info in value['files'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Wheel manifest escapes the wheelhouse')
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(4 * 1024**2), b''):
                h.update(block)
        if path.stat().st_size != info['bytes'] or h.hexdigest() != info['sha256']:
            raise ValueError('Offline wheel SHA256 mismatch: ' + relative)
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role', choices=['inference', 'simulation'])
    parser.add_argument('--wheelhouse', type=Path)
    parser.add_argument('--isaaclab', type=Path)
    parser.add_argument('--reuse-site-packages', type=Path)
    args = parser.parse_args()
    project = load()
    env = project['paths'][args.role + '_python'].parent.parent
    venv.EnvBuilder(with_pip=args.role == 'simulation').create(env)
    python = env / 'bin/python'
    if args.role == 'inference':
        if args.reuse_site_packages is None or not args.reuse_site_packages.is_dir():
            raise SystemExit('Provide a tested inference site-packages directory; it stays read-only')
        site = env / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
        repo = project['paths']['bench2dex']
        (site / 'vla_handloss.pth').write_text('\n'.join(map(str, [ROOT, repo,
            repo / 'policy/pi05/src', repo / 'policy/pi05/packages/openpi-client/src',
            args.reuse_site_packages.resolve()])) + '\n')
    else:
        if args.wheelhouse is None or args.isaaclab is None:
            raise SystemExit('Simulation setup requires an offline wheelhouse and IsaacLab v2.3.2 source')
        wheel_manifest_sha = verify_wheels(args.wheelhouse)
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', '--find-links', str(args.wheelhouse),
            'isaacsim[all,extscache]==5.1.0', 'torch==2.7.0+cu128', 'torchvision==0.22.0+cu128',
            'numpy==1.26.0', 'h5py', 'Flask', 'gymnasium==1.2.1', 'toml', 'wheel', 'setuptools<81'], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', '--find-links', str(args.wheelhouse),
            '--no-build-isolation', '-e', str(args.isaaclab / 'source/isaaclab')], check=True)
        subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
        packages = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
        atomic(project['paths']['runtime'] / 'simulation_packages.json', dict(freeze=packages,
            wheel_manifest_sha256=wheel_manifest_sha,
            gpu_render_test='PENDING', installer='vla_handloss/scripts/setup_environment.py'))


if __name__ == '__main__':
    main()
