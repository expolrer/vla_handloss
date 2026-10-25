"""Run on the download relay; build a complete offline simulation wheelhouse."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def prepare(wheelhouse, isaaclab, index):
    wheelhouse = Path(wheelhouse).resolve()
    package = Path(isaaclab).resolve() / 'source/isaaclab'
    if not (package / 'config/extension.toml').is_file():
        raise ValueError('Expected an IsaacLab v2.3.2 source checkout')
    wheelhouse.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PIP_DEFAULT_TIMEOUT='120', PIP_RETRIES='10')
    pip = [sys.executable, '-m', 'pip']
    subprocess.run(pip + ['download', '--dest', str(wheelhouse), '--index-url', index,
        'setuptools<81', 'wheel', 'toml'], env=env, check=True)
    subprocess.run(pip + ['install', '--no-index', '--find-links', str(wheelhouse),
        'setuptools<81', 'wheel', 'toml'], env=env, check=True)
    import toml
    version = toml.load(package / 'config/extension.toml')['package']['version']
    subprocess.run(pip + ['wheel', '--no-deps', '--no-build-isolation', '--wheel-dir',
        str(wheelhouse), str(package)], env=env, check=True)
    lab_wheel = wheelhouse / f'isaaclab-{version}-py3-none-any.whl'
    if not lab_wheel.exists():
        raise FileNotFoundError(lab_wheel)
    requirements = ['isaacsim[all,extscache]==5.1.0', str(lab_wheel), 'torch==2.7.0+cu128',
        'torchvision==0.22.0+cu128', 'numpy==1.26.0', 'gymnasium==1.2.1',
        'Flask', 'h5py', 'rich', 'hydra-core', 'einops', 'imageio[ffmpeg]',
        'pyyaml', 'psutil', 'scipy', 'setuptools<81', 'wheel', 'toml']
    subprocess.run(pip + ['download', '--no-build-isolation', '--dest', str(wheelhouse),
        '--find-links', str(wheelhouse), '--index-url', index,
        '--extra-index-url', 'https://pypi.nvidia.com',
        '--extra-index-url', 'https://download.pytorch.org/whl/cu128'] + requirements, env=env, check=True)
    for archive in wheelhouse.glob('*.tar.gz'):
        subprocess.run(pip + ['wheel', '--no-deps', '--no-build-isolation', '--wheel-dir',
            str(wheelhouse), str(archive)], env=env, check=True)
    # Re-resolve offline to prove that the target will not need a package index.
    subprocess.run(pip + ['download', '--no-index', '--no-build-isolation',
        '--find-links', str(wheelhouse), '--dest', str(wheelhouse)] + requirements, env=env, check=True)
    files = {}
    for path in sorted(wheelhouse.glob('*.whl')):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(4 * 1024**2), b''):
                h.update(block)
        files[path.name] = dict(bytes=path.stat().st_size, sha256=h.hexdigest())
    (wheelhouse / 'wheel_manifest.json').write_text(json.dumps(dict(files=files,
        epoch=time.time(), isaaclab_distribution=version, isaaclab_git_release='v2.3.2'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wheelhouse', type=Path, required=True)
    parser.add_argument('--isaaclab', type=Path, required=True)
    parser.add_argument('--index-url', default='https://mirrors.aliyun.com/pypi/simple/')
    args = parser.parse_args()
    prepare(args.wheelhouse, args.isaaclab, args.index_url)
