"""Adopt an existing verified runtime without copying checkpoints or entire environments."""
from pathlib import Path
import hashlib
import json
import shutil
import yaml
from .records import atomic


def adopt(project, runtime):
    runtime = Path(runtime).resolve()
    if not (runtime / 'common/Bench2Dex/run_policy.py').is_file():
        raise ValueError('Runtime is missing the verified Bench2Dex source')
    root = project['root']
    links = {'workspaces/runtime': runtime, 'workspaces/Bench2Dex': runtime / 'common/Bench2Dex',
        'workspaces/envs': runtime / 'envs', 'workspaces/dex2bench_dataset': runtime / 'common/dex2bench_dataset',
        'checkpoints': runtime / 'models'}
    for rel, source in links.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            if path.resolve() != source:
                raise ValueError(f'Existing link points somewhere else: {path}')
        elif path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
        else:
            path.symlink_to(source, target_is_directory=True)
    package = runtime / 'common/Bench2Dex/policy/vla_handloss'
    package.mkdir(exist_ok=True)
    (package / '__init__.py').touch()
    (package / 'deploy_policy.py').write_text('from hand_pipeline.policy import *\n')
    atomic(root / 'state/runtime_adoption.json', dict(runtime=str(runtime),
        copied_weights=False, copied_environment=False))


def check(project):
    paths = project['paths']
    result = {k: p.exists() for k, p in paths.items() if k in
        ('runtime', 'bench2dex', 'inference_python', 'simulation_python', 'models', 'anchors')}
    result['core_assets_verified'] = (paths['runtime'] / 'ASSETS_VERIFIED.json').exists()
    result['anchor_assets_verified'] = (paths['runtime'] / 'ANCHOR_ASSETS_VERIFIED.json').exists()
    result['simulation_packages_installed'] = (paths['runtime'] / 'simulation_packages.json').exists()
    source_manifest = project['root'] / 'workspaces/architectures/pi05/source_sha256.json'
    result['model_overlay_matches_training'] = False
    if source_manifest.exists() and paths['bench2dex'].exists():
        expected = json.loads(source_manifest.read_text())
        result['model_overlay_matches_training'] = all((paths['bench2dex'] / rel).is_file() and
            hashlib.sha256((paths['bench2dex'] / rel).read_bytes()).hexdigest() == sha
            for rel, sha in expected.items())
    return result
