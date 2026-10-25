"""Evaluation launch records are independent from checkpoint retention."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def snapshot(project, directory, config, command):
    directory.mkdir(parents=True, exist_ok=False)
    atomic(directory / 'config.json', config)
    atomic(directory / 'invocation.json', dict(argv=command, epoch=time.time()))
    atomic(directory / 'source_sha256.json', {str(p.relative_to(project['root'])):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in project['root'].rglob('*.py')
        if not any(s in p.relative_to(project['root']).parts for s in ('workspaces', '.venv', 'records', 'results'))})
    try:
        result = subprocess.check_output(['nvidia-smi', '-q'], text=True, timeout=20)
        (directory / 'hardware.txt').write_text(result)
    except (OSError, subprocess.SubprocessError) as exc:
        atomic(directory / 'hardware_error.json', dict(error=str(exc)))
    for name in ('inference_python', 'simulation_python'):
        python = project['paths'][name]
        if python.exists():
            output = subprocess.run([str(python), '-c',
                'import importlib.metadata,json;print(json.dumps({d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}))'],
                text=True, capture_output=True, timeout=60)
            (directory / (name + '_packages.json')).write_text(output.stdout)
