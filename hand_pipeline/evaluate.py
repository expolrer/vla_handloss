"""One GPU: persistent model server, official clients, transactional episode resume."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import uuid
import yaml
from .config import fingerprint, protocol
from .journal import Journal
from .records import atomic, snapshot


def receipt(project, experiment):
    checkpoint = project['paths']['models'] / experiment['run_name'] / str(project['evaluation']['checkpoint_step'])
    marker = checkpoint / 'VERIFIED.json'
    if not marker.exists():
        return None
    data = json.loads(marker.read_text())
    digest = hashlib.sha256((checkpoint / 'SHA256.json').read_bytes()).hexdigest()
    if data.get('manifest_sha256') != digest or not data.get('sha256_all_files'):
        raise ValueError('Invalid final checkpoint receipt')
    captured = json.loads((checkpoint / 'deployment.json').read_text())['config']
    for field in ('action_dim', 'robot_key', 'prompt', 'hand_branch', 'hand_loss_weight'):
        if captured[field] != experiment[field]:
            raise ValueError('Checkpoint/experiment configuration mismatch: ' + field)
    if tuple(captured['hand_indices']) != tuple(range(experiment['arm_dim'], experiment['action_dim'])):
        raise ValueError('Checkpoint hand coordinate mapping mismatch')
    if captured.get('branch_scope', 'hand') != experiment.get('branch_scope', 'hand'):
        raise ValueError('Checkpoint branch scope mismatch')
    if captured.get('hand_branch_width', 256) != experiment.get('hand_branch_width', 256):
        raise ValueError('Checkpoint branch width mismatch')
    if captured['max_token_len'] != project['training']['max_token_len']:
        raise ValueError('Checkpoint tokenizer length mismatch')
    if captured['training']['seed'] != project['training']['seed']:
        raise ValueError('Checkpoint training seed mismatch')
    return checkpoint, digest


def native_config(project, experiment, profile, checkpoint, output, port):
    e = project['evaluation']
    return dict(policy_name='vla_handloss', policy_display_name=experiment['run_name'],
        task_name=experiment['task'], task_config='replay', robot_key=experiment['robot_key'],
        use_active_dof=True, train_config_name=experiment['run_name'], checkpoint_path=str(checkpoint),
        action_dim=experiment['action_dim'], state_dim=experiment['action_dim'],
        train_action_horizon=e['action_horizon'], eval_action_horizon=e['action_horizon'],
        host='127.0.0.1', port=port, seed=e['base_seed'], generalization_profile=profile,
        generalization_split=e['generalization_split'],
        anchor_dir=str(project['paths']['anchors'] / experiment['task']),
        episode_steps=e['episode_steps'], warmup_steps=e['warmup_steps'],
        num_episodes=1, headless=True, output_dir=str(output), record_all=e['record_all'],
        record_dir=str(output / 'recordings'), append_record_dir=True, append_output=True)


def child_env(project, gpu, audit):
    root, paths = project['root'], project['paths']
    repo = paths['bench2dex']
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED='0',
        PYTHONPATH=os.pathsep.join(map(str, (root, repo, repo / 'policy/pi05/src',
                                           repo / 'policy/pi05/packages/openpi-client/src'))),
        OPENPI_DATA_HOME=str(paths['runtime'] / 'common/cache/openpi'), HF_HUB_OFFLINE='1',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_PYTHON_CLIENT_MEM_FRACTION=str(project['evaluation']['xla_memory_fraction']),
        OMP_NUM_THREADS='4', HAND_CONTROL_AUDIT_PATH=str(audit))


def stop(process):
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def wait_server(process, port, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Model server exited: {process.returncode}')
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError('Model server did not become ready')


def classify(native, observations):
    if native.get('error'):
        return 'RUNTIME_FAILURE'
    target = max((r.get('target_delta_l2', 0) for r in observations), default=0)
    motion = max((r.get('qpos_delta_l2', 0) for r in observations), default=0)
    action_count = sum(r.get('kind') == 'action' for r in observations)
    if not action_count or (target > 1e-4 and motion < 1e-6):
        return 'ACTION_EXECUTION_INVALID'
    return 'SUCCESS' if native.get('success') is True else 'TASK_FAILURE'


def episode_index(row):
    for key in ('episode', 'episode_index', 'episode_idx'):
        if key in row:
            return int(row[key])
    raise ValueError('Native result does not identify its episode')


def run(project, experiment, profile, gpu, smoke=False, stop_requested=None):
    ready = receipt(project, experiment)
    if ready is None:
        raise FileNotFoundError('Waiting for final checkpoint SHA256 verification')
    checkpoint, manifest_sha = ready
    p, e = project['paths'], project['evaluation']
    if not p['simulation_python'].is_file() or not (p['runtime'] / 'simulation_packages.json').exists():
        raise FileNotFoundError('Simulation environment is not installed')
    if e['require_anchor_assets'] and not (p['runtime'] / 'ANCHOR_ASSETS_VERIFIED.json').exists():
        raise FileNotFoundError('Anchor background/texture/clutter dependencies have not been verified')
    job = experiment['run_name'] + '__' + profile + ('__smoke' if smoke else '')
    key = fingerprint(protocol(project, experiment, profile, manifest_sha))
    journal = Journal(p['state'] / 'evaluation.sqlite')
    journal.register(job, key)
    wanted = [1] if smoke else list(range(1, int(e['episodes']) + 1))
    todo = [episode for episode in wanted if episode not in journal.completed(job)]
    if not todo:
        result = journal.summary(job)
        journal.close()
        return result
    if not smoke and e['require_smoke']:
        proof = p['state'] / 'smoke' / (job + '.json')
        if not proof.exists() or json.loads(proof.read_text()).get('protocol_sha256') != key:
            raise RuntimeError('Matching model/control smoke test has not passed')
    output = p['results'] / job
    output.mkdir(parents=True, exist_ok=True)
    directory = p['records'] / job / (time.strftime('%Y%m%d_%H%M%S', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    port = int(e['port_base']) + int(gpu)
    config = native_config(project, experiment, profile, checkpoint, output, port)
    server_command = [str(p['inference_python']), 'script/policy_model_server.py', '--config', str(directory / 'native.yaml')]
    snapshot(project, directory, dict(protocol=protocol(project, experiment, profile, manifest_sha),
        protocol_sha256=key, mode='smoke' if smoke else 'production'), server_command)
    (directory / 'native.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    audit_path = directory / 'control_audit.jsonl'
    env = child_env(project, gpu, audit_path)
    server = client = None
    try:
        with (directory / 'server.log').open('w') as server_log:
            server = subprocess.Popen(server_command, cwd=p['bench2dex'], env=env,
                stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            wait_server(server, port, e['ready_timeout_seconds'])
            for episode in todo:
                if stop_requested and stop_requested():
                    break
                config.update(start_episode=episode, num_episodes=1)
                episode_config = directory / f'episode_{episode:04d}.yaml'
                episode_config.write_text(yaml.safe_dump(config, sort_keys=False))
                command = [str(p['simulation_python']), 'script/eval_policy_client.py', '--config', str(episode_config)]
                atomic(directory / f'episode_{episode:04d}_invocation.json', dict(argv=command, episode=episode))
                before = audit_path.stat().st_size if audit_path.exists() else 0
                with (directory / f'episode_{episode:04d}.log').open('w') as log:
                    client = subprocess.Popen(command, cwd=p['bench2dex'], env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = client.wait(timeout=e['episode_timeout_seconds'])
                    except subprocess.TimeoutExpired:
                        stop(client)
                        raise RuntimeError('Episode timed out; not a task failure')
                if code:
                    raise RuntimeError(f'Simulator exited with {code}; not a task failure')
                rows = [json.loads(line) for line in (output / 'per_episode.jsonl').read_text().splitlines() if line.strip()]
                candidates = [row for row in rows if episode_index(row) == episode]
                if not candidates:
                    raise RuntimeError('Simulator produced no native result for the requested episode')
                row = candidates[-1]
                with audit_path.open() as f:
                    f.seek(before)
                    observations = [json.loads(line) for line in f if line.strip()]
                status = classify(row, observations)
                journal.commit(job, episode, status, row)
                if status not in ('SUCCESS', 'TASK_FAILURE'):
                    raise RuntimeError(status)
                atomic(output / 'validated_summary.json', dict(**journal.summary(job),
                    protocol_sha256=key, success_checker_audit='NATIVE_OFFICIAL; EXPERT_REPLAY_AUDIT_PENDING'))
            if smoke and 1 in journal.completed(job):
                formal_job = experiment['run_name'] + '__' + profile
                atomic(p['state'] / 'smoke' / (formal_job + '.json'), dict(protocol_sha256=key,
                    control_passed=True, record=str(directory), native_success=journal.summary(job)['successes']))
            return journal.summary(job)
    finally:
        stop(client)
        stop(server)
        journal.close()
