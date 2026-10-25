"""Dynamic GPU dispatch; readiness is explicit and blocked jobs never become zero scores."""
import concurrent.futures
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
from .config import experiments
from .evaluate import receipt, run
from .journal import Journal
from .records import atomic
from .setup import check


def gpu_idle(gpu, threshold):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    usage = {int(a): int(b) for a, b in (line.split(',') for line in output.strip().splitlines())}
    return gpu in usage and usage[gpu] <= threshold


def execute(project, gpus, authorized=False, once=False):
    jobs = [(experiment, profile) for experiment in experiments(project)
            for profile in project['evaluation']['profiles']]
    if not authorized:
        print(json.dumps(dict(execute=False, jobs=len(jobs), gpus=gpus,
            experiments=[x['run_name'] for x, _ in jobs]), indent=2))
        return
    state = project['paths']['state']
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop_event = threading.Event()
        def stop_requested(signum, frame):
            stop_event.set()
        signal.signal(signal.SIGTERM, stop_requested)
        signal.signal(signal.SIGINT, stop_requested)
        retry_after, active = {}, {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            while not stop_event.is_set():
                for gpu, (future, key) in list(active.items()):
                    if future.done():
                        try:
                            future.result()
                            retry_after.pop(key, None)
                        except Exception as exc:
                            retry_after[key] = dict(until=time.time() + 300, error=f'{type(exc).__name__}: {exc}')
                            atomic(state / 'errors' / (key + '.json'), retry_after[key])
                        del active[gpu]
                readiness = check(project)
                required = ('bench2dex', 'inference_python', 'simulation_python', 'models',
                            'anchors', 'simulation_packages_installed', 'core_assets_verified',
                            'model_overlay_matches_training')
                ready = all(readiness[k] for k in required)
                if project['evaluation']['require_anchor_assets']:
                    ready = ready and readiness['anchor_assets_verified']
                journal = Journal(state / 'evaluation.sqlite')
                pending = []
                completed = []
                for experiment, profile in jobs:
                    key = experiment['run_name'] + '__' + profile
                    if len(journal.completed(key)) >= project['evaluation']['episodes']:
                        completed.append(key)
                    else:
                        pending.append((experiment, profile, key))
                journal.close()
                for gpu in gpus:
                    if gpu in active or not ready or not gpu_idle(gpu, project['evaluation']['gpu_memory_idle_mb']):
                        continue
                    active_keys = {value[1] for value in active.values()}
                    for experiment, profile, key in pending:
                        if key in active_keys or retry_after.get(key, {}).get('until', 0) > time.time():
                            continue
                        try:
                            available = receipt(project, experiment)
                        except Exception as exc:
                            retry_after[key] = dict(until=time.time() + 300, error=str(exc))
                            continue
                        if available is None:
                            continue
                        def evaluate(experiment=experiment, profile=profile, gpu=gpu):
                            with (state / f'gpu_{gpu}.lock').open('a') as lease:
                                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                run(project, experiment, profile, gpu, smoke=True,
                                    stop_requested=stop_event.is_set)
                                if not stop_event.is_set():
                                    return run(project, experiment, profile, gpu,
                                        stop_requested=stop_event.is_set)
                        active[gpu] = (pool.submit(evaluate), key)
                        break
                atomic(state / 'queue.json', dict(epoch=time.time(),
                    state='RUNNING' if active else 'WAITING_DEPENDENCIES_OR_CHECKPOINT',
                    active={str(gpu): value[1] for gpu, value in active.items()},
                    completed=completed, total_jobs=len(jobs), readiness=readiness, errors=retry_after))
                if len(completed) == len(jobs) and not active:
                    atomic(state / 'queue.json', dict(state='COMPLETE', completed=completed, epoch=time.time()))
                    return
                if once:
                    break
                stop_event.wait(project['evaluation']['idle_poll_seconds'])
        # Active episodes finish at the current job boundary; journals remain resumable.
