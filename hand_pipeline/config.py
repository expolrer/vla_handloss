import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(root=ROOT):
    root = Path(root).resolve()
    project = yaml.safe_load((root / 'config/project.yaml').read_text())
    local = root / 'workspaces/local.yaml'
    if local.is_file():
        overrides = yaml.safe_load(local.read_text()) or {}
        for section in ('paths', 'evaluation'):
            project[section].update(overrides.get(section, {}))
    project['root'] = root
    project['paths'] = {key: (root / value).resolve() for key, value in project['paths'].items()}
    project['tasks'] = yaml.safe_load((root / 'config/tasks.yaml').read_text())
    project['conditions'] = yaml.safe_load((root / 'config/experiments.yaml').read_text())
    return project


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def experiments(project):
    for condition, original in project['conditions'].items():
        for task, task_values in project['tasks'].items():
            values = dict(original)
            if values['hand_loss_weight'] == 'group_balanced':
                values['hand_loss_weight'] = task_values['arm_dim'] / task_values['hand_dim']
            name = task + '_' + condition + values.pop('name_suffix', '')
            yield dict(task=task, condition=condition, **task_values, **values, run_name=name)


def protocol(project, experiment, profile, manifest_sha):
    native_files = ['run_policy.py', 'script/policy_sessions.py', 'utils/seed_policy.py',
        'benchmark/metrics.py', 'benchmark/results.py', 'robots/active_dof_maps.yml',
        'policy/pi05/pi_model.py', 'policy/pi05/deploy_policy.py',
        'policy/pi05/src/openpi/models/pi0.py', 'policy/pi05/src/openpi/models/pi0_config.py',
        'scenes/' + experiment['task'] + '.yaml']
    native = project['paths']['bench2dex']
    return dict(experiment=experiment, profile=profile, evaluation=project['evaluation'],
        training=project['training'], checkpoint_manifest=manifest_sha,
        upstream=project['upstream'], prompt_policy='fixed_training_prompt',
        implementation_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((project['root'] / 'hand_pipeline').glob('*.py'))},
        native_execution_sha256={name: hashlib.sha256((native / name).read_bytes()).hexdigest()
            for name in native_files if (native / name).is_file()},
        resume='episode_boundary_replay_with_native_per_episode_rng_reset')
