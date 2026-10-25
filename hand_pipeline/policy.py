"""Exact training architecture + fixed prompt, using the official Bench2Dex executor."""
import dataclasses
import json
import os
from pathlib import Path
import sys
import time


def register(args):
    from openpi.training import config as configs
    checkpoint = Path(args['checkpoint_path'])
    if not (checkpoint / 'VERIFIED.json').is_file():
        raise ValueError('Checkpoint has no final SHA256 receipt')
    v = json.loads((checkpoint / 'deployment.json').read_text())['config']
    if args.get('robot_key') != v['robot_key']:
        raise ValueError('Training/deployment robot mismatch')
    if args.get('train_config_name') != v['name']:
        raise ValueError('Training/deployment run mismatch')
    base = configs.get_config('pi05_base_dex2bench_full')
    model = dataclasses.replace(base.model, action_dim=v['action_dim'], action_horizon=v['action_horizon'],
        max_token_len=v['max_token_len'], hand_indices=tuple(v['hand_indices']),
        hand_branch=v['hand_branch'], hand_loss_weight=v['hand_loss_weight'],
        hand_branch_width=v.get('hand_branch_width', 256), branch_scope=v.get('branch_scope', 'hand'))
    data = configs.Dex2BenchDataConfig(state_dim=v['action_dim'],
        repo_id='dex2bench:NOT_USED_FOR_INFERENCE|' + v['prompt'],
        assets=configs.AssetsConfig(assets_dir=str(checkpoint / 'assets'), asset_id='dex2bench'))
    configs._CONFIGS_DICT[v['name']] = dataclasses.replace(base, name=v['name'], model=model,
        data=data, assets_base_dir=str(checkpoint / 'assets'), exp_name='seed42', wandb_enabled=False)
    args.update(action_dim=v['action_dim'], state_dim=v['action_dim'],
        train_action_horizon=v['action_horizon'], eval_action_horizon=v['action_horizon'])
    return v


def get_model(args):
    from policy.pi05 import deploy_policy as official
    from robots.active_dof_utils import get_active_dof_info
    v = register(args)
    if get_active_dof_info(v['robot_key']).active_dof != v['action_dim']:
        raise ValueError('Active joint mapping disagrees with training')
    model = official.get_model(args)
    setter, original_reset = model.set_language, model.reset_observation_windows
    model.set_language = lambda instruction: setter(v['prompt'])
    def reset():
        original_reset()
        setter(v['prompt'])
    model.reset_observation_windows = reset
    setter(v['prompt'])
    return model


def encode_obs(observation):
    from policy.pi05.deploy_policy import encode_obs as native
    return native(observation)


class AuditedSession:
    def __init__(self, args):
        from script.policy_sessions import ObservationWindowRemoteSession
        self.inner = ObservationWindowRemoteSession(sys.modules[__name__], args,
            set_instruction_method='set_language', update_method='update_observation_window',
            get_action_method='get_action', reset_method='reset_observation_windows',
            action_limit_attr='eval_action_horizon')
        self.previous = None
        self.seed = None
        self.dimension = int(args['action_dim'])

    def log(self, **values):
        path = os.environ.get('HAND_CONTROL_AUDIT_PATH')
        if path:
            with open(path, 'a') as f:
                f.write(json.dumps(dict(epoch=time.time(), episode_seed=self.seed, **values)) + '\n')

    def state(self, obs):
        import numpy as np
        images, state = encode_obs(obs)
        if len(images) != 4 or any(np.asarray(x).ndim != 3 or np.asarray(x).shape[-1] != 3 for x in images):
            raise ValueError('Expected four RGB views, with no depth/tactile substitution')
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.dimension,) or not np.isfinite(state).all():
            raise ValueError('Invalid active joint input')
        if self.previous is not None:
            self.log(kind='observation', qpos_delta_l2=float(np.linalg.norm(state - self.previous)))
        self.previous = state.copy()
        return state

    def reset(self, instruction=None, *, seed=None):
        self.seed, self.previous = seed, None
        self.inner.reset(instruction, seed=seed)
        self.log(kind='reset', native_rng_reset=True)

    def get_action_chunk(self, observation, instruction=None):
        import numpy as np
        state = self.state(observation)
        actions = self.inner.get_action_chunk(observation, instruction)
        array = np.asarray(actions, dtype=np.float32)
        if array.shape != (20, self.dimension) or not np.isfinite(array).all():
            raise ValueError(f'Invalid action chunk: {array.shape}')
        self.log(kind='action', target_delta_l2=float(np.max(np.linalg.norm(array - state, axis=-1))),
            shape=list(array.shape), finite=True)
        return actions

    def update_after_action(self, observation, instruction=None):
        self.state(observation)
        self.inner.update_after_action(observation, instruction)

    def close(self):
        self.inner.close()


def create_remote_session(args):
    return AuditedSession(args)
