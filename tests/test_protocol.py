import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from hand_pipeline.config import load, experiments, fingerprint, protocol
from hand_pipeline.evaluate import classify, episode_index, native_config, receipt
from hand_pipeline.journal import Journal


class ProtocolTests(unittest.TestCase):
    def test_captured_model_sources_are_exact(self):
        root = Path(__file__).resolve().parents[1] / 'workspaces/architectures/pi05'
        hashes = json.loads((root / 'source_sha256.json').read_text())
        self.assertEqual(len(hashes), 10)
        for name, expected in hashes.items():
            if name.startswith('policy/pi05/'):
                file = root / 'overlay' / name.removeprefix('policy/pi05/')
            else:
                file = root / 'bench2dex_overlay' / name
            self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), expected)

    def test_matrix_and_group_balance(self):
        project = load()
        rows = list(experiments(project))
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({r['run_name'] for r in rows}), 16)
        for r in rows:
            self.assertEqual(r['action_dim'], r['arm_dim'] + r['hand_dim'])
            if r['condition'] == 'C1_group_balanced':
                self.assertAlmostEqual(r['hand_loss_weight'] * r['hand_dim'], r['arm_dim'])
        branch = next(r for r in rows if r['condition'] == 'C0_all_branch')
        self.assertEqual(branch['branch_scope'], 'all')
        self.assertEqual(branch['hand_branch_width'], 253)

    def test_resume_counts_only_valid_unique_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'journal.sqlite'
            j = Journal(path)
            j.register('job', 'fixed')
            j.commit('job', 1, 'SUCCESS', {})
            j.commit('job', 1, 'SUCCESS', {})
            j.commit('job', 2, 'TASK_FAILURE', {})
            j.commit('job', 3, 'RUNTIME_FAILURE', {})
            j.close()
            j = Journal(path)
            self.assertEqual(j.completed('job'), {1, 2})
            self.assertEqual(j.summary('job')['success_rate'], 0.5)
            with self.assertRaises(ValueError):
                j.register('job', 'different_prompt_or_checkpoint')
            j.close()

    def test_native_numbering_and_no_fake_failures(self):
        self.assertEqual(episode_index({'episode_index': 1}), 1)
        self.assertEqual(episode_index({'episode': 50}), 50)
        with self.assertRaises(ValueError):
            episode_index({})
        self.assertEqual(classify({'success': False}, []), 'ACTION_EXECUTION_INVALID')
        actions = [{'kind': 'action', 'target_delta_l2': 1.0},
                   {'kind': 'observation', 'qpos_delta_l2': 0.0}]
        self.assertEqual(classify({'success': False}, actions), 'ACTION_EXECUTION_INVALID')
        actions[-1]['qpos_delta_l2'] = 0.01
        self.assertEqual(classify({'success': False}, actions), 'TASK_FAILURE')
        self.assertEqual(classify({'success': True}, actions), 'SUCCESS')
        self.assertEqual(classify({'success': False, 'error': 'GPU_OOM'}, actions), 'RUNTIME_FAILURE')

    def test_checkpoint_waits_for_end_to_end_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            project = load()
            project['paths']['models'] = Path(directory)
            experiment = next(experiments(project))
            self.assertIsNone(receipt(project, experiment))
            path = Path(directory) / experiment['run_name'] / '10000'
            path.mkdir(parents=True)
            (path / 'SHA256.json').write_text('{}')
            (path / 'VERIFIED.json').write_text(json.dumps(dict(manifest_sha256='incorrect', sha256_all_files=True)))
            with self.assertRaises(ValueError):
                receipt(project, experiment)

    def test_common_evaluation_config(self):
        project = load()
        for experiment in experiments(project):
            config = native_config(project, experiment, 'none', Path('checkpoint'), Path('output'), 19000)
            self.assertEqual(config['seed'], 100000000)
            self.assertEqual(config['train_action_horizon'], 20)
            self.assertEqual(config['eval_action_horizon'], 20)
            self.assertEqual(config['episode_steps'], 800)
            self.assertEqual(config['warmup_steps'], 60)
            self.assertTrue(config['append_output'])
            self.assertTrue(config['use_active_dof'])

    def test_anchor_asset_extraction_uses_samples_not_whole_pool(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/prepare_anchor_assets.py'
        spec = importlib.util.spec_from_file_location('assets', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sample = dict(background={'asset_uri': 'dex2bench_dataset/scenes/ithor/FloorPlan220_physics/scene.usda'},
                      clutter=[{'uri': 'dex2bench_dataset/Objects/025_mug/base.usd'}])
        self.assertEqual(set(module.asset_paths(sample)), {
            'scenes/ithor/FloorPlan220_physics/scene.usda', 'Objects/025_mug/base.usd'})
        with self.assertRaises(ValueError):
            list(module.asset_paths({'uri': 'dex2bench_dataset/../../secret'}))

    def test_dependency_roots_exclude_unused_object_variants(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/complete_asset_dependencies.py'
        spec = importlib.util.spec_from_file_location('closure', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        roots = module.dependency_roots({'paths': ['Objects/ball/usd_sdf_linux/base.usd']},
            {'files': [{'path': 'Objects/ball/usd/base.usd'},
                       {'path': 'Robots_p/arm/usd/base.usd'}]})
        self.assertEqual(roots, {'Objects/ball/usd_sdf_linux/base.usd', 'Robots_p/arm/usd/base.usd'})
        self.assertIn('gltf/pbr.mdl', module.BUILTIN_MDL)

    def test_offline_wheels_are_verified_before_installation(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/setup_environment.py'
        spec = importlib.util.spec_from_file_location('offline_setup', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wheel = root / 'test.whl'
            wheel.write_bytes(b'correct')
            (root / 'wheel_manifest.json').write_text(json.dumps({'files': {'test.whl':
                {'bytes': 7, 'sha256': hashlib.sha256(b'correct').hexdigest()}}}))
            self.assertEqual(len(module.verify_wheels(root)), 64)
            wheel.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                module.verify_wheels(root)


if __name__ == '__main__':
    unittest.main()
