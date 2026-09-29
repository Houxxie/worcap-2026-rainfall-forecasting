"""Check configuration boundaries, artifact integrity and failed-run behavior."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.workflow import config, runner


def example(base):
    return dict(schema='rainfall_workflow_v1', name='example', mode='saved', output_root=str(base / 'outputs'),
                inputs=dict(observations=str(base / 'official/treino_tp.nc'), predictions=str(base / 'maps')))


class WorkflowChecks(unittest.TestCase):
    def test_relative_paths_follow_configuration_not_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = example(root)
            raw['inputs']['observations'] = 'official/treino_tp.nc'
            value = config.normalize(raw, root)
            self.assertEqual(Path(value['inputs']['observations']), root / 'official/treino_tp.nc')

    def test_scientific_or_unknown_settings_are_rejected(self):
        raw = example(Path('/example'))
        raw['learning_rate'] = .1
        with self.assertRaisesRegex(config.WorkflowError, 'Unknown configuration'):
            config.normalize(raw, Path.cwd())

    def test_no_output_inside_input_or_gpu_setting_in_saved_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = example(Path(directory))
            raw['output_root'] = str(Path(raw['inputs']['predictions']) / 'overwrite')
            with self.assertRaisesRegex(config.WorkflowError, 'inside inputs'):
                config.normalize(raw, Path(directory))
            raw = example(Path(directory)); raw['device'] = 'cuda'
            with self.assertRaisesRegex(config.WorkflowError, 'does not use a GPU'):
                config.normalize(raw, Path(directory))

    def test_duplicate_json_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text('{"mode":"saved","mode":"train"}')
            with self.assertRaisesRegex(config.WorkflowError, 'Duplicate JSON key'):
                config.load(path)

    def test_discovery_rejects_missing_and_ambiguous_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'treino_tp.nc').touch()
            with self.assertRaisesRegex(config.WorkflowError, '0 complete'):
                config.discover_saved(root)
            for run in ['one', 'two']:
                base = root / run; base.mkdir(); (base / 'signature.json').touch()
                for block in config.BLOCKS:
                    (base / block).mkdir(); (base / block / 'predictions.nc').touch()
            with self.assertRaisesRegex(config.WorkflowError, '2 complete'):
                config.discover_saved(root)

    def completed_fixture(self, root):
        for name, value in [('state.json', dict(status='completed')), ('identity.json', dict(run_id='test')),
                            ('config.json', {}), ('diagnostics/summary.json', {})]:
            path = root / name; path.parent.mkdir(exist_ok=True, parents=True); runner.write_json(path, value)
        (root / 'report.html').write_text('original')
        runner.write_manifest(root)

    def test_changed_and_extra_artifacts_invalidate_completed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.completed_fixture(root)
            self.assertEqual(runner.verify_run(root)['status'], 'verified')
            (root / 'report.html').write_text('changed')
            with self.assertRaisesRegex(config.WorkflowError, 'hash differs'):
                runner.verify_run(root)
            (root / 'report.html').write_text('original')
            (root / 'extra.txt').write_text('not frozen')
            with self.assertRaisesRegex(config.WorkflowError, 'added or removed'):
                runner.verify_run(root)

    def test_manifest_cannot_read_outside_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.completed_fixture(root)
            manifest = config.read_json(root / 'manifest.json')
            manifest['files']['../outside'] = dict(sha256='0' * 64, bytes=0)
            runner.write_json(root / 'manifest.json', manifest)
            with self.assertRaisesRegex(config.WorkflowError, 'leaves the run'):
                runner.verify_run(root)

    def test_failure_preserves_inputs_and_never_marks_run_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'settings.json'
            runner.write_json(path, example(root))
            original = path.read_bytes()
            audit = dict(mode='saved', code={}, inputs={}, fingerprint='a' * 64, source_signature_sha256='b' * 64)
            with patch.object(runner, 'preflight', return_value=audit), patch.object(runner, 'copy_evidence', side_effect=ValueError('injected missing input')):
                with self.assertRaisesRegex(config.WorkflowError, 'Partial outputs preserved'):
                    runner.run(path)
            states = list((root / 'outputs').glob('*/state.json'))
            self.assertEqual(len(states), 1)
            state = config.read_json(states[0])
            self.assertEqual(state['status'], 'failed')
            self.assertFalse(state['model_promoted'])
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(config.WorkflowError, 'incomplete or failed'):
                runner.verify_run(states[0].parent)


if __name__ == '__main__':
    unittest.main()
