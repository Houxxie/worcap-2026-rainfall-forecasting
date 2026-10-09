"""Boundary and recovery checks; no forecast model is fitted."""
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import tempfile
import unittest
import zipfile
import pandas as pd
from . import config, inputs, results, runner


def report_fixture(root):
    root.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame([dict(model=n,n=10,rmse=v,mae=v*.6,bias=.1) for n,v in
                        [('hybrid',2.),('blend_reference',1.9),('hybrid_sst',1.8),('blend_sst',1.79)]])
    table.to_csv(root/'global_metrics.csv',index=False)
    table.assign(year=2021).to_csv(root/'years.csv',index=False)
    summary=dict(complete_comparison=True)
    (root/'summary.json').write_text(json.dumps(summary))
    (root/'signature.json').write_text(json.dumps(dict(plan=dict(id='sst_unet_fixed_blend_2021_2022_v1'),input_hashes={})))
    # A valid tiny PNG, also exercises self-contained image export.
    import base64
    (root/'monthly_comparison.png').write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6bwAAAABJRU5ErkJggg=='))
    hashes={p.name:runner.sha256(p) for p in root.iterdir()}
    hashes['predictions.nc']='0'*64
    (root/'complete.json').write_text(json.dumps(dict(summary=summary,files=hashes)))
    return root


def settings(root, **overrides):
    value=dict(schema='rainfall_workbench_v1',name='example',task='review',
               inputs=dict(results=str(root/'input')),output_root=str(root/'output'))
    value.update(overrides)
    path=root/'settings.json'; path.write_text(json.dumps(value))
    return path


class WorkbenchTests(unittest.TestCase):
    def test_unknown_scientific_knobs_and_accidental_training_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for extra in [dict(epochs=100),dict(device='cuda'),dict(task='train',experiment='sst_unet_blend_v1')]:
                p=settings(root)
                raw=json.loads(p.read_text());raw.update(extra)
                with self.assertRaises(ValueError): config.normalize(raw,root)
            normalized=config.normalize(dict(schema='rainfall_workbench_v1',task='review',inputs=dict(results='input'),output_root='out'),root)
            self.assertEqual(normalized['inputs']['results'],str(root/'input'))

    def test_discovery_reports_all_missing_inputs_without_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            c=config.load(settings(root,task='evaluate',experiment='sst_unet_blend_v1',inputs={},search_roots=[str(root)]))
            with patch.object(runner,'execute_backend',side_effect=AssertionError('must not run')):
                audit=runner.check(c)
                self.assertFalse(audit['ready'])
                self.assertEqual(len(audit['errors']),3)
                self.assertFalse((root/'output').exists())

    def test_wrong_hash_and_duplicate_matching_copies_need_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for sub in ['a','b']:
                (root/sub).mkdir();(root/sub/'predictions.nc').write_bytes(b'correct')
            plan=dict(inputs={'sst_predictions':dict(filename='predictions.nc',sha256=hashlib.sha256(b'correct').hexdigest())})
            c=config.load(settings(root,task='evaluate',experiment='sst_unet_blend_v1',inputs={},search_roots=[str(root)]))
            with patch('research.sst_unet_blend.experiment.protocol',return_value=plan):
                self.assertFalse(inputs.discover(c)['ready'])
                c['inputs']['sst_predictions']=str(root/'a/predictions.nc')
                self.assertTrue(inputs.discover(c)['ready'])
                (root/'a/predictions.nc').write_bytes(b'wrong')
                self.assertFalse(inputs.discover(c)['ready'])

    def test_review_cache_export_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report_fixture(root/'input');p=settings(root)
            with patch.object(runner,'execute_backend',wraps=runner.execute_backend) as execute:
                run=runner.run(p)
                self.assertEqual(runner.run(p),run)
                self.assertEqual(execute.call_count,1)
            self.assertEqual(runner.verify(run)['status'],'verified')
            exported=runner.export(run)
            view=results.inspect_result(exported)
            self.assertEqual(len(view['table']),4)
            self.assertTrue(view['missing'])
            (run/'report.html').write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'): runner.run(p)

    def test_report_failure_preserves_completed_payload_and_recovers_without_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report_fixture(root/'input');p=settings(root)
            with patch('research.workbench.report.render',side_effect=RuntimeError('report failed')):
                with self.assertRaisesRegex(ValueError,'Outputs preserved'): runner.run(p)
            run=next((root/'output').glob('*/state.json')).parent
            self.assertEqual(json.loads((run/'state.json').read_text())['stage'],'report')
            recovery=runner.recover(run)
            self.assertEqual(recovery['settings']['task'],'review')
            continued=root/'continue.json';continued.write_text(json.dumps(recovery['settings']))
            completed=runner.run(continued)
            self.assertNotEqual(completed,run)
            self.assertEqual(runner.verify(completed)['status'],'verified')
            self.assertEqual(json.loads((run/'state.json').read_text())['status'],'failed')

    def test_concurrent_job_lock_blocks_duplicate_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report_fixture(root/'input');p=settings(root)
            audit=runner.check(config.load(p));(root/'output').mkdir()
            lock=root/'output'/('.'+audit['fingerprint']+'.lock');lock.write_text('{}')
            with patch.object(runner,'execute_backend',side_effect=AssertionError('duplicate work')):
                with self.assertRaisesRegex(ValueError,'matching job'): runner.run(p)
            self.assertTrue(lock.exists())

    def test_zip_traversal_and_changed_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            bad=root/'bad.zip'
            with zipfile.ZipFile(bad,'w') as z: z.writestr('../escape.json','{}')
            with self.assertRaisesRegex(ValueError,'Unsafe'): results.Artifacts(bad)
            source=report_fixture(root/'input')
            (source/'years.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Changed result'):results.inspect_result(source)

    def test_prepare_keeps_existing_registry_unchanged_and_never_issues(self):
        from research.prospective.registry import Registro
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registry=root/'registry'
            r=Registro(registry)
            r.iniciar(dict(primeiro_mes='2099-01',ultimo_mes='2099-12',fontes={'cfsv2':dict(lag=1,obrigatoria=True)}))
            before=runner.files_at(registry)
            p=settings(root,task='prepare',target='2099-01',inputs=dict(registry=str(registry)))
            run=runner.run(p)
            summary=json.loads((run/'summary.json').read_text())
            self.assertEqual(summary['status'],'blocked')
            self.assertFalse(summary['registry']['status']['previsao_emitida'])
            self.assertEqual(before,runner.files_at(registry))
            with self.assertRaisesRegex(ValueError,'Missing registry'): inputs.registry_snapshot(root/'absent','2099-01')
            self.assertFalse((root/'absent').exists())

    def test_fresh_receipt_with_old_month_still_blocks_source_readiness(self):
        from research.prospective.registry import Registro
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r = Registro(root/'registry')
            r.iniciar(dict(primeiro_mes='2099-01', ultimo_mes='2099-12',
                           fontes={'nino34': dict(lag=3, obrigatoria=True)}))
            receipt = r.recibo('nino34', b'old but valid',
                               dict(validado=True, meses=['2098-09']), dict(metodo='test'))
            r.recibo('nino34', b'unvalidated newer data',
                     dict(validado=False, meses=['2098-10']), dict(metodo='test'))
            snapshot = inputs.registry_snapshot(root/'registry', '2099-01')
            source = snapshot['sources'][0]
            self.assertEqual(source['required_month'], '2098-10')
            self.assertEqual(source['latest_valid_month'], '2098-09')
            self.assertIsNotNone(source['last_acquisition'])
            self.assertIsNone(source['selected_event'])
            self.assertEqual(source['status'], 'missing or unvalidated')
            self.assertFalse(snapshot['status']['pronto'])

    def test_training_routes_only_to_existing_protocol_with_resolved_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            c=config.load(settings(root,task='train',experiment='hybrid_unet_v1',device='cuda',
                                   inputs={k:str(root/k) for k in ['official','seas5','cfsv2']}))
            folder=root/'run';folder.mkdir()
            audit=dict(inputs=c['inputs'])
            with patch('research.workflow.runner.run',return_value=root/'completed') as backend:
                result=runner.execute_backend(folder,c,audit)
            generated=json.loads((folder/'engine_config.json').read_text())
            self.assertEqual(generated['mode'],'train')
            self.assertEqual(generated['experiment'],'hybrid_unet_v1')
            self.assertEqual(generated['device'],'cuda')
            self.assertEqual(generated['inputs'],c['inputs'])
            self.assertEqual(result,str(root/'completed'))
            backend.assert_called_once_with(folder/'engine_config.json')

    def test_complete_training_survives_reporting_failure_and_is_proposed_for_saved_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            trained=root/'engine/run/training';trained.mkdir(parents=True)
            (trained/'signature.json').write_text('{}')
            (root/'state.json').write_text(json.dumps(dict(status='failed',stage='execution')))
            # Paths may have been discovered, rather than written manually in settings.
            (root/'identity.json').write_text(json.dumps(dict(inputs=dict(official=str(root/'official')))))
            with patch('research.workflow.runner.inspect_saved',return_value=({},'signature')):
                recovered=runner.recover(root)
            self.assertEqual(recovered['settings']['task'],'evaluate')
            self.assertEqual(recovered['settings']['inputs']['observations'],str(root/'official/treino_tp.nc'))
            self.assertEqual(recovered['settings']['inputs']['predictions'],str(trained))


if __name__=='__main__':
    unittest.main()
