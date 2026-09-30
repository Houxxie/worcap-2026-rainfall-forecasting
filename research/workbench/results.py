"""Read completed results as data. Never execute an imported source snapshot."""
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import zipfile
import pandas as pd
from research.workflow.config import require


class Artifacts:
    def __init__(self, source):
        self.source = Path(source).resolve()
        self.zip = self.source.is_file()
        if self.zip:
            require(zipfile.is_zipfile(self.source), 'Review input must be a result folder or reports ZIP.')
            with zipfile.ZipFile(self.source) as archive:
                items = archive.infolist()
                require(sum(i.file_size for i in items) <= 100_000_000, 'Use a small report ZIP or the extracted full result folder.')
                self.names = {i.filename for i in items if not i.is_dir()}
                require(len(self.names) == len([i for i in items if not i.is_dir()]), 'Duplicate ZIP members.')
            for name in self.names:
                self.safe(name)
        else:
            require(self.source.is_dir(), 'Result directory does not exist.')
            self.names = {p.relative_to(self.source).as_posix() for p in self.source.rglob('*') if p.is_file()}

    def safe(self, name):
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts and ':' not in name and '\\' not in name,
                'Unsafe artifact path: ' + name)
        require((self.source/name).resolve().is_relative_to(self.source), 'Artifact leaves its directory.')

    def read(self, name):
        self.safe(name)
        require(name in self.names, 'Missing result artifact: ' + name)
        if self.zip:
            with zipfile.ZipFile(self.source) as archive:
                return archive.read(name)
        return (self.source/name).read_bytes()

    def json(self, name):
        return json.loads(self.read(name))

    def table(self, name):
        return pd.read_csv(io.BytesIO(self.read(name)))

    def inventory(self, expected, allow_missing=()):
        missing = []
        for name, spec in expected.items():
            self.safe(name)
            if name not in self.names:
                require(name in allow_missing, 'Missing result artifact: ' + name)
                missing.append(name)
                continue
            data = self.read(name)
            digest = spec['sha256'] if isinstance(spec, dict) else spec
            require(hashlib.sha256(data).hexdigest() == digest, 'Changed result artifact: ' + name)
            if isinstance(spec, dict):
                require(len(data) == spec['bytes'], 'Artifact size changed: ' + name)
        return missing


def inspect_result(source):
    a = Artifacts(source)
    missing = []
    if 'report_bundle.json' in a.names:
        bundle = a.json('report_bundle.json')
        require(bundle['schema'] == 'rainfall_workbench_report_v1', 'Unknown workbench report.')
        a.inventory(bundle['files'])
        report = a.json('summary.json')
        require(report['kind'] == 'comparison', 'For a current operational status, use task=prepare with the original registry.')
        table, years = a.table('metrics.csv'), a.table('years.csv')
        description, period, primary = report['title'], report['period'], report['primary']
        provenance = report['provenance']
        figures = [f['file'] for f in report['figures']]
        missing = ['original prediction maps (report-only package)']
    elif 'manifest.json' in a.names and 'identity.json' in a.names:
        require(a.json('state.json').get('status') == 'completed', 'This workflow run is incomplete. Use recover to inspect it.')
        manifest = a.json('manifest.json')
        require(manifest.get('schema') == 'rainfall_run_manifest_v1', 'Not a supported completed workflow run.')
        a.inventory(manifest['files'])
        record = a.json('identity.json')
        blend = record.get('experiment') == 'fixed_blend_v1'
        metrics_path = 'fixed_blend/global.csv' if blend else 'diagnostics/global.csv'
        # The existing fixed-blend adapter retains the original table column names.
        if metrics_path not in a.names:
            metrics_path = 'fixed_blend/global_metrics.csv'
        table = a.table(metrics_path)
        years = a.table('fixed_blend/years.csv' if blend else 'comparison/years.csv')
        description = 'Fixed hybrid/U-Net comparison · seven historical blocks'
        period = '2007–2020'
        primary = 'hybrid'
        provenance = dict(source_fingerprint=record['fingerprint'], protocol=record.get('scientific_protocol'),
                          source_signature=record.get('source_signature_sha256'))
        figures = [n for n in ['diagnostics/spatial_errors.png', 'diagnostics/calendar_month.png'] if n in a.names]
    elif {'complete.json', 'global_metrics.csv', 'signature.json', 'summary.json'} <= a.names:
        completion = a.json('complete.json')
        summary = a.json('summary.json')
        require(summary.get('complete_comparison') is True and completion['summary'] == summary, 'Comparison is not complete.')
        missing = a.inventory(completion['files'], allow_missing={'predictions.nc'})
        signature = a.json('signature.json')
        require(signature['plan']['id'] == 'sst_unet_fixed_blend_2021_2022_v1', 'Unsupported saved-map protocol.')
        table, years = a.table('global_metrics.csv'), a.table('years.csv')
        description, period, primary = 'Fixed SST-hybrid/U-Net comparison', '2021–2022', 'blend_reference'
        provenance = dict(input_hashes=signature['input_hashes'], protocol=signature['plan'], source_summary=summary)
        figures = ['monthly_comparison.png']
    elif {'evaluation/complete.json', 'evaluation/global.csv', 'frozen.json'} <= a.names:
        completion = a.json('evaluation/complete.json')
        # Receipts in this adapter are relative to evaluation/.
        expected = {'evaluation/'+n: h for n, h in completion['files'].items()}
        a.inventory(expected)
        frozen = a.json('frozen.json')
        allowed = {n for n in frozen['files'] if Path(n).suffix in {'.nc', '.npz', '.txt'}}
        missing = a.inventory(frozen['files'], allow_missing=allowed)
        require(completion.get('complete_extension') is True or completion.get('summary', {}).get('complete_extension') is True,
                'SST extension is not complete.')
        table, years = a.table('evaluation/global.csv'), a.table('evaluation/years.csv')
        description, period, primary = 'Hybrid with eight SST principal components', '2021–2022', 'hybrid'
        provenance = dict(frozen_receipt=frozen, evaluation=completion)
        figures = ['evaluation/monthly_comparison.png']
    else:
        raise ValueError('Unsupported result layout. Select a completed workflow folder, SST extension output, or SST/U-Net report ZIP. Notebook HTML/code alone is insufficient.')
    rename = {'modelo': 'model', 'ano': 'year', 'vies': 'bias'}
    table, years = table.rename(columns=rename), years.rename(columns=rename)
    require({'model', 'rmse', 'mae', 'bias', 'n'} <= set(table), 'Incomplete metric table.')
    require(not table.model.duplicated().any() and primary in table.model.values, 'Ambiguous reference metrics.')
    require({'model', 'year', 'rmse'} <= set(years), 'Missing yearly comparison.')
    require(not years.duplicated(['model', 'year']).any(), 'Duplicated yearly scores.')
    return dict(artifacts=a, table=table, years=years, title=description, period=period, primary=primary,
                provenance=provenance, missing=missing, figures=figures,
                limitations='Previously consulted development years. Historical publication vintages are unverified. '
                + ('Report-only review: omitted maps prevent raw-observation rescoring.' if missing else
                   'This review verifies saved artifacts; it does not independently rescore observations.'))
