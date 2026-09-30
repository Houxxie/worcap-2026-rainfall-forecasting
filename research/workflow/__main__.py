"""Usage: python -m research.workflow check|run|verify|list ..."""
import argparse
import json
import sys
from .config import load, WorkflowError
from .runner import preflight, run, verify_run, list_runs


def main(argv=None):
    parser = argparse.ArgumentParser(description='Configured rainfall development comparisons. No automatic model promotion or forecast issuance.')
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ['check', 'run']:
        commands.add_parser(command).add_argument('--config', required=True, help='JSON file; relative paths resolve from its directory')
    commands.add_parser('verify').add_argument('--run', required=True, help='Completed run directory')
    commands.add_parser('list').add_argument('--root', required=True, help='Output root containing workflow runs')
    args = parser.parse_args(argv)
    try:
        if args.command == 'check':
            audit = preflight(load(args.config))
            value = dict(status='ready', mode=audit['mode'], fingerprint=audit['fingerprint'],
                         input_artifacts=len(audit['inputs']), independent_holdout=False,
                         message='Inputs and environment checked. No model was fitted and no output run was created.')
        elif args.command == 'run':
            value = dict(status='completed', directory=str(run(args.config)))
        elif args.command == 'verify':
            value = verify_run(args.run)
        else:
            value = list_runs(args.root)
        print(json.dumps(value, indent=2))
        return 0
    except (WorkflowError, ValueError, OSError, ImportError, KeyError, TypeError) as error:
        print('Workflow stopped:', error, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
