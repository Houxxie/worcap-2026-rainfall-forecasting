"""One command for configuration checks, execution, results and recovery."""
import argparse
import json
import sys
from .config import load
from .runner import check, run, verify, status, recover, export


def main(argv=None):
    parser=argparse.ArgumentParser(description='Rainfall workbench: evaluate saved forecasts, train the existing protocol, or check monthly readiness.')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ['check','run']:
        commands.add_parser(name).add_argument('--config',required=True)
    for name in ['verify','recover','export']:
        commands.add_parser(name).add_argument('--run',required=True)
    commands.add_parser('status').add_argument('--root',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='check': value=check(load(args.config))
        elif args.command=='run': value=dict(run=str(run(args.config)))
        elif args.command=='verify': value=verify(args.run)
        elif args.command=='recover': value=recover(args.run)
        elif args.command=='export': value=dict(report_zip=str(export(args.run)))
        else: value=status(args.root)
        print(json.dumps(value,indent=2,ensure_ascii=False))
        return 2 if args.command=='check' and not value['ready'] else 0
    except (ValueError,OSError,ImportError,KeyError,TypeError) as error:
        print('Workbench stopped:',error,file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
