import argparse
import json
from pathlib import Path
from .config import load, experiments
from .setup import adopt, check


def main():
    parser = argparse.ArgumentParser(description='Dexterous-hand VLA experiments and resumable Bench2Dex evaluation')
    parser.add_argument('--root', type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    setup = sub.add_parser('setup')
    setup.add_argument('--runtime', type=Path, required=True)
    sub.add_parser('doctor')
    sub.add_parser('status')
    sub.add_parser('matrix')
    run = sub.add_parser('evaluate')
    run.add_argument('--run', required=True)
    run.add_argument('--profile', default='none', choices=['none', 'cov_only', 'inv_only', 'inv_cov'])
    run.add_argument('--gpu', type=int, required=True)
    run.add_argument('--smoke', action='store_true')
    queue = sub.add_parser('queue')
    queue.add_argument('--gpus', type=int, nargs='+', required=True)
    queue.add_argument('--execute', action='store_true')
    queue.add_argument('--once', action='store_true')
    args = parser.parse_args()
    project = load(args.root) if args.root else load()
    if args.command == 'setup':
        adopt(project, args.runtime)
        print(json.dumps(check(project), indent=2))
    elif args.command == 'doctor':
        print(json.dumps(check(project), indent=2))
    elif args.command == 'matrix':
        print(json.dumps(list(experiments(project)), indent=2))
    elif args.command == 'status':
        state = project['paths']['state'] / 'queue.json'
        print(state.read_text() if state.exists() else json.dumps(dict(state='NOT_STARTED')))
    elif args.command == 'evaluate':
        from .evaluate import run
        experiment = next((x for x in experiments(project) if x['run_name'] == args.run), None)
        if experiment is None:
            raise SystemExit('Unknown experiment; run the matrix command')
        print(json.dumps(run(project, experiment, args.profile, args.gpu, smoke=args.smoke)))
    elif args.command == 'queue':
        from .queue import execute
        execute(project, args.gpus, args.execute, args.once)


if __name__ == '__main__':
    main()
