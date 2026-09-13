"""Unified KIQFL command-line entry point."""
from __future__ import annotations

import sys


USAGE = """usage: kiqfl [-h] {{{commands}}} ...

KIQFL — hybrid quantum-classical federated learning toolkit

commands:
  experiments   Run experiment suites (writes CSV under result/)
  analyze       Summarize result/experiments_log.csv
  pipeline      Full multi-suite experiment pipeline
  resume        Resume unfinished experiment suites
  web           Start the Flask demo UI (http://127.0.0.1:5000)

examples:
  kiqfl experiments --quick --suite main
  kiqfl analyze
  kiqfl pipeline
  kiqfl web

Also: python -m core <command> ...
""".format(commands='experiments,analyze,pipeline,resume,web')


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        print(USAGE)
        return 0

    command, *forward = argv
    known = {'experiments', 'analyze', 'pipeline', 'resume', 'web'}
    if command not in known:
        print(f'unknown command: {command}\n', file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    if command == 'experiments':
        from core.experiments.runner import main as run
        sys.argv = ['kiqfl-experiments', *forward]
        run()
        return 0
    if command == 'analyze':
        from core.experiments.analyze import main as run
        sys.argv = ['kiqfl-analyze', *forward]
        run()
        return 0
    if command == 'pipeline':
        from core.experiments.pipeline import main as run
        sys.argv = ['kiqfl-pipeline', *forward]
        run()
        return 0
    if command == 'resume':
        from core.experiments.resume import main as run
        sys.argv = ['kiqfl-resume', *forward]
        return int(run() or 0)
    if command == 'web':
        import os
        from webapp.app import app
        app.run(
            host='127.0.0.1',
            port=int(os.environ.get('PORT', '5000')),
            debug=os.environ.get('FLASK_DEBUG', '0') == '1',
            threaded=True,
        )
        return 0
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
