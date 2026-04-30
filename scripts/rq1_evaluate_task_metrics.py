"""CLI entrypoint for RQ1 task metric evaluation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepsc_ext.rq1.metrics import build_arg_parser, main  # noqa: E402


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())
