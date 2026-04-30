"""CLI entrypoint for RQ2 DeepSC training."""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepsc_ext.rq2.train import build_arg_parser, main  # noqa: E402


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args(), argv=sys.argv)

