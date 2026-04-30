"""CLI entrypoint for the RQ2 symbol efficiency pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepsc_ext.rq2.common import normalize_negative_csv_args  # noqa: E402
from deepsc_ext.rq2.pipeline import build_arg_parser, main  # noqa: E402


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args(normalize_negative_csv_args(sys.argv[1:], ["--snrs"])))
