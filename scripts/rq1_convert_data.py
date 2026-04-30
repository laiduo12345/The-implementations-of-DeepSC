"""CLI entrypoint for RQ1 DeepSC format conversion."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepsc_ext.rq1.convert import build_arg_parser, main  # noqa: E402


if __name__ == "__main__":
    parser = build_arg_parser()
    metadata = main(parser.parse_args())
    print("Converted data: train={train_size} valid={valid_size} test={test_size} vocab={vocab_size}".format(**metadata))
