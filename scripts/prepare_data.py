"""Convenience wrapper; prefer `model-lab data prepare --help`."""

import sys

from model_adaptation_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["data", "prepare", *sys.argv[1:]]))
