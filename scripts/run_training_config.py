#!/usr/bin/env python3
"""Run one V-JEPA config in the current process.

This is intentionally simpler than app.main_distributed: Slurm or another
launcher starts one process per task, and the training code reads SLURM_* env
vars to initialize distributed training.
"""

from __future__ import annotations

import argparse
import pprint

import yaml

from app.scaffold import main as app_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True, help="Training YAML config.")
    args = parser.parse_args()

    with open(args.fname) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    pprint.PrettyPrinter(indent=4).pprint(params)
    app_main(params["app"], args=params)


if __name__ == "__main__":
    main()
