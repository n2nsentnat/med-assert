#!/usr/bin/env python3
"""Convenience entrypoint: ``python collect.py "<query>" --count N --output file.json``."""

from article_miner.cli.collect.app import run

if __name__ == "__main__":
    run()
