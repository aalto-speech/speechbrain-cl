import argparse
import os, glob
import ast
import numpy as np


def calc_corr(*paths):
    try:
        from scipy.stats import spearmanr
    except ImportError:
        raise ImportError("You need to install scipy to use this script. Cannot continue...")
    for p in paths:
        assert os.path.isfile(p), f"Could not locate path {p}"


def _get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("log1", help="Path to the first curriculum log path.")
    parser.add_argument("log2", help="Path to the second curriculum log path with which we will try \
        to find the correlation of the orderings.")
    return parser

def _parse_args(args=None):
    if args is None:
        parser = _get_parser()
        args = parser.parse_args()
    calc_corr(args.log1, args.log2)


if __name__ == "__main__":
    _parse_args()