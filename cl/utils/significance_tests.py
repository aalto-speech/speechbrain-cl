import argparse
import os
import subprocess as sp

BASH_FILE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)),
    "significance_tests.sh"
)

def _significance_tests_subprocess(*args, bash_file=BASH_FILE):
    print("Running the following command:", bash_file, *args)
    # print(f"{os.path.exists(bash_file)=}\n\n{os.path.exists(args[0])=}")
    sp.run(["bash", bash_file, *args], capture_output=True, text=True, check=True)

def _get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("ref", help="Path to the reference .trn file.")
    parser.add_argument("hyps", nargs="+", help="A sequence of paths to \
        .trn files.")
    parser.add_argument("--out-dir", "-o", default="", required=False,
        help="Directory where the output files will be written \
            (.unigram, .wilc, .mcn, .sign, .mapsswe).")
    return parser

def _parse_args(args=None):
    if args is None:
        parser = _get_parser()
        args = parser.parse_args()
    _significance_tests_subprocess(args.ref, *args.hyps, "-o {}".format(args.out_dir))

if __name__ == "__main__":
    _parse_args()