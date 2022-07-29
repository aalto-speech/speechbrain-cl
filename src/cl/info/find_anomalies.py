import argparse
import os
import re
from typing import Optional
import random
import warnings


FIRST_EXAMPLE_LINE = 12
SEPARATOR = "================================================================================"
DEFAULT_METRIC_THRESHOLD = 50.0  # i.e. we will keep all entries with a WER > 50.0 in order to analyze them

def read_single(path, threshold=DEFAULT_METRIC_THRESHOLD, print_stats=True):
    assert os.path.exists(path), f"Could not locate {path=}"
    basename = os.path.basename(path)
    is_wer = False if "cer_test" in path else True
    separate_func = lambda x: separate_test_entry(x, is_wer)
    with open(path, 'r', encoding='utf-8') as fr:
        # Before the FIRST_EXAMPLE_LINE line, we have general information 
        # and instructions on how to read the file so we don't care.
        lines = fr.readlines()[FIRST_EXAMPLE_LINE:]
        lines = "".join(lines).split(SEPARATOR)
        lines = [e[:-2] for e in map(separate_func, lines) if e[1] >= threshold]
        if print_stats:
            print("="*60)
            print(f"Number of utts with more than {threshold} {'WER' if is_wer else 'CER'}: {len(lines)}.")
            print(f"Insertions: {sum(e[2] for e in lines)}  |  Deletions: {sum(e[3] for e in lines)}  |  Substitutions: {sum(e[4] for e in lines)}")
            try:
                random_insertions = random.sample([e[-1] for e in lines if e[2] > 4], 2)
                print("Some random utterances with more than 4 insertions:\n  "+'\n  '.join(random_insertions))
            except ValueError as e:
                warnings.warn(str(e))
    return lines
        
def compare(cer_path: str, wer_path: str, threshold: float = DEFAULT_METRIC_THRESHOLD, out_path: Optional[str] = None):
    if cer_path.endswith("wer_test.txt"):
        tmp = wer_path
        wer_path = cer_path
        cer_path = tmp
    assert os.path.isfile(cer_path) and os.path.isfile(wer_path)
    cl = read_single(cer_path, threshold)
    wl = read_single(wer_path, threshold)
    cer_bad_utts = map(lambda x: x[0], cl)
    wer_bad_utts = map(lambda x: x[0], wl)
    print("="*60)
    bad_overlaps = set(cer_bad_utts).intersection(set(wer_bad_utts))
    print(f"Found {len(bad_overlaps)} common bad utterances in the two sets.")
    if out_path is not None:
        with open(out_path, 'w') as fw:
            for e in cl:
                if e in bad_overlaps:
                    raise NotImplementedError()

    
def separate_test_entry(entry: str, is_wer: bool):
    # E.g.
    # sample-001053, %WER 0.00 [ 0 / 18, 0 ins, 0 del, 0 sub ]
    # i ; ' ; l ; l ; _ ; b ; e ; _ ; r ; i ; g ; h ; t ; _ ; h ; e ; r ; e
    # = ; = ; = ; = ; = ; = ; = ; = ; = ; = ; = ; = ; = ; = ; = ; = ; = ; =
    # i ; ' ; l ; l ; _ ; b ; e ; _ ; r ; i ; g ; h ; t ; _ ; h ; e ; r ; e
    # Returns a tuple of all relevant elemnts
    entry = [e for e in entry.split("\n") if len(e.strip()) != 0]
    assert len(entry) == 4, f"Length of entry should be 4. {entry=}"
    stats, truth, _, preds = entry
    if is_wer:
        reconstruct = lambda x: re.sub("\s+", " ", x.replace(";", "")).replace("<eps>", "").strip()
    else:
        reconstruct = lambda x: x.replace(";", "").replace(" ", "").replace("<eps>", "").replace("_", " ").strip()
    utt_id = stats.split(",")[0]
    truth_reconstructed = reconstruct(truth)
    preds_reconstructed = reconstruct(preds)
    score = float(stats.split("[")[-2].split(" ")[-2].strip())
    ins = int(stats.split("ins")[-2].split(",")[-1].strip())
    dels = int(stats.split("del")[-2].split(",")[-1].strip())
    subs = int(stats.split("sub")[-2].split(",")[-1].strip())
    return utt_id, score, ins, dels, subs, truth_reconstructed, preds_reconstructed, truth, preds


def _get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", 
        help="Path or paths to wer_test*.txt files."
    )
    parser.add_argument("--error-threshold", "-t", type=float, default=DEFAULT_METRIC_THRESHOLD, 
        help="WER/CER/Whatever threshold below which the utterances are considered okay.")
    parser.add_argument("--print-stats", "-p", action="store_true", default=False,
        help="Whether to print the stats or not.")
    parser.add_argument("--compare", "-c", action="store_true", default=False, 
        help="If true then you must have provided pairs of paths to wer_test*.txt \
            and cer_test*.txt file which will be compared.")
    parser.add_argument("--out-log-path", "-ol", required=False, default=None,
        help="Not implemented.")
    return parser

def _parse_args(args=None):
    if args is None:
        parser = _get_parser()
        args = parser.parse_args()
    else:
        args.paths += args.exps
    if args.compare:
        n_paths = len(args.paths)
        if n_paths % 2 != 0:
            raise argparse.ArgumentTypeError("Since you used --compare, you must provide pairs of paths.")
        wer_paths = [p for p in args.paths if "wer" in p]
        cer_paths = [p for p in args.paths if "cer" in p]
        print(wer_paths, cer_paths)
        if len(wer_paths) != len(cer_paths):
            raise argparse.ArgumentTypeError("You must provide an equal number of wer_test*.txt and cer_test*.txt files.")
        for wer_path, cer_path in zip(wer_paths, cer_paths):
            compare(cer_path, wer_path, args.error_threshold, args.out_log_path)
    else:
        for path in args.paths:
            read_single(path, args.error_threshold, args.print_stats)

# dispatching:

if __name__ == '__main__':
    _parse_args()