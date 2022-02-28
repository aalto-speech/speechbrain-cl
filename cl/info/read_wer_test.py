#!/usr/bin/python3
import random
import os, glob
import argparse
from re import M
import matplotlib.pyplot as plt
from cl.info.globals import MPL_COLORS


def read_wer_test(exp_dirs, wer_test_file="wer_test.txt", out_path=None, wer_threshold=100):
    if len(exp_dirs) == 1:
        wer_files = glob.glob(os.path.join(exp_dirs[0], "*", wer_test_file))
        wer_files += glob.glob(os.path.join(exp_dirs[0], wer_test_file))
    elif len(exp_dirs) > 1:
        wer_files = []
        for d in exp_dirs:
            if d.endswith(".txt"):
                wer_files.append(d)
            else:
                wer_files += glob.glob(os.path.join(d, "*", wer_test_file))
                wer_files += glob.glob(os.path.join(d, wer_test_file))
    else:
        raise IndexError(f"exp_dirs has a size of zero.")
    model_to_wer = {}
    for wf in wer_files:
        with open(wf, 'r', encoding='utf-8') as f:
            wer = float(f.readline().split()[1])
        if wer > wer_threshold:
            continue
        model_name = os.path.basename(os.path.dirname(os.path.dirname(wf)))
        seed = os.path.basename(os.path.dirname(wf)).split("-")[0]
        identifier = f"{model_name} ({seed})"
        # model_to_wer[identifier] = wer
        if model_name in model_to_wer:
            model_to_wer[model_name].append(wer)
        else:
            model_to_wer[model_name] = [wer]
        # print(f"Model: {identifier} ==> WER={wer}.")
        print(model_name, "===>", model_to_wer)
    models, wers = [], []
    for model, wer in model_to_wer.items():
        identifier = f"{model} (#runs={len(wer)})"
        wer = sum(wer) / len(wer)
        models.append(identifier)
        wers.append(wer)
    # models = list(model_to_wer.keys())
    # wers = list(model_to_wer.values())
    fig = plt.figure(figsize = (16, 12))
    #  Bar plot
    barplot = plt.barh(models, wers)
    random.shuffle(MPL_COLORS)
    random_colors = (MPL_COLORS * round(len(models)/len(MPL_COLORS) + 0.5))[:len(models)]
    for i in range(len(models)):
        barplot[i].set_color(random_colors[i])
    plt.xlabel("Model Name")
    plt.ylabel("WER")
    plt.title("Word Error Rate (WER) for each model")
    fig.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()

def _get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("exps", nargs="+", help="E.g. `/path/to/exps/` or\
        `/path/to/exps/exp1/1001/ /path/to/exps/exp2/2002 ...`")
    parser.add_argument("--vad", "-v", action="store_true", default=False,
        help="If true then we will read wer files named wer_test_vadded.txt")
    parser.add_argument("--forced-segmented", "-f", action="store_true", default=False,
        help="If true then we will read wer files named wer_test_forced_segmented.txt")
    parser.add_argument("--wer-suffix", "-s", required=False, default=None,
        help="How should the wer txt file be named? Overrides the '-v' and '-f' options.")
    parser.add_argument("--out-path", "-o", required=False, default=None,
        help="Location of the output bar plot.")
    return parser

if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    if args.wer_suffix is None:
        if args.vad:
            args.wer_file = "wer_test_vadded.txt"
        elif args.forced_segmented:
            args.wer_file = "wer_test_forced_segmented.txt"
        else:
            args.wer_file = "wer_test.txt"
    else:
        args.wer_file = f"wer_test{args.wer_suffix}.txt"
    read_wer_test(args.exps, args.wer_file, args.out_path)