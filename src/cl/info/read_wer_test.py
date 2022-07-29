#!/usr/bin/python3
from collections import defaultdict
import random
import os, glob
import argparse
import json
import warnings
try:
    import matplotlib.pyplot as plt
except ImportError:
    # warnings.warn("Could not import matplotlib. If you are planning to use the visualization functions then you need to install it.")
    pass

from cl.info.globals import MPL_COLORS, map_name


def read_wer_test(
    exp_dirs, 
    wer_test_file="wer_test.txt", 
    out_path=None, 
    wer_threshold=100,
    name_mappings_file=None,
):
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
    if os.path.isfile(name_mappings_file):
        print("YES")
        with open(name_mappings_file, 'r') as f:
            name_mappings = json.loads(f.read())
    else:
        name_mappings = None
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
        # print(model_name, "===>", model_to_wer)
    models, wers = [], []
    for model, wer in model_to_wer.items():
        model = map_name(model, name_mappings)
        identifier = f"{model} (#runs={len(wer)})"
        wer = sum(wer) / len(wer)
        models.append(identifier)
        wers.append(wer)
        print(f"Model: {identifier} ==> WER={wer}.")
    # models = list(model_to_wer.keys())
    # wers = list(model_to_wer.values())
    # fig = plt.figure(figsize = (16, 12))
    # #  Bar plot
    # barplot = plt.barh(models, wers)
    # random.shuffle(MPL_COLORS)
    # random_colors = (MPL_COLORS * round(len(models)/len(MPL_COLORS) + 0.5))[:len(models)]
    # for i in range(len(models)):
    #     barplot[i].set_color(random_colors[i])
    # plt.xlabel("Model Name")
    # plt.ylabel("WER")
    # plt.title("Word Error Rate (WER) for each model")
    # Figure Size

    models, wers = list(map(list, zip(*sorted(zip(models, wers), key=lambda x: x[1]))))

    fig, ax = plt.subplots(figsize =(16, 9))
    
    # Horizontal Bar Plot
    barplot = ax.barh(models, wers)
    
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    random.shuffle(MPL_COLORS)
    random_colors = (MPL_COLORS * round(len(models)/len(MPL_COLORS) + 0.5))[:len(models)]
    for i in range(len(models)):
        barplot[i].set_color(random_colors[i])
    
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)
    
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = .8,
            alpha = 0.4)
    
    # Show top values
    ax.invert_yaxis()
    
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                str(round((i.get_width()), 2)),
                fontsize = 10, fontweight ='bold',
                color ='grey')
    
    # Add Plot Title
    ax.set_title("Word Error Rate (WER) for each model",
                loc ='left', )

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
    parser.add_argument("--cer", action="store_true", default=False,
        help="Use CER instead of WER.")
    parser.add_argument("--model-name-mappings", "-m", default=None, 
        help="Path to a .json file containing a dictionary with the keys\
              `curriculum_mappings`, `transfer_mappings` and `subset_mappings`.\
              This remains to be documented.")
    return parser

if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    prefix = "cer" if args.cer else "wer"
    if args.wer_suffix is None:
        if args.vad:
            args.wer_file = f"{prefix}_test_vadded.txt"
        elif args.forced_segmented:
            args.wer_file = f"{prefix}_test_forced_segmented.txt"
        else:
            args.wer_file = f"{prefix}_test.txt"
    else:
        args.wer_file = f"{prefix}_test{args.wer_suffix}.txt"
    read_wer_test(args.exps, args.wer_file, args.out_path)