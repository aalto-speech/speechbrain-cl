import argparse
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from tqdm import tqdm
import json
from cl.info.read_wer_test import map_name
from cl.info.globals import MPL_COLORS
from cl.info.find_anomalies import SEPARATOR, FIRST_EXAMPLE_LINE
# sns.set()


def testset_corr(
    exp_dirs, 
    wer_test_file="wer_test.txt", 
    out_path=None, 
    name_mappings_file=None,
):
    if len(exp_dirs) == 1:
        wer_files = [os.path.join(exp_dirs[0], wer_test_file)]
    elif len(exp_dirs) > 1:
        wer_files = []
        for d in exp_dirs:
            if d.endswith(wer_test_file):
                wer_files.append(d)
            else:
                wer_files.append(os.path.join(d, wer_test_file))
    else:
        raise IndexError(f"exp_dirs has a size of zero.")
    if name_mappings_file is not None and os.path.isfile(name_mappings_file):
        with open(name_mappings_file, 'r') as f:
            name_mappings = json.loads(f.read())
    else:
        name_mappings = None
    model_to_orders = {}
    for i, wf in enumerate(wer_files):
        if not os.path.isfile(wf):
            print(f"Ignoring file {wf} since it doesn't exist.")
            continue
        with open(wf, 'r', encoding='utf-8') as fr:
            # Before the FIRST_EXAMPLE_LINE line, we have general information 
            # and instructions on how to read the file so we don't care.
            lines = fr.readlines()[FIRST_EXAMPLE_LINE:]
            lines = "".join(lines).split(SEPARATOR)
            # `lines` is now a dict of pairs (utterance_id: wer_score)
            lines = {l.split(",")[0].strip(): float(l.split("[")[0].split("WER")[1].strip()) for l in lines}
        
        model_name = os.path.basename(os.path.dirname(os.path.dirname(wf)))
        seed = os.path.basename(os.path.dirname(wf)).split("-")[0]
        identifier = f"{model_name} ({seed}) ({i})"  # add 'i' so that the key is unique
        model_to_orders[identifier] = lines
    models, corrs = [], []
    # pbar = tqdm(zip(list(model_to_orders.keys())[:-1], list(model_to_orders.keys())[1:]), total=len(model_to_orders)-1)
    pbar = tqdm(range(0, len(model_to_orders), 2))
    for i in pbar:
        model1 = list(model_to_orders.keys())[i]
        model_name1 = map_name(model1, name_mappings).replace("CL: ", "") + model1.split()[-2]
        model2 = list(model_to_orders.keys())[i+1]
        model_name2 = map_name(model2, name_mappings).replace("CL: ", "") + model2.split()[-2]
        pbar.set_description(f"Correlation for pair {model_name1}-{model_name2}")
        orders1 = model_to_orders[model1]
        orders2 = model_to_orders[model2]
        i_curriculum = list(sorted(orders1, key=lambda x: orders1[x]))
        j_curriculum = list(sorted(orders2, key=lambda x: orders2[x]))
        corr_m = spearmanr(i_curriculum, j_curriculum)[0]
        # corr_per_pair[f"{model_name1}->{model_name2}"] = corr_m
        models.append(f"{model_name1} - {model_name2}")
        corrs.append(corr_m)
    # Figure Size

    models, corrs = list(map(list, zip(*sorted(zip(models, corrs), key=lambda x: x[1]))))

    fig, ax = plt.subplots(figsize =(18, 9))
    
    # Horizontal Bar Plot
    barplot = ax.barh(models, corrs)
    
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
    
    # Add Plot Title
    plt.title("Spearmann Correlation Between Model Pairs")

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
    parser.add_argument("--wer-suffix", "-s", required=False, default=None,
        help="How should the wer txt file be named? Overrides the '-v' and '-f' options.")
    parser.add_argument("--out-path", "-o", required=False, default=None,
        help="Location of the output bar plot.")
    parser.add_argument("--model-name-mappings", "-m", default=None, 
        help="Path to a .py file containing a dictionary with the keys\
              `curriculum_mappings`, `transfer_mappings` and `subset_mappings`.\
              This remains to be documented.")
    return parser

def _parse_args(args=None):
    if args is None:
        parser = _get_parser()
        args = parser.parse_args()
    if len(args.exps) % 2 != 0:
        raise argparse.ArgumentTypeError(f"You should provide pairs of models. \
            Instead you provided {args.exps}.")
    if args.wer_suffix is None:
        args.wer_file = "wer_test.txt"
    else:
        args.wer_file = f"wer_test{args.wer_suffix}.txt"
    testset_corr(
        args.exps, args.wer_file, args.out_path, 
        name_mappings_file=args.model_name_mappings
    )


if __name__ == "__main__":
    _parse_args()