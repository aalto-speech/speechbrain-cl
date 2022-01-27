import math
import os
import sys
import random
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
# from numpy import mod
from .statmd import _read_stats, get_args, _read_args
from .find_anomalies import read_single


plt.rcParams.update({'font.size': 19})

# List of possible colors for matplotlib
MPL_COLORS = [
    "blue", "orange", "green", "red", "yellow", "black", 
    "brown", "pink", "gray", "olive", "cyan", "purple"
]

MAPPINGS = {
    "Crdnn": "CRDNN",
    "Wer": "WER",
    "Cer": "CER",
    "Seqloss": "Sequence-to-Sequence Loss",
    "Confs": "w/ Confidences",
    "5k": "",
    "Seqctcloss": "Sequence-to-Sequence w/ CTC Loss",
    "Simple": "Ascending",
    "Base": "Ascending",
    "W2v2": "Wav2Vec 2.0",
}


labels = []
def add_label(violin, label):
    # adds label to violinplot:
    # https://stackoverflow.com/questions/33864578/matplotlib-making-labels-for-violin-plots
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))

def capitalize(l: list):
    l = [s.capitalize() for s in l]  # capitalize all
    l = list(map(lambda x: MAPPINGS.get(x, x), l))  # check mappings
    return " ".join(l) # return single string


def plot_logs(paths, metrics=["PER"], output_path=None, print_seed=False):
    epochs, train_losses, valid_losses, valid_metrics_dict = _read_stats(paths, metrics)
    assert len(epochs[0]) <= len(MPL_COLORS), f"You will need to rotate the MPL_COLORS list. {epochs=}"
    assert len(epochs) == len(train_losses) == len(valid_losses) == len(valid_metrics_dict[metrics[0]]), f"{epochs=}\n{train_losses=}\n{valid_losses=}\n{valid_metrics_dict=}"
    if len(epochs[0]) > 1:
        for i, e in enumerate(epochs):
            if len(set(e)) > 1:
                epochs[i] = (e[0],)*len(e)
        assert all([len(set(e)) == 1 for e in epochs]), epochs
    x_axis = [e[0] for e in epochs]  # for each epoch
    random.shuffle(MPL_COLORS)
    n_different_colors = 1 + 1 + len(metrics)  # train, valid loss and for each metric
    random_colors = MPL_COLORS[:n_different_colors]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Model Performances', fontsize=25)
    for i in range(len(paths)):
        tls = [tl[i] for tl in train_losses]
        vls = [vl[i] for vl in valid_losses]
        seed = paths[i].split("results/")[-1].split("/")[1]
        label = paths[i].split("results/")[-1].split("/")[0].split("_")
        if "random" in paths[i]:
            label += [seed]
        elif print_seed:
            label += [seed]
        label = capitalize(label)
        n_cols = 1 if len(paths) == 1 else len(metrics)
        n_rows = len(metrics) if len(paths) == 1 else len(paths)

        plt.subplot(n_cols, n_rows, i+1)
        plt.plot(x_axis, tls, label="Train Loss", color=random_colors[0])  # rotate here (with mod)
        plt.plot(x_axis, vls, label="Valid Loss", color=random_colors[1])
        plt.xticks(x_axis[::len(paths)+1])
        plt.legend(loc='upper right')
        if (len(paths) > 1 and i < len(paths)) or (len(paths) == 1):
            plt.title(f"Model: {label}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(n_cols, n_rows, i + len(paths) + 1)
        for j, metric in enumerate(metrics):
            vms = [vm[i] for vm in valid_metrics_dict[metric]]
            plt.plot(x_axis, vms, label=f"{metric}", color=random_colors[2+j])
        plt.xticks(x_axis[::len(paths)+1])
        plt.legend(loc='upper right')
        # if len(paths) > 1 and i == 0:
        if len(paths) == 1:
            plt.title(f"Model: {label}")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Scores")
    fig.tight_layout()
    if (output_path is None) or not (os.path.isdir(os.path.dirname(output_path))):
        print("Showing final plot...")
        plt.show()
    else:
        print("Saving plot under:", output_path)
        plt.savefig(output_path)

def _testset_boxplot_single(wer_file, model_name=None, max_allowed_score=250.0):
    wer_lines = read_single(wer_file, print_stats=False)
    ins, dels, subs = [], [], []
    changes = []
    for entry in wer_lines:
        n_ins, n_dels, n_subs = map(lambda x: min(float(x), max_allowed_score), entry[2:5])
        ins.append(n_ins)
        dels.append(n_dels)
        subs.append(n_subs)
        changes.append(sum(map(float, entry[2:5])))
    plt.violinplot([ins, dels, subs])
    plt.xticks([1, 2, 3], ['Ins', 'Dels', 'Subs'])
    plt.title(model_name, fontsize=15)
    # print(f"{model_name}: \t\t\tInsertions={int(sum(ins))}, Deletions={int(sum(dels))}, Substitutions={int(sum(subs))}.")
    with open(wer_file, 'r') as f:
        l = f.readlines()[0]
        wer = float(l.split("WER")[1].split("[")[0].strip())
    print(f"| {wer_file.split('seq2seq/')[1]}\t|\t{wer}\t|\t{int(sum(ins))}\t|\t{int(sum(dels))}\t|\t{int(sum(subs))}\t|")
    
    return changes

def testset_boxplot_comparison(args):
    print(args)
    out_path = getattr(args, "out_path")
    wer_files = args.wer_paths
    assert len(wer_files) >= 1, f"You need to provide at least one wer_test.txt file or directory ({wer_files=})."
    wer_changes = {}
    final_wer_files = []
    for wer_file in wer_files:
        assert os.path.exists(wer_file), f"Could not locate {wer_file}."
        wer_file = os.path.abspath(wer_file)
        if os.path.isdir(wer_file):
            wer_file = os.path.join(wer_file, "wer_test.txt")
        if not os.path.exists(wer_file):
            if args.silent_ignore is True:
                continue
            else:
                raise FileNotFoundError(f"File {wer_file} should exist.")
        final_wer_files.append(wer_file)
    l = len(final_wer_files) + 1
    # n_cols*n_rows = l
    n_cols = math.ceil(math.sqrt(l))
    n_rows = math.ceil(l/n_cols)
    print(f"{l=}, {n_cols=}, {n_rows=}")
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('Insertion/Deletion/Substitution Distribution', fontsize=25)
    for i, wer_file in enumerate(final_wer_files):
        model_name = os.path.basename(os.path.dirname(os.path.dirname(wer_file)))
        model_name = re.sub("noshards|sharded|exps|segmented|fixed_text|train", "", model_name)
        model_name = re.sub("_|-", " ", model_name)
        model_name = " ".join(w.capitalize() for w in re.sub("\s+", " ", model_name).split()).strip()
        if model_name == "Complete":
            model_name = "Complete (Ascending)"
        # print(f"Processing: {wer_file}")
        plt.subplot(n_cols, n_rows, i + 1)
        wer_changes[model_name] = _testset_boxplot_single(wer_file, model_name)
    plt.subplot(n_cols, n_rows, i + 2)
    for pos, m in enumerate(wer_changes.keys()):
        v = plt.violinplot([wer_changes[m]], [pos])
        add_label(v, m)
    plt.legend(*zip(*labels), prop={'size': 6})
    # plt.xticks(np.linspace(1, l, num=l, dtype=np.int16), wer_changes.keys(), fontsize=15)
    plt.title("Number of changes for each model", fontsize=12)
    fig.tight_layout()
    if out_path is not None:
        out_path = os.path.abspath(out_path)
        if not os.path.isdir(os.path.dirname(out_path)):
            print(f"Creating directory: {os.path.dirname(out_path)}.")
            os.makedirs(os.path.dirname(out_path))
        plt.savefig(out_path)
        print(f"Figure saved under: {out_path}")
    else:
        plt.show()
        
def plot_logs_dispatcher(args):
    paths, metrics, output_path, _ = _read_args(args)
    return plot_logs(paths, metrics, output_path, args.print_seed)

def main():

    print_seed = False
    if "--print-seed" in sys.argv:
        print_seed = True
        sys.argv.pop(sys.argv.index("--print-seed"))
    paths, metrics, output_path, _ = get_args()
    plot_logs(paths, metrics, output_path, print_seed)
