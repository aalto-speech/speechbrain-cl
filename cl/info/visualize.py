import enum
import math
import os
from collections import Counter
from pyexpat import model
import sys
import random
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
# from numpy import mod
from .statmd import NoEpochsTrained, _read_stats, get_args, _read_args
from .find_anomalies import read_single
from .get_train_times import _find_best_epoch
from cl.info.globals import MPL_COLORS, MAPPINGS, MPL_MARKERS


plt.rcParams.update({'font.size': 19})

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

def plot_valid_results(paths, metric="WER", output_path=None):
    model_to_stats = {}
    # n_models = 0
    max_epoch = 0
    min_wer = 200
    max_wer = 0
    stats = {}
    for path in paths:
        try:
            epochs, _, _, valid_metrics_dict = _read_stats([path], [metric])
        except NoEpochsTrained:
            print(f"Model {path} ignored since it hasn't been trained.")
            continue
        # n_models += 1
        tmp = max(map(lambda x: int(x[0]), epochs))
        if tmp > max_epoch:
            max_epoch = tmp
        tmp_min = min(map(lambda w: float(w[0]), valid_metrics_dict[metric]))
        if tmp_min < min_wer:
            min_wer = tmp_min
        tmp_max = max(map(lambda w: float(w[0]), valid_metrics_dict[metric]))
        if tmp_max > max_wer:
            max_wer = tmp_max
        model_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        seed = os.path.basename(os.path.dirname(path)).split("-")[0]
        identifier = f"{model_id} ({seed})"
        if model_id in stats:
            stats[model_id]["valid_metrics_dicts"].append(valid_metrics_dict)
            stats[model_id]["epochs"].append(epochs)
            stats[model_id]["seeds"].append(seed)
        else:
            stats[model_id] = {"epochs": [epochs], "valid_metrics_dicts": [valid_metrics_dict], "seeds": [seed]}
    for model_id, d in stats.items():
        n_curr_models = len(d['seeds'])
        if n_curr_models == 1:
            identifier = f"{model_id} ({d['seeds'][0]})"
            model_to_stats[identifier] = {"epochs": d["epochs"][0], "valid_metrics_dict": d['valid_metrics_dicts'][0]}
        else:
            n_epochs_per_run = [len(e) for e in d['epochs']]
            if len(set(n_epochs_per_run)) < len(n_epochs_per_run):
                # Then we have at least two runs with the same number of epochs which we can average
                epoch_counts = Counter(n_epochs_per_run)
                # print(f"{epoch_counts=}\n{model_id=}\n===================\n")
                for n_epochs, count in epoch_counts.items():
                    epochs = [e for e in d['epochs'] if len(e) == n_epochs][0]
                    vm_dict = [vmd[metric] for i, vmd in enumerate(d['valid_metrics_dicts']) if len(d['epochs'][i])==n_epochs]
                    vm_dict = {metric: list(map(lambda y: sum(y)/len(y), map(lambda x: x[0], zip(*vm_dict))))}
                    vm_dict = {metric: [[e] for e in vm_dict[metric]]}
                    if count > 1:
                        identifier = f"{model_id} (#runs={count})"
                    else:
                        seed = [s for i, s in enumerate(d['seeds']) if len(d['epochs'][i])==n_epochs][0]
                        identifier = f"{model_id} ({seed})"
                    model_to_stats[identifier] = {"epochs": epochs, "valid_metrics_dict": vm_dict}
            else:
                for i, s in enumerate(d['seeds']):
                    identifier = f"{model_id} ({s})"
                    model_to_stats[identifier] = {"epochs": d["epochs"][i], "valid_metrics_dict": d['valid_metrics_dicts'][i]}
    n_models = len(model_to_stats)
        # model_to_stats[model_id]["valid_metrics_dict"] = [vm1[0]+vm2[0] for vm1, vm2 in zip(model_to_stats[model_id]["valid_metrics_dict"][metric], valid_metrics_dict[metric])]
        # model_to_stats[model_id]["seeds"].append(seed)
        
    random.shuffle(MPL_COLORS)
    random.shuffle(MPL_MARKERS)
    # MPL_COLORS = MPL_COLORS*(1+n_models//len(MPL_COLORS))[:n_models]
    # MPL_MARKERS = MPL_MARKERS*(1+n_models//len(MPL_MARKERS))[:n_models]
    # assert n_models <= len(MPL_COLORS), f"You will need to rotate the MPL_COLORS list. {epochs=}"
    step = 1
    start_epoch = 0
    zoom = 4
    if max_epoch > 50:
        step = 10
        start_epoch = 15  # first epoch to plot
        zoom=3.5
    elif max_epoch > 20:
        step = 5
        zoom = 3
    elif max_epoch > 15:
        step = 2

    max_allowed_wer = 70
    random_colors = (MPL_COLORS * (1+n_models//len(MPL_COLORS)))[:n_models]
    random_markers = (MPL_MARKERS * (1+n_models//len(MPL_MARKERS)))[:n_models]
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Model Performances', fontsize=25)
    ax = fig.add_subplot(111)
    plt.title(f"Model Performances on Validation Set")
    x_axis = list(range(1, max_epoch+1))
    plot_data = {}
    # for i, path in enumerate(model_to_stats):
    for i, identifier in enumerate(model_to_stats):
        # model_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        epochs, valid_metrics_dict = list(model_to_stats[identifier].values())
        best_valid_epoch, best_valid_wer = _find_best_epoch(epochs, valid_metrics_dict, metric=metric)
        # model_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        # seed = os.path.basename(os.path.dirname(path)).split("-")[0]
        # identifier = f"{model_id} ({seed})"
        vms = [min(100, vm[0]) for vm in valid_metrics_dict[metric]]
        # if len(vms) < max_epoch:
        #     vms += [vms[-1]] * (max_epoch-len(vms))
        x_axis = [int(e[0]) for e in epochs]
        best_epoch_index = [ind for ind, x in enumerate(x_axis) if int(x) == best_valid_epoch][0]
        vm_best_wer = [y for ind, y in enumerate(vms) if ind == best_epoch_index][0]
        vm_best_wer = min(max_allowed_wer, vm_best_wer)
        ax.plot(x_axis, vms, marker=random_markers[i], linewidth=4, label=f"{identifier}", color=random_colors[i])
        ax.text(best_valid_epoch, vm_best_wer, f"{best_valid_wer}")
        ax.plot([best_valid_epoch], [vm_best_wer], random_markers[i], ms=14, color=random_colors[i])
        ax.set_xticks(list(range(0, max_epoch+1, step)))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Metric {metric}")
        ax.set_xlim([start_epoch, max_epoch + step])
        ax.set_ylim([min_wer-2, max_allowed_wer])
        ax.legend(loc='upper left')
        plot_data[identifier] = {
            "plot1": {
                "args": [x_axis, vms], 
                "kwargs": {"marker": random_markers[i], "linewidth": 4, "label": f"{identifier}", "color": random_colors[i]}
            },
            "text": {
                "args": [best_valid_epoch, vm_best_wer, f"{best_valid_wer}"], "kwargs": {}
            },
            "plot2": {
                "args": [[best_valid_epoch], [vm_best_wer], random_markers[i]], 
                "kwargs": {"ms": 14, "color": random_colors[i]}
            }
        }
    axins = zoomed_inset_axes(ax, zoom=zoom, loc="center right")
    # fix the number of ticks on the inset axes
    axins.yaxis.get_major_locator().set_params(nbins=7)
    axins.xaxis.get_major_locator().set_params(nbins=7)
    # axins.tick_params(labelleft=False, labelbottom=False)
    # sub region of the original image
    x1, x2, y1, y2 = max_epoch-min(10, step), max_epoch+min(step/2, 2), min_wer-1, min_wer+4
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    for identifier in model_to_stats:
        p = plot_data[identifier]
        axins.plot(*p['plot1']['args'], **p['plot1']['kwargs'])
        best_epoch = p['text']['args'][0]
        best_wer = p['text']['args'][1]
        if best_epoch >= x1 and best_wer <= y2:
            axins.text(*p['text']['args'], **p['text']['kwargs'])
            axins.plot(*p['plot2']['args'], **p['plot2']['kwargs'])

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    # fig.tight_layout()
    if (output_path is None) or not (os.path.isdir(os.path.dirname(output_path))):
        print("Showing final plot...")
        plt.show()
    else:
        print("Saving plot under:", output_path)
        plt.savefig(output_path)


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

def _testset_boxplot_single(wer_file, model_name=None, max_allowed_score=250.0, metric="WER"):
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
        wer = float(l.split(metric)[1].split("[")[0].strip())
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
    if args.plot_valid_results is True:
        return plot_valid_results(paths, metrics[0], output_path)
    return plot_logs(paths, metrics, output_path, args.print_seed)

def main():

    print_seed = False
    if "--print-seed" in sys.argv:
        print_seed = True
        sys.argv.pop(sys.argv.index("--print-seed"))
    paths, metrics, output_path, _ = get_args()
    plot_logs(paths, metrics, output_path, print_seed)
