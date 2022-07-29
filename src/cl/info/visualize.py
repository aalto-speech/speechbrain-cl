import json
import math
import os
from collections import Counter
import sys
import random
import numpy as np
import warnings
# from numpy import mod
from .statmd import NoEpochsTrained, _read_stats, get_args, _read_args
from .find_anomalies import read_single
from .get_train_times import _find_best_epoch
from cl.info.globals import DARK_COLORS, MPL_COLORS, MAPPINGS, MPL_MARKERS, map_name, map_name_thesis

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    import seaborn as sns
    sns.set()
    plt.rcParams.update({'font.size': 19})
except ImportError:
    # warnings.warn("Could not import matplotlib. If you are planning to use the visualization functions then you need to install it.")
    pass



labels = []
def add_label(violin, label, labs=labels):
    # adds label to violinplot:
    # https://stackoverflow.com/questions/33864578/matplotlib-making-labels-for-violin-plots
    color = violin["bodies"][0].get_facecolor().flatten()
    labs.append((mpatches.Patch(color=color), label))

def capitalize(l: list):
    l = [s.capitalize() for s in l]  # capitalize all
    l = list(map(lambda x: MAPPINGS.get(x, x), l))  # check mappings
    return " ".join(l) # return single string

def plot_valid_results(paths, metric="WER", 
  output_path=None, 
  plot_axins=False, 
  show_marker_text=False,
  plot_title="Validation Set WERs"):
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
        # identifier = f"{model_id} ({seed})"
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
    # random_colors = ["gray", "brown", "olive", "black"]
    # random_colors = (DARK_COLORS * (1+n_models//len(DARK_COLORS)))[:n_models]
    random_markers = (MPL_MARKERS * (1+n_models//len(MPL_MARKERS)))[:n_models]
    fig = plt.figure(figsize=(16, 12))
    # fig.suptitle('Model Performances on Validation Set', fontsize=25)
    ax = fig.add_subplot(111)
    title = plot_title or f"Model Performances on Validation Set"
    plt.title(title, fontsize=20)
    x_axis = list(range(1, max_epoch+1))
    plot_data = {}
    csv_lines = np.empty((max_epoch, len(model_to_stats)+1))
    csv_lines[:, 0] = list(range(1, max_epoch+1))
    mapped_names = []
    # for i, path in enumerate(model_to_stats):
    for i, identifier in enumerate(model_to_stats):
        identifier_latex = map_name_thesis(identifier).split(" (#")[0].split("(1")[0]
        mapped_names.append(identifier_latex)
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
        csv_lines[:len(vms), i+1] = vms
        ax.plot(x_axis, vms, marker=random_markers[i], ms=4, linewidth=4, label=f"{identifier_latex}", color=random_colors[i])
        
        if best_epoch_index > start_epoch and vm_best_wer<=max_allowed_wer:
            if show_marker_text:
                ax.text(best_valid_epoch, vm_best_wer, f"{best_valid_wer}")
            ax.plot([best_valid_epoch], [vm_best_wer], random_markers[i], ms=11, color=random_colors[i])
        ax.set_xticks(list(range(0, max_epoch+1, step))+[max_epoch], fontsize=16)
        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel(f"{metric} score", fontsize=16)
        ax.set_xlim([start_epoch, max_epoch+1])
        ax.set_ylim([min_wer-2, max_allowed_wer])
        ax.legend(loc='upper right', prop={'size': 18})
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
    if plot_axins:
        axins = zoomed_inset_axes(ax, zoom=zoom, loc="upper right", borderpad=2.0)
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
    fig.tight_layout()
    if (output_path is None) or not (os.path.isdir(os.path.dirname(output_path))):
        print("Showing final plot...")
        plt.show()
    else:
        print("Saving plot under:", output_path)
        plt.savefig(output_path)
        np.savetxt(
            ".".join(output_path.split(".")[:-1]) + ".csv", 
            csv_lines,
            delimiter=",",
            header="Epoch," + ",".join(mapped_names)
        )

def plot_train_valid_loss(
        paths, output_path=None,
        name_mappings_file=None,
        include_n_runs: bool = False,
    ):
    model_to_stats = {}
    # n_models = 0
    max_epoch = 0
    min_train = 200
    min_val = 200
    stats = {}
    max_loss = 0
    if name_mappings_file and os.path.isfile(name_mappings_file):
        with open(name_mappings_file, 'r') as f:
            name_mappings = json.loads(f.read())
    else:
        name_mappings = None
    for path in paths:
        try:
            epochs, train_losses, valid_losses, _ = _read_stats([path])
        except NoEpochsTrained:
            print(f"Model {path} ignored since it hasn't been trained.")
            continue
        # n_models += 1
        tmp = max(map(lambda x: int(x[0]), epochs))
        if tmp > max_epoch:
            max_epoch = tmp
        tls = list(map(lambda l: float(l[0]), train_losses))
        vls = list(map(lambda l: float(l[0]), valid_losses))
        tmp_min_train = min(tls)
        if tmp_min_train < min_train:
            min_train = tmp_min_train
        tmp_min_val = min(vls)
        if tmp_min_val < min_val:
            min_val = tmp_min_val
        tmp_max_loss = max(tls+vls)
        if tmp_max_loss > max_loss:
            max_loss = tmp_max_loss
        model_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        seed = os.path.basename(os.path.dirname(path)).split("-")[0]
        identifier = f"{model_id} ({seed})"
        if model_id in stats:
            stats[model_id]["train_losses"].append(train_losses)
            stats[model_id]["valid_losses"].append(valid_losses)
            stats[model_id]["epochs"].append(epochs)
            stats[model_id]["seeds"].append(seed)
        else:
            stats[model_id] = {
                "epochs": [epochs],
                "train_losses": [train_losses],
                "valid_losses": [valid_losses],
                "seeds": [seed],
            }
    def sum_losses_list(d, d_losses_key, n_epochs):
        l_list = [tld for i, tld in enumerate(d[d_losses_key]) if len(d['epochs'][i])==n_epochs]
        l_list = list(map(lambda y: sum(y)/len(y), map(lambda x: x[0], zip(*l_list))))
        l_list = [[e] for e in l_list]
        return l_list
    for model_id, d in stats.items():
        n_curr_models = len(d['seeds'])
        # model_id = map_name(model_id, name_mappings)
        model_id = map_name_thesis(model_id)
        if n_curr_models == 1:
            identifier = f"{model_id} ({d['seeds'][0]})"
            model_to_stats[identifier] = {
                "epochs": d["epochs"][0],
                "train_losses": d['train_losses'][0],
                "valid_losses": d['valid_losses'][0],
            }
        else:
            n_epochs_per_run = [len(e) for e in d['epochs']]
            if len(set(n_epochs_per_run)) < len(n_epochs_per_run):
                # Then we have at least two runs with the same number of epochs which we can average
                epoch_counts = Counter(n_epochs_per_run)
                # print(f"{epoch_counts=}\n{model_id=}\n===================\n")
                for n_epochs, count in epoch_counts.items():
                    epochs = [e for e in d['epochs'] if len(e) == n_epochs][0]
                    tls = sum_losses_list(d, "train_losses", n_epochs)
                    vls = sum_losses_list(d, "valid_losses", n_epochs)
                    if count > 1:
                        identifier = f"{model_id} (#runs={count})"
                    else:
                        seed = [s for i, s in enumerate(d['seeds']) if len(d['epochs'][i])==n_epochs][0]
                        identifier = f"{model_id} ({seed})"
                    model_to_stats[identifier] = {
                        "epochs": epochs,
                        "train_losses": tls,
                        "valid_losses": vls,
                    }
            else:
                for i, s in enumerate(d['seeds']):
                    identifier = f"{model_id} ({s})"
                    model_to_stats[identifier] = {
                        "epochs": d["epochs"][i],
                        "train_losses": d['train_losses'][i],
                        "valid_losses": d['valid_losses'][i],
                    }
    n_models = len(model_to_stats)

    random.shuffle(MPL_COLORS)
    random.shuffle(MPL_MARKERS)
    # MPL_COLORS = MPL_COLORS*(1+n_models//len(MPL_COLORS))[:n_models]
    # MPL_MARKERS = MPL_MARKERS*(1+n_models//len(MPL_MARKERS))[:n_models]
    # assert n_models <= len(MPL_COLORS), f"You will need to rotate the MPL_COLORS list. {epochs=}"
    step = 1
    if max_epoch > 50:
        step = 10
    elif max_epoch > 20:
        step = 5
    elif max_epoch > 15:
        step = 2

    random_colors = (DARK_COLORS * (1+(n_models*2)//len(DARK_COLORS)))[:n_models*2]
    random_markers = (MPL_MARKERS * (1+(n_models*2)//len(MPL_MARKERS)))[:n_models*2]
    # fig = plt.figure(figsize=(16, 12))
    # fig, axes = plt.subplots(2, math.ceil(len(model_to_stats)/2), sharex=True, figsize=(5.5,7))
    fig, axes = plt.subplots(2, math.ceil(len(model_to_stats)/2), sharey=True, figsize=(5, 6))
    # fig.suptitle('Model Performances', fontsize=25)
    # plt.title(f"Model Performances on Validation Set")
    x_axis = list(range(1, max_epoch+1))
    print("Number of models:", len(model_to_stats))
    for i, identifier in enumerate(model_to_stats):
        # model_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        epochs, train_losses, valid_losses = list(model_to_stats[identifier].values())
        tls = [tl[0] for tl in train_losses]
        vls = [vl[0] for vl in valid_losses]
        x_axis = [int(e[0]) for e in epochs]
        x_axis = list(range(1, len(epochs)+1))

        # Plot possition
        if len(model_to_stats) == 1:
            current_ax = axes
        else:
            # current_ax = axes[i]
            current_ax = axes[i]#axes[i//2][i%2] if len(model_to_stats)>2 else axes[i]
        if not include_n_runs:
            identifier = identifier.split("(")[0]
        assert len(x_axis) == len(tls) == len(vls)
        sns.regplot(x=x_axis, y=tls,
            ci=None, scatter=False, order=4, label="Train Loss",
            ax=current_ax, color=random_colors[i*2]
        )
        sns.regplot(x=x_axis, y=vls,
            ci=None, scatter=False, order=4, label="Valid Loss",
            ax=current_ax, color=random_colors[i*2+1],
            line_kws={'linewidth':3, 'linestyle': '--'}
        )
        current_ax.set_title(f"{identifier}", fontsize=17)
        # current_ax.set_xticks(list(range(0, max_epoch+1, step)))
        current_ax.legend(loc='upper right', fontsize=15)
        # current_ax.set_xlim([0, max_epoch+step])
        # current_ax.set_ylim([min(min_train, min_val)-1, max_loss])
        if i >= 1:#len(model_to_stats)-2:
            current_ax.set_xlabel("Epoch", fontsize=16)
        # current_ax.set_xticklabels(x_axis)
        if i % 2 != -1:
            current_ax.set_ylabel("Loss", fontsize=16)

    fig.tight_layout()
    if (output_path is None) or not (os.path.isdir(os.path.dirname(output_path))):
        print("Showing final plot...")
        plt.show()
    else:
        print("Saving plot under:", output_path)
        plt.savefig(output_path)

def valid_scores_grouped_bp(
        paths,
        metric='WER',
        output_path=None,
        name_mappings_file=None,
        include_n_runs: bool = False,
    ):
    model_to_stats = {}
    # n_models = 0
    stats = {}
    max_epoch = 0
    if name_mappings_file and os.path.isfile(name_mappings_file):
        with open(name_mappings_file, 'r') as f:
            name_mappings = json.loads(f.read())
    else:
        name_mappings = None
    for path in paths:
        try:
            epochs, _, _, valid_metrics_dict = _read_stats([path], [metric])
        except NoEpochsTrained:
            print(f"Model {path} ignored since it hasn't been trained.")
            continue
        tmp = max(map(lambda x: int(x[0]), epochs))
        if tmp > max_epoch:
            max_epoch = tmp
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
        model_id = map_name(model_id, name_mappings)
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
    fig = plt.figure(figsize=(15, 12))
    barplot_list = []

    step = 3
    if max_epoch > 50:
        step = 15
    elif max_epoch > 30:
        step = 10
    elif max_epoch > 15:
        step = 5
    for i, identifier in enumerate(model_to_stats):
        # model_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        epochs, valid_metrics_dict = list(model_to_stats[identifier].values())
        if not include_n_runs:
            identifier = identifier.split("(")[0]
        identifier_latex = map_name_thesis(identifier)
        # epochs = [int(e[0]) for e in epochs]
        epochs = list(range(0, max_epoch))[::step]
        epochs[0] = 1  # start from epoch 1
        epochs[-1] = max_epoch
        metric_vals = [min(100, vm[0]) for vm in valid_metrics_dict[metric]]
        last_score = metric_vals[-1]
        metric_vals = metric_vals[::step]
        metric_vals[-1] = last_score
        print(identifier, max_epoch, last_score)
        names = ([identifier_latex]*len(epochs))
        l = list(zip(names, epochs, metric_vals))
        barplot_list += l
    multiple_bar_plots(barplot_list, values_name=f"{metric}")
    plt.legend(loc='upper right', prop={'size': 25})
    plt.xticks(size = 19)
    plt.yticks(size = 19)
    fig.tight_layout()
    if (output_path is None) or not (os.path.isdir(os.path.dirname(output_path))):
        print("Showing final plot...")
        plt.show()
    else:
        print("Saving plot under:", output_path)
        plt.savefig(output_path)


def multiple_bar_plots(barplot_list, values_name="Values", show_text=False):
    import pandas as pd
    sns.set(font_scale=2, rc={"figure.figsize":(9, 12)})
    names, epochs, metric_vals = zip(*barplot_list)
    n_models = len(set(names))
    d = {
        "Model Name": names,
        "Epoch": epochs,
        values_name: metric_vals,
    }
    df = pd.DataFrame(d)
    g = sns.catplot(
        data=df, kind="bar", color=plt.cm.gray,
        palette=sns.color_palette("dark:#D6D6D6"),
        x="Epoch", y=values_name, hue="Model Name",
        ci="sd",
        # palette="dark",
        alpha=.65,
        legend=False, height=6.5, aspect=2.2,
    )
    plt.setp(g.ax.patches, linewidth=1, edgecolor="k")
    ax = g.facet_axis(0, 0)
    hatch_markers = ['/', '\\', '-', '+', 'x', '.', 'O', 'o', '*']
    hatches = []
    for i in range(len(names)//n_models):
        hatches += [hatch_markers[i]]*n_models
    print(hatches, n_models)
    for i, bar in enumerate(ax.patches):
        # if i % num_locations == 0:
        hatch = hatches[i]
        bar.set_hatch(hatch)
    g.fig.set_size_inches(11, 6)
    g.despine(left=True)
    if show_text:
        # extract the matplotlib axes_subplot objects from the FacetGrid
        ax = g.facet_axis(0, 0)
        # Set WER values as text on top of the barplots
        for i in ax.patches:
            wer = round((i.get_height()), 1)
            # if wer != float('nan'):
            #     wer = int(wer) if wer == int(wer) else wer
            ax.text(i.get_x(), i.get_height()+0.2,
                    str(wer),
                    fontsize = 10, fontweight ='bold',
                    color ='grey')


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

def _testset_boxplot_single(wer_file, model_name=None, max_allowed_score=250.0, metric="WER", return_all_edits=False):
    wer_lines = read_single(wer_file, print_stats=False)
    ins, dels, subs = [], [], []
    changes = []
    for entry in wer_lines:
        n_ins, n_dels, n_subs = map(lambda x: min(float(x), max_allowed_score), entry[2:5])
        ins.append(n_ins)
        dels.append(n_dels)
        subs.append(n_subs)
        changes.append(sum(map(float, entry[2:5])))
    # print(f"{model_name}: \t\t\tInsertions={int(sum(ins))}, Deletions={int(sum(dels))}, Substitutions={int(sum(subs))}.")
    with open(wer_file, 'r') as f:
        l = f.readlines()[0]
        wer = float(l.split(metric)[1].split("[")[0].strip())
    print(f"| {wer_file.split('seq2seq/')[1]}\t|\t{wer}\t|\t{int(sum(ins))}\t|\t{int(sum(dels))}\t|\t{int(sum(subs))}\t|")

    if return_all_edits:
        return changes, wer, ins, dels, subs
    plt.violinplot([ins, dels, subs])
    plt.xticks([1, 2, 3], ['Ins', 'Dels', 'Subs'])
    plt.title(model_name, fontsize=15)
    return changes, wer

def violinplots_by3(args):
    n_files = len(args.wer_paths)
    out_path_gen = args.out_path
    if out_path_gen is not None:
        main_name, ext = os.path.splitext(out_path_gen)
    if n_files > 4:
        sub_wer_files = [args.wer_paths[i-3:i] for i in range(3, n_files, 3)]
        if n_files % 3 != 0:
            sub_wer_files.append(args.wer_paths[-(n_files%3):])
        for i, sub_list in enumerate(sub_wer_files):
            out_path = None
            if out_path_gen is not None:
                out_path = main_name + f"_{i}" + ext
            args.out_path = out_path
            args.wer_paths = sub_list
            testset_boxplot_comparison(args)
    else:
        testset_boxplot_comparison(args)


def testset_boxplot_comparison(args):
    # print(args)
    labels = []
    out_path = getattr(args, "out_path", None)
    wer_files = args.wer_paths
    return_all_edits = getattr(args, "return_all_edits", True)
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
    n_cols = 2#math.ceil(math.sqrt(l))
    n_rows = 2#math.ceil(l/n_cols)
    print(f"{l=}, {n_cols=}, {n_rows=}")
    fig = plt.figure(figsize=(8, 6))
    # fig.suptitle('Insertion/Deletion/Substitution Distribution', fontsize=25)
    if return_all_edits:
        insertions, deletions, substitutions = [], [], []
    for i, wer_file in enumerate(final_wer_files):
        model_name = os.path.basename(os.path.dirname(os.path.dirname(wer_file)))
        # model_name = re.sub("noshards|sharded|exps|segmented|fixed_text|train", "", model_name)
        # model_name = re.sub("_|-", " ", model_name)
        # model_name = " ".join(w.capitalize() for w in re.sub("\s+", " ", model_name).split()).strip()
        # if model_name == "Complete":
        #     model_name = "Complete (Ascending)"
        model_name = map_name_thesis(model_name)
        # print(f"Processing: {wer_file}")
        if not return_all_edits:
            plt.subplot(n_cols, n_rows, i + 1)
        single_res = _testset_boxplot_single(
            wer_file, 
            model_name,
            return_all_edits=return_all_edits
        )
        if return_all_edits:
            insertions.append((model_name, single_res[2]))
            deletions.append((model_name, single_res[3]))
            substitutions.append((model_name, single_res[4]))
            
        wer_changes[model_name] = single_res
    if return_all_edits:
        # Create insertions plot
        plt.subplot(n_cols, n_rows, 1)
        mnames = []
        for pos, (mname, ins) in enumerate(insertions):
            v = plt.violinplot([ins], [pos])
            mnames.append(mname)
            add_label(v, mname, labels)
        plt.xticks(list(range(len(insertions))), mnames)
        # plt.legend(*zip(*labels), prop={'size': 8}, framealpha=0.65)
        plt.title("Insertions", fontsize=15)
        labels = []
        plt.subplot(n_cols, n_rows, 2)
        for pos, (mname, dels) in enumerate(deletions):
            v = plt.violinplot([dels], [pos])
            add_label(v, mname, labels)
        # plt.legend(*zip(*labels), prop={'size': 8})
        plt.xticks(list(range(len(insertions))), mnames)
        plt.title("Deletions", fontsize=15)
        labels = []
        plt.subplot(n_cols, n_rows, 3)
        for pos, (mname, subs) in enumerate(substitutions):
            v = plt.violinplot([subs], [pos])
            add_label(v, mname, labels)
        # plt.legend(*zip(*labels), prop={'size': 8})
        plt.xticks(list(range(len(insertions))), mnames)
        plt.title("Substitutions", fontsize=15)
        i=2
        labels = []
        latex_table_str = ""
        for ins, dels, subs in zip(insertions, deletions, substitutions):
            assert ins[0] == dels[0] == subs[0], f"{ins[0]=}"
            mname = ins[0]
            tot = int(sum(wer_changes[mname][0]))
            ins, dels, subs = int(sum(ins[1])), int(sum(dels[1])), int(sum(subs[1]))
            entry = f"{mname} & {ins} & {dels} & {subs} & {tot} \\\\\n\hline\n"
            latex_table_str += entry
        print(latex_table_str)
    plt.subplot(n_cols, n_rows, i + 2)
    for pos, m in enumerate(wer_changes.keys()):
        v = plt.violinplot([wer_changes[m][0]], [pos])
        add_label(v, m + f" (WER={wer_changes[m][1]})", labels)
    # plt.legend(*zip(*labels), prop={'size': 8})
    plt.xticks(list(range(len(insertions))), mnames)
    # plt.xticks(np.linspace(1, l, num=l, dtype=np.int16), wer_changes.keys(), fontsize=15)
    plt.title("Total Edits", fontsize=15)
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
        if args.barplot is False:
            return plot_valid_results(paths, metrics[0], output_path)
        return valid_scores_grouped_bp(paths, metrics[0], output_path, args.model_name_mappings)
    # return plot_logs(paths, metrics, output_path, args.print_seed)
    return plot_train_valid_loss(paths, output_path, args.model_name_mappings)

def main():

    print_seed = False
    if "--print-seed" in sys.argv:
        print_seed = True
        sys.argv.pop(sys.argv.index("--print-seed"))
    paths, metrics, output_path, _ = get_args()
    plot_logs(paths, metrics, output_path, print_seed)
