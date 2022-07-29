import argparse
import json
from operator import sub
import os
import glob
import warnings
from collections import Counter
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
except ImportError:
    # warnings.warn("Could not import matplotlib. If you are planning to use the visualization functions then you need to install it.")
    pass
from cl.info.globals import MPL_COLORS
from cl.info.time_stats import calculate_total_hours_seen
from .statmd import _read_stats, NoEpochsTrained

def get_train_times(log_txt_pattern, silent=False):
    model_to_times = {}
    seeds = {}
    iterator = _get_iterator(log_txt_pattern)
    for path in iterator:
        if "subsample" in path:
            continue
        model_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        seed = os.path.basename(os.path.dirname(path)).split("-")[0]
        identifier = f"{model_id} ({seed})"
        # identifier = os.path.dirname(os.path.dirname(path))
        with open(path, 'r') as f:
            epoch, mins, last_epoch_seen = None, None, None
            minutes_per_epoch = {}
            for line in f:
                if "Going into epoch" in line:
                    epoch = int(line.split()[-1].replace("\n", "").strip())
                    last_epoch_seen = epoch
                    continue
                if "Currently training for" in line:
                    mins = float(line.split("for ")[-1].split(" minutes")[0].strip())
                    if mins <= 0 or mins in minutes_per_epoch.values():
                        continue
                    if epoch is None:
                        assert last_epoch_seen is not None, "This shouldn't happen."
                        if last_epoch_seen in minutes_per_epoch.keys():
                            minutes_per_epoch[last_epoch_seen] = max(mins, minutes_per_epoch[last_epoch_seen])
                            continue
                    assert epoch is not None, "Model id: {}, Epoch: {}, mins: {}\nDict: {}".\
                        format(identifier, epoch, mins, minutes_per_epoch)
                    minutes_per_epoch[epoch] = mins
                    epoch, mins = None, None
                    continue
        if len(minutes_per_epoch) == 0:
            msg = "Model {} has not been trained yet.".format(identifier)
            if silent:
                print(msg)
                continue
            raise NoEpochsTrained(msg)
        total_time = max(minutes_per_epoch.values()) / 60  # to hours
        if model_id in model_to_times:
            model_to_times[model_id]["minutes_per_epoch"].append(minutes_per_epoch)
            model_to_times[model_id]["seeds"].append(seed)
        else:
            model_to_times[model_id] = dict(minutes_per_epoch=[minutes_per_epoch], seeds=[seed])
        # model_to_times[identifier] = minutes_per_epoch
        print("Model {} took: \t\t {} hours (epochs={})".format(identifier, total_time, len(set(minutes_per_epoch.keys()))))
    out = {}
    for model_id, d in model_to_times.items():
        minutes_per_epoch = d['minutes_per_epoch']
        if len(minutes_per_epoch) == 1:
            out[model_id] = minutes_per_epoch[0]
        else:
            n_epochs_per_run = [len(list(e.values())) for e in minutes_per_epoch]
            if len(set(n_epochs_per_run)) < len(n_epochs_per_run):
                # Then we have at least two runs with the same number of epochs which we can average
                epoch_counts = Counter(n_epochs_per_run)
                # print(f"{minutes_per_epoch=}\n{model_id=}, seeds={d['seeds']}, counts={n_epochs_per_run}\n===================\n")
                for n_epochs, count in epoch_counts.items():
                    mins_per_epoch_current = {}
                    common_indices = [i for i in range(len(n_epochs_per_run)) if len(minutes_per_epoch[i]) == n_epochs]
                    relevant = [e for i, e in enumerate(minutes_per_epoch) if i in common_indices]
                    relevant_epochs = relevant[0].keys()
                    for e in relevant_epochs:
                        times = list(map(lambda x: x[e], relevant))
                        mins_per_epoch_current[e] = sum(times)/len(times)
                    if count > 1:
                        identifier = f"{model_id} (#runs={count})"
                    else:
                        seed = [s for i, s in enumerate(d['seeds']) if len(minutes_per_epoch[i].keys())==n_epochs][0]
                        identifier = f"{model_id} ({seed})"
                    out[identifier] = mins_per_epoch_current
            else:
                for i, s in enumerate(d['seeds']):
                    identifier = f"{model_id} ({s})"
                    out[identifier] = minutes_per_epoch[i]
    return out

def plot_train_times(log_txt_pattern, out_path=None, silent=False):
    model_to_times = get_train_times(log_txt_pattern, silent)
    model_to_best_val = _get_best_epoch(log_txt_pattern, silent)
    print(model_to_times)
    n_models = len(model_to_times)
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Train Times', fontsize=25)
    ax = fig.add_subplot(111)
    plot_data = {}
    n_epochs = 0
    highest_time = 0
    random_colors = (MPL_COLORS * round(n_models/len(MPL_COLORS) + 0.5))[:n_models]
    # x_big_markers, y_big_markers = [], []
    for i, model_base in enumerate(model_to_times):
        try:
            best_epoch, best_wer = model_to_best_val[model_base]
        except KeyError:
            relevant_entries = [v for k, v in model_to_best_val.items() if k.split("(")[0].strip() in model_base]
            best_epoch, best_wer = max(relevant_entries, key=lambda x: x[1])
        x_axis = list(model_to_times[model_base].keys())
        y_axis = list(map(lambda x: x/60, list(model_to_times[model_base].values())))
        best_epoch_index = [ind for ind, x in enumerate(x_axis) if x == best_epoch][0]
        time_of_best_wer = [y for ind, y in enumerate(y_axis) if ind == best_epoch_index][0]
        # x_big_markers.append(best_epoch_index)
        # y_big_markers.append(time_of_best_wer)
        n_epochs = max(len(y_axis), n_epochs)
        highest_time = max(y_axis[-1], highest_time)
        ax.plot(x_axis, y_axis, label=model_base, marker="o", color=random_colors[i])
        ax.text(best_epoch, time_of_best_wer, best_wer)
        ax.plot([best_epoch], [time_of_best_wer], 'o', ms=14, color=random_colors[i])
        plot_data[model_base] = {
            "plot1": {
                "args": [x_axis, y_axis], 
                "kwargs": {"label": model_base, "marker": "o", "color": random_colors[i]}
            },
            "text": {
                "args": [best_epoch, time_of_best_wer, best_wer], "kwargs": {}
            },
            "plot2": {
                "args": [[best_epoch], [time_of_best_wer], 'o'], 
                "kwargs": {"ms": 14, "color": random_colors[i]}
            }
        }
    highest_time, lowest_time = 0, highest_time
    for identifier in plot_data:
        y_axis = plot_data[identifier]['plot1']['args'][1]
        if len(y_axis) == n_epochs:
            highest_time = max(y_axis[-1], highest_time)
            lowest_time = min(y_axis[-1], lowest_time)
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train time (hours)")
    if n_epochs > 50:
        zoom = 10.5
        x1, x2 = 73, 75.4
        y1, y2 = 74.5, 78.5
    elif n_epochs > 20:
        zoom = 9
        x1, x2 = 48.8, 50.2
        y1, y2 = 174.5, 184.5
    else:
        zoom = 11
        x1, x2 = 14.7, 15.1
        y1, y2 = 156, 163.5
    axins = zoomed_inset_axes(ax, zoom=zoom, loc="lower right", borderpad=6)
    # fix the number of ticks on the inset axes
    axins.yaxis.get_major_locator().set_params(nbins=7)
    axins.xaxis.get_major_locator().set_params(nbins=7)
    # sub region of the original image
    # x1, x2 = n_epochs-max(n_epochs//15, 1.5), n_epochs+0.2
    # y1, y2 = lowest_time-1, lowest_time + (highest_time-lowest_time)/2
    for identifier in plot_data:
        p = plot_data[identifier]
        x_axis, y_axis = p['plot1']['args']
        if not ((x1 <= x_axis[-1] <= x2) and (y1 <= y_axis[-1] <= y2)):
            p['plot1']['kwargs'].pop('label')
        axins.plot(*p['plot1']['args'], **p['plot1']['kwargs'])
        best_epoch = p['text']['args'][0]
        best_wer = p['text']['args'][1]
        if best_epoch >= x1 and best_wer <= y2:
            axins.text(*p['text']['args'], **p['text']['kwargs'])
            axins.plot(*p['plot2']['args'], **p['plot2']['kwargs'])
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.legend()

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.setp(ax.get_legend().get_texts(), fontsize='20') # for legend text
    fig.tight_layout()
    if out_path is None:
        plt.show()
        return
    elif out_path is False:
        out_path = "./train_times_per_epoch.png"
    print("Saving plot under:", os.path.abspath(out_path))
    plt.savefig(out_path)

def hours_to_wers_plot(
        paths: list, 
        train_csv_name="train-complete_segmented.csv"
    ):
    hours = {}
    for directory in paths:
        log = os.path.join(directory, "train_log.txt")
        if not os.path.isfile(log): 
            print(f"Ignoring {directory} since it does not contain a train_log.txt file.")
            continue
        model_name = os.path.basename(os.path.dirname(os.path.dirname(directory)))
        seed = os.path.basename(os.path.dirname(directory)).split("-")[0]
        identifier = f"{model_name} {seed}"
        if identifier in hours: continue
        train_csv = os.path.join(directory, train_csv_name)
        with open(log, 'r') as fr:
            n_epochs = [int(l.split("epoch:")[1].split(",")[0].strip()) for l in fr if l.startswith("epoch:")]
            if len(n_epochs) < 15: continue
            else: n_epochs = max(n_epochs)
        if n_epochs < 15:
            continue
        is_paced = ("subsampl" in directory)
        subsampling_n_epochs = None
        if is_paced:
            with open(os.path.join(directory, "hyperparams.yaml")) as fr:
                subsampling_n_epochs = [l for l in fr if l.startswith("subsampling_n_epochs:")]
            if len(subsampling_n_epochs) != 1:
                raise ValueError("Could not find the subsampling_n_epochs attribute even though a pacing function is used.")
            subsampling_n_epochs = int(subsampling_n_epochs[0].split()[-1].replace("\n", "").strip())
        hours[directory]= calculate_total_hours_seen(
            train_csv, n_epochs, 
            is_paced=is_paced,
            subsampling_n_epochs=subsampling_n_epochs
        )
    print(json.dumps(hours))


def _get_best_epoch(log_txt_pattern, silent=False):
    paths = _get_iterator(log_txt_pattern)
    model_to_best_val = {}
    for path in paths:
        try:
            epochs, _, _, valid_metrics_dict = _read_stats([path])
        except NoEpochsTrained as e:
            if silent:
                continue
            else:
                raise e
        model_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        seed = os.path.basename(os.path.dirname(path)).split("-")[0]
        model_name = f"{model_id} ({seed})"
        # model_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        best_valid_epoch, best_valid_wer = _find_best_epoch(epochs, valid_metrics_dict)
        model_to_best_val[model_name] = (int(best_valid_epoch), float(best_valid_wer))
    return model_to_best_val

def _find_best_epoch(epochs, valid_metrics_dict, metric="WER"):
    best_valid_index = 0
    for i in range(1, len(valid_metrics_dict[metric])):
        if valid_metrics_dict[metric][i][0] < valid_metrics_dict[metric][best_valid_index][0]:
            best_valid_index = i
    best_valid_wer = valid_metrics_dict[metric][best_valid_index][0]
    best_valid_epoch = [epochs[i][0] for i in range(len(epochs)) if i==best_valid_index][0]
    return int(best_valid_epoch), float(best_valid_wer)

def _get_iterator(log_txt_pattern):
    if isinstance(log_txt_pattern, list):
        iterator = []
        for item in log_txt_pattern:
            iterator += _get_iterator(item)
        return iterator
    if isinstance(log_txt_pattern, str) and not log_txt_pattern.endswith("log.txt"):
        log_txt_pattern = os.path.join(log_txt_pattern, "log.txt")
    if os.path.isfile(log_txt_pattern):
        iterator = [log_txt_pattern]
    elif "*" in log_txt_pattern:
        if not log_txt_pattern.endswith("log.txt"):
            log_txt_pattern = os.path.join(log_txt_pattern, "log.txt")
        iterator = glob.glob(log_txt_pattern)
        if len(iterator) == 0:
            raise Exception("Invalid log.txt pattern: {}".format(log_txt_pattern))
    else:
        raise Exception("Neither a valid log file not a pattern: {}".format(log_txt_pattern))
    return iterator

def _parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="*", help="E.g. './path/to/recipes/*/*/log.txt' or './path/to/log.txt'")
    parser.add_argument("--visualize", "-v", action="store_true", default=False, 
        help="If provided, we will also plot the train times per epoch for each model.")
    parser.add_argument("--out-plot-path", "-o", required=False, default=None,
        help="If provided, the output plot (assuming -v is also provided) will be saved there.")
    parser.add_argument("--silent", "-s", default=False, action="store_true",
        help="If provided, the program won't throw NoEpochsTrained errors.")
    parser.add_argument("--show-hours-per-model", "--hpm", dest="show_hours_per_model", 
        default=False, action="store_true", 
        help="If provided then we are going to show the hours that each model has seen during its\
        training. You also need to provide the `train_csv_name` argument in\
        case it's not the default value.")
    parser.add_argument("--train_csv_name", "--csv", default="train-complete_segmented.csv",
        help="What's the filename of the .csv file used for training the speechbrain model.")
    args = parser.parse_args()
    return args

def main(args):
    if args.visualize:
        plot_train_times(args.input, args.out_plot_path, args.silent)
    elif args.show_hours_per_model:
        if args.train_csv_name is None:
            raise argparse.ArgumentTypeError("You should provide the \
                `train_csv_name` argument in order to see the hours \
                    that each model has seen during training.")
        hours_to_wers_plot(args.input, args.train_csv_name)
    else:
        get_train_times(args.input, args.silent)

if __name__ == "__main__":
    args = _parse_args()
    main(args)