import argparse
import os
import glob
from collections import Counter
import matplotlib.pyplot as plt
from cl.info.globals import MPL_COLORS
from .statmd import _read_stats, NoEpochsTrained

def get_train_times(log_txt_pattern, silent=False):
    model_to_times = {}
    seeds = {}
    iterator = _get_iterator(log_txt_pattern)
    for path in iterator:
        model_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        seed = os.path.basename(os.path.dirname(path)).split("-")[0]
        identifier = f"{model_id} ({seed})"
        # identifier = os.path.dirname(os.path.dirname(path))
        with open(path, 'r') as f:
            epoch, mins = None, None
            minutes_per_epoch = {}
            for line in f:
                if "Going into epoch" in line:
                    epoch = int(line.split()[-1].replace("\n", "").strip())
                    continue
                if "Currently training for" in line:
                    mins = float(line.split("for ")[-1].split(" minutes")[0].strip())
                    if mins <= 0 or mins in minutes_per_epoch.values():
                        continue
                    assert epoch is not None, "Model id: {}, Epoch: {}, mins: {}".format(identifier, epoch, mins)
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
                print(f"{minutes_per_epoch=}\n{model_id=}, seeds={d['seeds']}, counts={n_epochs_per_run}\n===================\n")
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
    n_models = len(model_to_times)
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Train Times', fontsize=25)
    random_colors = (MPL_COLORS * round(n_models/len(MPL_COLORS) + 0.5))[:n_models]
    # x_big_markers, y_big_markers = [], []
    for i, model_base in enumerate(model_to_times):
        try:
            best_epoch, best_wer = model_to_best_val[model_base]
        except KeyError:
            relevant_entries = [v for k, v in model_to_best_val.items() if k.split("(")[0].strip() in model_base]
            best_epoch, best_wer = max(relevant_entries, key=lambda x: x[1])
        x_axis = list(model_to_times[model_base].keys())
        vals_to_hours = map(lambda x: x/60, list(model_to_times[model_base].values()))
        y_axis = list(vals_to_hours)
        best_epoch_index = [ind for ind, x in enumerate(x_axis) if x == best_epoch][0]
        time_of_best_wer = [y for ind, y in enumerate(y_axis) if ind == best_epoch_index][0]
        # x_big_markers.append(best_epoch_index)
        # y_big_markers.append(time_of_best_wer)
        plt.plot(x_axis, y_axis, label=model_base, marker="o", color=random_colors[i])
        plt.text(best_epoch, time_of_best_wer, best_wer)
        plt.plot([best_epoch], [time_of_best_wer], 'o', ms=14, color=random_colors[i])
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Train time (hours)")
    if out_path is None:
        plt.show()
        return
    elif out_path is False:
        out_path = "./train_times_per_epoch.png"
    print("Saving plot under:", os.path.abspath(out_path))
    plt.savefig(out_path)

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
    args = parser.parse_args()
    return args

def main(args):
    if args.visualize:
        plot_train_times(args.input, args.out_plot_path, args.silent)
    else:
        get_train_times(args.input, args.silent)

if __name__ == "__main__":
    args = _parse_args()
    main(args)