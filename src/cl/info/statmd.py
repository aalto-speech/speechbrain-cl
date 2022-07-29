#!/usr/bin/python
import os
import argparse
import glob
try:
    from cl.info.globals import DEFAULT_METRICS, AVAILABLE_METRICS
except ImportError:
    DEFAULT_METRICS = ["WER", "CER"]
    AVAILABLE_METRICS = ["WER", "CER", "PER"]

class NoEpochsTrained(Exception): pass

_metric_maps = {m[0].lower(): m for m in AVAILABLE_METRICS}

def _read_stats(paths, metrics=DEFAULT_METRICS, return_test_results=False):
    assert isinstance(metrics, list)
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError("Invalid path to 'log.txt': {}.".format(p))
    contents = []
    test_results = []
    for path in paths:
        with open(path, 'r') as f:
            content = f.readlines()
            test_results += [_check_test_stats(c, metrics) for c in content if "test loss:" in c]
            content = [c for c in content if ("epoch:" in c) and ("train loss:" in c)]
            contents.append(content)
    def get_epoch_numbers():
        return [[c.split("epoch:")[1].split(",")[0].strip() for content in content_tuple for c in content if "epoch:" in c] for content_tuple in zip(contents)]
    epochs = get_epoch_numbers()
    try:
        max_epochs = [int(e[-1]) for e in epochs]
    except IndexError:
        raise NoEpochsTrained("A provided path did not include a log with >= 1 epochs trained.")
    # If the log.txt file has been restarted many times. E.g. epoch 1 appears 3 times. 
    # then we only care about the most recent result
    if any(len(epoch) != max_epoch_num for epoch, max_epoch_num in zip(epochs, max_epochs)):
        assert len(max_epochs) == len(contents)
        contents = [c[-max_epoch_num:] for c, max_epoch_num in zip(contents, max_epochs)]
        epochs = get_epoch_numbers()
        assert len(set([len(e) for e in epochs])) == 1, list(set([len(e) for e in epochs]))
    epochs = list(zip(*epochs))
    if len(epochs) == 0:
        raise ValueError("Something went wrong. Are you sure you have provided a valid log.txt file?")
    # train_losses = [float(c.split("train loss:")[1].split(" - ")[0].strip()) for c in content]
    train_losses = [[float(c.split("train loss:")[1].split(" - ")[0].strip()) for content in content_tuple for c in content if "epoch:" in c] for content_tuple in zip(contents)] 
    train_losses = list(zip(*train_losses))
    # valid_losses = [float(c.split("valid loss:")[1].split(",")[0].strip()) for c in content]
    valid_losses = [[float(c.split("valid loss:")[1].split(",")[0].strip()) for content in content_tuple for c in content if "epoch:" in c] for content_tuple in zip(contents)] 
    valid_losses = list(zip(*valid_losses))
    # valid_metrics = [float(c.split("valid metric:")[1].strip()) for c in content]
    valid_metrics_dict = {}
    for metric in metrics:
        try:
            valid_metrics = [[float(c.split("valid {}:".format(metric))[1].split(",")[0].strip()) for content in content_tuple for c in content if "epoch:" in c] for content_tuple in zip(contents)] 
        except IndexError:
            raise ValueError("Could not find a valid entry with the {} metric. Maybe you are using another metric?".format(metric))
        # valid_metrics = list(zip(*valid_metrics))
        valid_metrics_dict[metric] = list(zip(*valid_metrics))
        del valid_metrics
    assert all([len(epochs) == len(train_losses) == len(valid_losses) == len(valid_metric) for valid_metric in valid_metrics_dict.values()])
    assert all([len(epochs[0]) == len(train_losses[0]) == len(valid_losses[0]) == len(valid_metric[0]) for valid_metric in valid_metrics_dict.values()])
    out = epochs, train_losses, valid_losses, valid_metrics_dict
    if return_test_results is True:
        out += (test_results,)
    return out

def _check_test_stats(test_result, metrics):
    test_loss = float(test_result.split("test loss: ")[1].split(",")[0].strip())
    test_metrics_dict = {}
    for metric in metrics:
        try:
            test_metrics_dict[metric] = float(test_result.split("test {}:".format(metric))[1].split(",")[0].strip())
        except IndexError:
            raise ValueError("Could not find a valid entry with the {} metric. Maybe you are using another metric?".format(metric))
    return test_loss, test_metrics_dict    

def statmd(paths, metrics=DEFAULT_METRICS, output_path=None, show_losses=True):
    out = _read_stats(paths, metrics, True)
    epochs, train_losses, valid_losses, valid_metrics_dict = out[:4]
    # Start building the string
    s = ""
    for i, (epochs, tls, vls) in enumerate(zip(epochs, train_losses, valid_losses)):
        # print("| {}  | {}  | {}  | {}  |".format(epoch, tl, vl, vm))
        s += "| {}  |".format(epochs[0])
        for j, (tl, vl) in enumerate(zip(tls, vls)):
            if show_losses:
                s += " {}  | {}  |".format(tl, vl)
            for metric in metrics:
                s += " {}  |".format(valid_metrics_dict[metric][i][j])
        s += "\n"
    if len(out) == 5:
        test_results = out[4]
        for i, res in enumerate(test_results):
            if len(res) > 0:
                s += "Model {}: test_loss={}, ".format(i, test_results[i][0])
                for metric in metrics:
                    s+= ", test_{}={}".format(metric.lower(), test_results[i][1][metric])
                s += "\n"
    if output_path is not None and os.path.isdir(os.path.dirname(output_path)):
        with open(output_path, 'w', encoding='utf-8') as fw:
            fw.writelines(s[:-1])
        return
    print(s[:-1])
    
def _add_parser_args(parser=None):
    reverse_maps = {v: k for k, v in _metric_maps.items()}
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths", metavar="PATHS", nargs="+", 
        help="Path(s) to log.txt files."
    )
    parser.add_argument(
        "--metrics", "-m", choices=AVAILABLE_METRICS+list(_metric_maps.keys()), 
        default=[], nargs="*",
        help="Expected metric based on which we will extract the statistics. "\
             "{} are the only ones currently allowed.".format(', '.join(
                 [str(k) +  " (" + str(v) + ")" for k, v in reverse_maps.items()]
            ))
    )
    parser.add_argument(
        "--no-show-losses", "-nl", action="store_true", default=False,
        help="If provided then we will not print the train/valid losses."
    )
    parser.add_argument(
        "-o", "--output-path", default=None, required=False, 
        help="If a valid path is provided then the output will be saved to the corresponding file. "\
        "Otherwise, the output will be simply printed in the console."
    )
    # Also accept arguments as `--cer` or `--per`, `-p`
    for m in AVAILABLE_METRICS:
        parser.add_argument(
            "--{}".format(m.lower()), "-{}".format(m[0].lower()), 
            action="store_true", default=False,
            help="Use {} as the expected metric.".format(m)
        )
    return parser

def _read_args(args):
    # e.g. map 'w' to 'WER'
    metrics = [_metric_maps.get(metric, metric) for metric in args.metrics]
    # In case the user provided e.g. --cer or -c instead of --metrics c
    metrics += [m for m in AVAILABLE_METRICS if getattr(args, m.lower()) is True]
    if len(metrics) == 0:
        metrics = DEFAULT_METRICS
    # Make sure the output_path looks like a path with at least on /
    output_path = args.output_path 
    if output_path is not None:
        output_path = output_path if "/" in output_path else "./" + output_path
    return args.paths, metrics, output_path, args.no_show_losses

def get_args(parser=None, args=None, add_args=True):
    if add_args:
        parser = _add_parser_args(parser)
    if args is None:
        args = parser.parse_args()
    paths, metrics, output_path, no_show_losses = _read_args(args)
    for i, p in enumerate(paths):
        if "*" in p:
            if not p.endswith("log.txt"):
                p = os.path.join(p, "log.txt")
            paths += glob.glob(p)
            continue
        if os.path.isdir(p):
            paths[i] = os.path.join(p, 'log.txt')
    return paths, metrics, output_path, no_show_losses


if __name__ == "__main__":
    paths, metrics, output_path, no_show_losses = get_args()
    statmd(paths, list(dict.fromkeys(metrics)), output_path, not no_show_losses)