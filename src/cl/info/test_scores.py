import argparse
import os
import glob
from cl.info.statmd import _read_stats, DEFAULT_METRICS, AVAILABLE_METRICS, NoEpochsTrained

def main(res_dir, metrics=DEFAULT_METRICS):
    """ Prints out test statistics for all models inside a directory.
        Expects `res_dir` to have many subdirectories containing iterations.
        E.g. 
        results
        ├── crdnn_baseline
        │   └── 1001
        │   └── 1002
        ├── crdnn_medium
        │   └── 1001
        └── crdnn_op
            └── 1001

    """
    if isinstance(res_dir, list) and len(res_dir) > 1:
        if res_dir[0].endswith("log.txt"):
            log_paths = res_dir
        else:
            log_paths = []
            for p in res_dir:
                log_paths += glob.glob(os.path.join(p, "*", "log.txt"))
    elif isinstance(res_dir, str):
        assert os.path.isdir(res_dir), res_dir
        log_paths = glob.glob(os.path.join(res_dir, "*", "*", "log.txt"))
    elif isinstance(res_dir, list) and len(res_dir) == 1:
        log_paths = glob.glob(os.path.join(res_dir[0], "*", "*", "log.txt"))
        log_paths += glob.glob(os.path.join(res_dir[0], "*", "log.txt"))
    else:
        raise ValueError("Got invalid directory: {}.".format(res_dir))
    for log in log_paths:
        seed = os.path.dirname(log)
        model_name = os.path.basename(os.path.dirname(seed))
        seed = os.path.basename(seed)
        try:
            out = _read_stats([log], metrics, return_test_results=True)
        except NoEpochsTrained:
            print("Model: {} (seed={})\n\tNo epochs trained :(".format(model_name, seed))
            continue
        if len(out) == 5 and len(out[-1]) > 0:
            # We only provide one log so the length of the test_results var is always 1.
            test_stat = out[-1][-1]
            epochs = out[0]
            valid_metrics_dict = out[-2]
            if "WER" in valid_metrics_dict:
                best_valid_index = 0
                for i in range(len(valid_metrics_dict['WER'])):
                    if valid_metrics_dict['WER'][i][0] < valid_metrics_dict['WER'][best_valid_index][0]:
                        best_valid_index = i
            else:
                best_valid_index = -1
            best_valid_epoch = [epochs[i][0] for i in range(len(epochs)) if i==best_valid_index][0]
            s = "Model: {} (seed={})\n\tTest Loss: {}".format(model_name, seed, test_stat[0])
            for metric in metrics:
                s += ", Test {}: {}".format(metric.capitalize(), test_stat[1][metric])
            s += " ("
            for metric in metrics:
                best_valid_metric = valid_metrics_dict[metric][best_valid_index][0]
                s += "valid {}: {}, ".format(metric.lower(), best_valid_metric)
            s += " best epoch: {})".format(best_valid_epoch)
            print(s)
        else:
            epochs = out[0][-1]
            max_epoch = epochs[-1]
            valid_metrics_dict = out[-2]
            s = "Model: {} (seed={})\n\tCurrently on epoch {} with".format(model_name, seed, max_epoch)
            for metric in metrics:
                s+= " valid_{}={}, ".format(metric.lower(), valid_metrics_dict[metric][-1][0])
            print(s[:-2])

if __name__ == "__main__":
    metric_maps = {m[0].lower(): m for m in AVAILABLE_METRICS}
    reverse_maps = {v: k for k, v in metric_maps.items()}
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", "--res-dir", "-d", nargs="+", metavar="RESULTS_DIR", dest="res_dir", required=True, 
        help="Path(s) to the directory containing the results of your models."
    )
    parser.add_argument(
        "--metrics", "-m", choices=AVAILABLE_METRICS+list(metric_maps.keys()), 
        default=[], nargs="*",
        help="Expected metric based on which we will extract the statistics. "\
             "{} are the only ones currently allowed.".format(', '.join(
                 [str(k) +  " (" + str(v) + ")" for k, v in reverse_maps.items()]
            ))
    )
    # Also accept arguments as `--cer` or `--per`, `-p`
    for m in AVAILABLE_METRICS:
        parser.add_argument(
            "--{}".format(m.lower()), "-{}".format(m[0].lower()), 
            action="store_true", default=False,
            help="Use {} as the expected metric.".format(m)
        )
    args = parser.parse_args()
    # e.g. map 'w' to 'WER'
    metrics = [metric_maps.get(metric, metric) for metric in args.metrics]
    # In case the user provided e.g. --cer or -c instead of --metrics c
    metrics += [m for m in AVAILABLE_METRICS if getattr(args, m.lower()) is True]
    if len(metrics) == 0:
        metrics = DEFAULT_METRICS
    main(args.res_dir, metrics)
