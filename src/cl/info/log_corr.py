import argparse
import os, glob
import ast
import numpy as np
import warnings
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
except ImportError:
    # warnings.warn("Could not import matplotlib. If you are planning to use the visualization functions then you need to install it.")
    pass

from scipy.stats import spearmanr
from tqdm import tqdm


class NoCurriculumOrderings(Exception): pass

def calc_corr(*paths, **kwargs):
    out_path = kwargs.pop('out_path', None)
    try:
        from scipy.stats import spearmanr
    except ImportError:
        raise ImportError("You need to install scipy to use this script. Cannot continue...")
    show_later = False
    fig = None
    for p in paths:
        if os.path.isdir(p):
            if "ascending" in p:
                current_paths = [os.path.join(p, "ascending_dict.log")]
            else:
                current_paths = glob.glob(os.path.join(p, 'curriculum_logs', '*.log'))
            if fig is None:
                fig = plt.figure(figsize=(16, 12))
            try:
                _calc_corr_single(*current_paths, **kwargs, show=False, out_path=out_path, allow_length_one=False)
            except NoCurriculumOrderings as ne:
                print(ne)
                print(f"Occurred on path {p}")
                continue
            except Exception as e:
                print(f"Error while processing path {p} and current_paths being {current_paths}")
                raise e
            show_later = True
        else:
            assert os.path.isfile(p), f"Could not locate path {p}"
    if not show_later:
        _calc_corr_single(*paths, **kwargs, out_path=out_path)
    else:
        fig.tight_layout()
        plt.legend()
        if out_path is not None:
            plt.savefig(out_path)
        else:
            plt.show()

def _calc_corr_single(*paths, **kwargs):
    curriculums = {}
    types = []
    if len(paths) == 0:
        raise NoCurriculumOrderings("Could not find any curriculum ordering logs.")
    if not kwargs.pop("allow_length_one", True) and len(paths) == 1:
        return
    pbar = tqdm(paths)
    for p in pbar:
        if "random" in p:
            raise Exception("Random methods are not supported yet because the \
                corresponding orderings are saved as indices in numpy format.\
                To fix that you should load the numpy indices and the train csv file \
                and then map the indices to the corresponding utterance ids.")
        if ("no_ga" in p) and ("tr" not in p):
            raise Exception("for no curriculum you need to read the train csv file\
                and return the utterance ids as they appear.")
        if "ascending" in p:
            epoch = 0
            type_index = -3
        else:
            try:
                epoch = int(p.split("epoch=")[1].split('.')[0])
            except IndexError:
                raise IndexError(f"Index out of bounds for path {p}.")
            type_index = -4
        type = p.split("/")[type_index]
        seed = p.split("/")[type_index+1].split("-")[0]
        pbar.set_description(f"Processing {type} ({seed=})")
        order = {}
        with open(p) as reader:
            for l in reader:
                l = l.replace('\n', '').replace('\t', ' ').strip()
                utt_id = l.split()[0]
                score = ast.literal_eval(' '.join(l.split()[1:]))
                order[utt_id] = score
        types.append(type)
        curriculums[epoch] = order.copy()
        # if len(curriculums) == 5:
        #     break
    identifier = f"{type} ({seed=})"
    corr_m = np.zeros((len(curriculums), len(curriculums)))
    selected_keys = list(curriculums.keys())
    print("Got selected_keys:", selected_keys)
    # sort based on epochs
    correlations = {k: v for k, v in sorted(curriculums.items(), key=lambda x: x[0])}
    corr_per_epoch = {}
    pbar = tqdm(zip(list(correlations.keys())[:-1], list(correlations.keys())[1:]), total=len(correlations)-1)
    for epoch1, epoch2 in pbar:
        pbar.set_description(f"Correlation for pair {epoch1}->{epoch2}")
        orders1 = correlations[epoch1]
        orders2 = correlations[epoch2]
        i_curriculum = list(sorted(orders1, key=lambda x: orders1[x]))
        j_curriculum = list(sorted(orders2, key=lambda x: orders2[x]))
        corr_m = spearmanr(i_curriculum, j_curriculum)[0]
        corr_per_epoch[f"{epoch1}->{epoch2}"] = corr_m
    if kwargs.pop('visualize', False) is True:
        plot_corr_per_epoch(corr_per_epoch, identifier, show=kwargs.pop('show', True), **kwargs)
    return corr_per_epoch
    

def plot_corr_per_epoch(correlations, identifier, show=True, **kwargs):
    out_path = kwargs.pop('out_path', None)
    if show:
        fig = plt.figure(figsize=(16, 12))
    plt.plot(list(range(len(correlations))), correlations.values(), label=identifier, **kwargs)
    plt.xticks(list(range(len(correlations))), list(correlations.keys()))
    plt.xlabel("Epochs")
    plt.ylabel("Spearman Correlation")
    plt.title("Correlation of curriculum orderings among pairs of epochs")
    if show:
        fig.tight_layout()
        plt.legend()
        if out_path is not None:
            plt.savefig(out_path)
        else:
            plt.show()


def _get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Either a path to the directory \
        that contains the curriculum logs (the orderings) or a sequence of paths \
        that point to .log files. We will try to find the correlation of the orderings.")
    parser.add_argument("--visualize", "-v", action="store_true", default=False,
        help="Whether to produce a line plot or not.")
    parser.add_argument("--out-plot", "-o", default=None, help="If --visualize is \
        provided then you can specify the path here. By default will just try to \
        show the plot.")
    return parser

def _parse_args(args=None):
    if args is None:
        parser = _get_parser()
        args = parser.parse_args()
    calc_corr(*args.paths, visualize=args.visualize, out_path=args.out_plot)


if __name__ == "__main__":
    _parse_args()