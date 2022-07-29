import argparse
import os
import pandas as pd
import warnings
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    # warnings.warn("Could not import matplotlib. If you are planning to use the visualization functions then you need to install it.")
    pass

from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import json
from cl.info.globals import map_name
from cl.info.globals import MPL_COLORS
from cl.info.find_anomalies import SEPARATOR, FIRST_EXAMPLE_LINE
# sns.set()


def testset_corr(
    exp_dirs, 
    wer_test_file="wer_test.txt", 
    out_path=None, 
    name_mappings_file=None,
    corr_method="pearson",
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
    df_dicts = []
    wer_files = sorted(wer_files)  # sort based on the name of the files
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
        model_name = map_name(identifier, name_mappings).replace("CL: ", "") + identifier.split()[-2]
        df_dict = pd.DataFrame({'utterance_id': list(lines.keys()), model_name: list(lines.values())})
        df_dict.set_index('utterance_id')
        df_dicts.append(df_dict)
    df = df_dicts[0]
    for df_dict in df_dicts[1:]:
        df = pd.merge(df, df_dict, how='inner', on='utterance_id')
    assert all(len(df) == len(df_dict) for df_dict in df_dicts), f"{df.head()}\n{len(df)}\n{len(df_dicts[0])=}"
    corr = df.corr(method=corr_method)
    fig = plt.figure(figsize=(23, 20))
    ax = sns.heatmap(corr, annot=True, cmap='Blues')

    ax.set_xlabel('Correlation')
    
    # Add Plot Title
    plt.title(f"{corr_method.capitalize()} Correlation Between Model Pairs", fontsize=22)

    fig.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
        print(f"Model saved under: {out_path}")
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