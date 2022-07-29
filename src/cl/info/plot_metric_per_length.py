import argparse
import os
import csv
import warnings
try:
    import matplotlib.pyplot as plt
except ImportError:
    # warnings.warn("Could not import matplotlib. If you are planning to use the visualization functions then you need to install it.")
    pass
from .find_anomalies import separate_test_entry, FIRST_EXAMPLE_LINE, SEPARATOR


UTT_ID_CSV_NAME = 'ID'  # Column name of the utterance identifier in the (test) .csv file.
DUR_CSV_NAME = 'duration'  # similar as above
TEXT_CSV_NAME = 'wrd'  # column under which we can find the transcription
UPPER_SCORE_LIMIT = 400

def utterance_map(res_path, csv_path):
    assert os.path.exists(res_path), "Could not locate " + res_path
    assert os.path.exists(csv_path), "Could not locate " + csv_path
    is_wer = False if res_path.endswith("cer_test.txt") else True
    separate_func = lambda x: separate_test_entry(x, is_wer)
    with open(res_path, 'r', encoding='utf-8') as fr:
        # Before the FIRST_EXAMPLE_LINE line, we have general information 
        # and instructions on how to read the file so we don't care.
        lines = fr.readlines()[FIRST_EXAMPLE_LINE:]
        lines = "".join(lines).split(SEPARATOR)
        # Keep only the utterance id and the score
        utt2score = {e[0]: float(e[1]) for e in map(separate_func, lines)}
    # Durations: Match each utterance id to a duration.
    utt2stats = {}
    with open(csv_path, 'r', encoding='utf-8') as fr:
        reader = csv.DictReader(fr)
        for row in reader:
            # Slow but ok
            utt_id = row[UTT_ID_CSV_NAME]
            score = utt2score[utt_id]
            # Reject outliers
            if score > UPPER_SCORE_LIMIT:
                continue
            utt2stats[utt_id] = (score, float(row[DUR_CSV_NAME]), row[TEXT_CSV_NAME])
    # utt2stats example: {05013lingsoft41373: (59.63, 10.1, "this is a test"), ...}
    # utt2stats = {utt_id: (SCORE, DURATION, TEXT)}
    return utt2stats

def plot_score_to_dur(utt2stats, title=None, out="tmp.eps"):
    # Sort utterances based on their duration
    utt2stats = {utt_id: v for utt_id, v in sorted(utt2stats.items(), key=lambda v: v[1][1])}
    # x-axis contains the durations
    x_axis = list(map(lambda x: utt2stats[x][1], utt2stats))
    # y-axis contains the scores
    y_axis = list(map(lambda x: utt2stats[x][0], utt2stats))

    plt.figure(figsize=(14, 10))
    if title is not None:
        plt.title(title)
    plt.subplot(2, 1, 1)
    plt.scatter(x_axis, y_axis)
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Score")
    plt.subplot(2, 1, 2)
    # Sorted based on the number of words apeparing int he transcript
    utt2stats_new = {utt_id: (v[0], len(v[-1].split())) for utt_id, v in sorted(utt2stats.items(), key=lambda v: len(v[1][-1].split()))}
    # x-axis contains the length
    x_axis = list(map(lambda x: utt2stats_new[x][1], utt2stats_new))
    # y-axis contains the scores
    y_axis = list(map(lambda x: utt2stats_new[x][0], utt2stats_new))
    plt.scatter(x_axis, y_axis)
    plt.xlabel("Utterance Length (#words)")
    plt.ylabel("Score")
    plt.savefig(out, format="eps")

def plot_per_dataset(utt2stats):
    lingsoft = {utt: v for utt, v in utt2stats.items() if "lingsoft" in utt}
    spoken = {utt: v for utt, v in utt2stats.items() if "spoken" in utt}
    tutkimustie = {utt: v for utt, v in utt2stats.items() if "tutkimustie" in utt}
    plot_score_to_dur(lingsoft, 'Lingsoft Dataset', 'lingsoft.png')
    plot_score_to_dur(spoken, 'Spoken Dataset', 'spoken.png')
    plot_score_to_dur(tutkimustie, 'Tutkimustie Dataset', 'tutkimustie.png')

def _get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", help="Path to an experiment directory\
        containing the wer_test*.csv and cer_test*.csv files.")
    parser.add_argument("--suffix", "-s", default="", help="Suffix for the \
        wer_test*.txt and cer_test*.txt. E.g. if suffix='_segmented' then\
        we will look for the files wer_test_segmented.txt and cer_test_segmetned.txt.")
    parser.add_argument("--test-csv", "-t", default=None, help="Path to the test*.csv\
        file which is used for testing. If not provided and --suffix is provided \
        then the test file we are going to look for is going to be test*suffix*.csv")
    parser.add_argument("--cer", "-c", default=False, action="store_true", help="Whether \
        to check the cer_test*.txt file instead of the wer_test*.txt one.")
    return parser

def _parse_args(args=None):
    if args is None:
        parser = _get_parser()
        args = parser.parse_args()
    test_csv = args.test_csv
    suffix = args.suffix
    assert os.path.isdir(args.exp_dir), f"{args.exp_dir} does not exist."
    if test_csv is None:
        test_csv =  os.path.join(args.exp_dir, f"test{suffix}.csv")
    if args.cer:
        test_metric = os.path.join(args.exp_dir, f"cer_test{suffix}.txt")
    else:
        test_metric = os.path.join(args.exp_dir, f"wer_test{suffix}.txt")
    for f in [test_csv, test_metric]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"File {f} could not be found.")
    
    utt2stats = utterance_map(res_path=test_metric, csv_path=test_csv)
    plot_score_to_dur(utt2stats)
    # plot_per_dataset(utt2stats)

if __name__ == "__main__":
    _parse_args()