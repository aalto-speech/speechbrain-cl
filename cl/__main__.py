#!/usr/bin/python
""" CLI tool for the library.

Authors
 * Georgios Karakasidis 2022
"""
import argparse

from .info.visualize import testset_boxplot_comparison, plot_logs_dispatcher
from .info.statmd import get_args, statmd, _add_parser_args
from .info.get_train_times import main as get_train_times

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers()

# ============================================
# =============== BOXPLOTS ===================
# ============================================
wer_bp_parser = subparsers.add_parser(
    "boxplot", aliases=["b"], help = "Produce violin plots for\
        different wer_test.txt files. Each file will corresponding to \
        three violin plots (insertions, deletions and substitutions).\
        Usage: python -m cl b <path-to-wer-txt1> [<path-to-wer-txt2> ...]"
)
wer_bp_parser.add_argument(
    "--wer-paths", "-p", nargs="+", metavar="PATHS", required=True,
    help="List of paths to wer_test.txt files. "
)
wer_bp_parser.add_argument(
    "--out-path", "-o", default=None,
    help="Path where to save the output plot. If not provided, then the figure will\
        be shown (plt.show()) without being saved somewhere."
)
wer_bp_parser.add_argument(
    "--silent-ignore", "-s", action="store_true", default=False,
    help="If provided then we are going to silently ignore the paths\
        to the wer_test.txt files that don't exist."
)
wer_bp_parser.set_defaults(func=testset_boxplot_comparison)

# ============================================
# ======= PRINT TRAIN/DEV/TEST STATS =========
# ============================================
statmd_parser = subparsers.add_parser(
    "statmd", aliases=["s", "st"], help = "Prints a list of train/dev losses and metric\
        values for the files provided. At the end of the list, it also prints the \
        performance on the test set (if the test set has been decoded)."
)
statmd_parser = _add_parser_args(statmd_parser)
def statmd_func(args):
    paths, metrics, output_path, no_show_losses = get_args(statmd_parser, args=args, add_args=False)
    statmd(paths, list(dict.fromkeys(metrics)), output_path, not no_show_losses) 
statmd_parser.set_defaults(func=statmd_func)


# ============================================
# =============== LOSS PLOTS =================
# ============================================
log_plot_parser = subparsers.add_parser(
    "logplot", aliases=['l'], help="Produces some plots from the log.txt\
        files produced from training. These plots will contain information\
        about how the train/valid losses and train/valid WER/CERs evolve\
        over the training epochs."
)
log_plot_parser.add_argument(
    "--print-seed", "-ps", default=False, 
    help="Whether to also include the seed information of the experiment's\
        directories."
)
log_plot_parser = _add_parser_args(log_plot_parser)
log_plot_parser.set_defaults(func=plot_logs_dispatcher)

# ============================================
# =============== TRAIN TIMES ================
# ============================================
train_times_parser = subparsers.add_parser(
    "traintimes", aliases=['t', 'times'], help="Simple line plots\
        denoting the training times (for each epoch) that each \
        model requires"
)
train_times_parser.add_argument("input", nargs="*",
    help="E.g. './path/to/recipes/*/*/log.txt' or './path/to/log.txt'"
)
train_times_parser.add_argument("--visualize", "-v", action="store_true", 
    default=False, help="If provided, we will also plot the train \
        times per epoch for each model."
)
train_times_parser.add_argument("--out-plot-path", "-o", required=False, 
    default=None, help="If provided, the output plot (assuming -v is \
        also provided) will be saved there."
)
train_times_parser.set_defaults(func=get_train_times)

args = parser.parse_args()
args.func(args)