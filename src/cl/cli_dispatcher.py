#!/m/teamwork/t40511_asr/p/curriculum-e2e/startover/sssb/bin/python
import argparse
import sys

from .info.visualize import violinplots_by3 as testset_boxplot_comparison, plot_logs_dispatcher
from .info.statmd import get_args, statmd, _add_parser_args
from .info.get_train_times import main as get_train_times
from .info.read_wer_test import read_wer_test, _get_parser as test_wer_parser
from .info.find_anomalies import _parse_args as test_anomalies, _get_parser as test_anomalies_parser
from .info.plot_metric_per_length import _get_parser as parse_test_wrt_durations, _parse_args as plot_test_per_length
# from .info.log_corr import _get_parser as _get_corr_parser, _parse_args as _parse_corr_args
from .info.testset_correlation import _get_parser as _get_test_corr_parser, _parse_args as _parse_corr_args
from .utils.sb_wer_to_kaldi import _get_parser as sb_wer_parser, _parse_args as sb_wer_to_kaldi
from .utils.kaldi_format_calc_wer import _get_parser as kaldi_calc_wer_parser, _parse_args as kaldi_calc_wer
from .utils.remove_repetitions import _get_parser as remove_reps_parser, _parse_args as remove_reps
from .utils.convert_txt_to_trn import _get_parser as convert_to_trn_parser, _parse_args as convert_to_trn
from .utils.significance_tests import _get_parser as get_sign_test_parser, _parse_args as parse_sign_test_args

def dispatch():
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
    # ======== TEST STATS W.R.T. DURATIONS =======
    # ============================================
    test_and_dur_parser = subparsers.add_parser(
        "test_and_duration", aliases=["td"], help = "Print wer scores on the test set w.r.t. the durations of the utterances."
    )
    test_and_dur_parser = parse_test_wrt_durations(test_and_dur_parser)
    test_and_dur_parser.set_defaults(func=plot_test_per_length)

    # ============================================
    # ============ PRINT TEST STATS ==============
    # ============================================
    testwer_parser = subparsers.add_parser(
        "testwer", aliases=["tw"], help = "Print wer scores on the test set with the ability to make a barplot."
    )
    testwer_parser = test_wer_parser(testwer_parser)
    testwer_parser = remove_reps_parser(testwer_parser)
    testwer_parser = sb_wer_parser(testwer_parser)
    testwer_parser = kaldi_calc_wer_parser(testwer_parser)
    testwer_parser = test_anomalies_parser(testwer_parser)
    testwer_parser.add_argument(
        "--find-anomalies", "-fa", action="store_true", default=False,
        help="If provided then we are going to try to find anomalies in\
            the provided wer_test*.txt or cer_test*.txt files and compare them\
            if the --compare option is also provided."
    )
    testwer_parser.add_argument(
        "--remove-repetitions", "-rr", action="store_true", default=False,
        help="Remove repetitions from .kaldi files."
    )
    testwer_parser.add_argument(
        "--to-kaldi-text", "-kt", action="store_true", default=False,
        help="If provided then the input {wer,cer}_test*.txt files\
            will be converted to kaldi format (utt_id word1 word2 ...)."
    )
    testwer_parser.add_argument(
        "--noreps-to-wer-txt", "-nr", action="store_true", default=False,
        help="If provided then a {wer,cer}_test_noreps.txt file\
            will be created based on the input _noreps.kaldi files.\
            This will contain the typical wer_test.txt stats of speechbrain"
    )
    def testwer_func(args):
        if args.remove_repetitions:
            return remove_reps(args)
        if args.find_anomalies:
            return test_anomalies(args)
        if args.to_kaldi_text:
            return sb_wer_to_kaldi(args)
        if args.noreps_to_wer_txt:
            return kaldi_calc_wer(args)
        prefix = "cer" if args.cer else "wer"
        if args.wer_suffix is None:
            if args.vad:
                args.wer_file = f"{prefix}_test_vadded.txt"
            elif args.forced_segmented:
                args.wer_file = f"{prefix}_test_forced_segmented.txt"
            else:
                args.wer_file = f"{prefix}_test.txt"
        else:
            args.wer_file = f"{prefix}_test{args.wer_suffix}.txt"
        read_wer_test(args.exps, args.wer_file, args.out_path, name_mappings_file=args.model_name_mappings)
    testwer_parser.set_defaults(func=testwer_func)

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
    log_plot_parser.add_argument(
        "--plot-valid-results", "-v", default=False, action="store_true",
        help="If provided then the validation set's performance progress will be plotted for each model."
    )
    log_plot_parser.add_argument(
        "--barplot", "-b", default=False, action="store_true",
        help="If provided then instead of line plots we are going to plot\
            grouped barplots (per epoch)."
    )
    log_plot_parser.add_argument("--model-name-mappings", "-nm", default=None, 
        help="Path to a .json file containing a dictionary with the keys\
              `curriculum_mappings`, `transfer_mappings` and `subset_mappings`.\
              This remains to be documented.")
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
    train_times_parser.add_argument("--silent", "-s", default=False, action="store_true",
        help="If provided, the program won't throw NoEpochsTrained errors.")
    train_times_parser.add_argument("--show-hours-per-model", "--hpm", dest="show_hours_per_model", 
        default=False, action="store_true", 
        help="If provided then we are going to show the hours that each model has seen during its\
        training. You also need to provide the `train_csv_name` argument in\
        case it's not the default value."
    )
    train_times_parser.add_argument("--train_csv_name", "--csv", default="train-complete_segmented.csv",
        help="What's the filename of the .csv file used for training the speechbrain model."
    )
    train_times_parser.set_defaults(func=get_train_times)

    # ============================================
    # =============== BOXPLOTS ===================
    # ============================================
    corr_parser = subparsers.add_parser(
        "correlations", aliases=["c"], help = "Calculate correlations model pairs.\
            The output will be a horizontal barplot where each bar corresponds to \
            the correlation of a pair. You should provide at least 1 pair of models \
            containing a wer_test{suffix}.txt file each."
    )
    corr_parser = _get_test_corr_parser(corr_parser)
    corr_parser.set_defaults(func=_parse_corr_args)

    # ============================================
    # =========== CONVERSIONS (v7.0) =============
    # ============================================
    conv_parser = subparsers.add_parser(
        "convert", aliases=["cv"], help = "Convert log files to specified formats.\
            E.g. the -trn option converts wer_test*.txt files to .trn files.\
            I.e. w1 w2 w3 ... w_n (utterance_id)"
    )
    conv_parser.add_argument("--to-trn-format", "-trn", default=False, action="store_true",
        help="If provided, will convert the wer_test*.txt to the trn format."
    )
    def parse_conv_args(args):
        # if args.to_trn_format:
        #     return convert_to_trn(args)
        return convert_to_trn(args)
    conv_parser = convert_to_trn_parser(conv_parser)
    conv_parser.set_defaults(func=parse_conv_args)

    # ============================================
    # =========== SIGNIFICANCE TESTS =============
    # ============================================
    sign_test_parser = subparsers.add_parser(
        "significance-tests", aliases=["st", "sign"], help = "Get the .unigram file\
            required to perform significance tests."
    )
    sign_test_parser = get_sign_test_parser(sign_test_parser)
    sign_test_parser.set_defaults(func=parse_sign_test_args)

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    # args = parser.parse_args()
    return parser