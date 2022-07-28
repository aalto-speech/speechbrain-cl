import argparse
import os
# import time
from tqdm import tqdm
from speechbrain.utils.edit_distance import wer_details_by_utterance, wer_summary
from speechbrain.dataio.wer import print_wer_summary, print_alignments
from cl.utils.remove_repetitions import read_texts

def calc_wer(pred_file, ground_truth_file, save_path=None, is_cer=False):
    if is_cer:
        post = lambda sent: list(sent)
    else:
        post = lambda sent: sent
    ref_dict = {utt_id: post(text) for utt_id, text in read_texts(ground_truth_file)}
    hyp_dict = {utt_id: post(text) for utt_id, text in read_texts(pred_file)}
    wer_list = wer_details_by_utterance(ref_dict, hyp_dict, compute_alignments=True)
    summary = wer_summary(wer_list)
    if save_path is None:
        print_wer_summary(summary)
        print_alignments(wer_list)
    else:
        with open(save_path, 'w') as fw:
            print_wer_summary(summary, fw)
            print_alignments(wer_list, fw)
    return summary

def kaldi_wer_from_files(pred_files, ground_truth_files, is_cer=False):
    pbar = tqdm(zip(pred_files, ground_truth_files), total=len(pred_files))
    for p, gt in pbar:
        assert os.path.isfile(p), p
        assert os.path.isfile(gt), gt
        save_file = os.path.join(
            os.path.dirname(p),
            os.path.basename(p).split(".")[0] + ".txt"
        )
        summary = calc_wer(p, gt, save_file, is_cer)
        pbar.set_description(f"WER={summary['WER']}")
        # time.sleep(0.4)


def _get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("exps", nargs="*", 
            help="Path or paths to directories with {wer,cer}_test*.txt files."
        )
    parser.add_argument("--is-cer", action="store_true", default=False)
    return parser

def _parse_args(args=None):
    if args is None:
        parser = _get_parser()
        args = parser.parse_args()
    prefix = "cer" if args.is_cer else "wer"
    finished = lambda d: os.path.exists(os.path.join(d, "wer_test.txt"))
    preds = [os.path.join(p, prefix+"_test_noreps.kaldi") for p in args.exps if finished(p)]
    truths = [os.path.join(p, prefix+"_test_truth_noreps.kaldi") for p in args.exps if finished(p)]
    kaldi_wer_from_files(preds, truths, args.is_cer)
    

if __name__ == '__main__':
    _parse_args()