import argparse
import os
import re
import warnings
from cl.info.find_anomalies import read_single

DEFAULT_METRIC_THRESHOLD = 0.0


def convert_to_kaldi_text(
      test_results_file: str, 
      output_kaldi_text_path: str,
      out_ground_truth_kaldi_path: str = None,
      threshold: float = DEFAULT_METRIC_THRESHOLD, 
      overwrite_if_exists: bool = True,
    ):
    contents = read_single(test_results_file, threshold, print_stats=False)
    if os.path.isfile(output_kaldi_text_path) and not overwrite_if_exists:
        warnings.warn(f"File {output_kaldi_text_path} already exists. We won't overwrite it.")
        return
    print("Writing:", output_kaldi_text_path, "and ground truth:", out_ground_truth_kaldi_path)
    with open(output_kaldi_text_path, 'w') as fw:
        if out_ground_truth_kaldi_path is not None:
            fw_truth = open(out_ground_truth_kaldi_path, 'w')
        for utt_id, score, ins, dels, subs, truth, preds in contents:
            text = "{} {}\n".format(
                utt_id.strip().replace('\n', ''), 
                re.sub("\s+", " ", preds.strip().replace('\n', ''))
            )
            fw.write(text)
            if out_ground_truth_kaldi_path is not None:
                truth_text = "{} {}\n".format(
                    utt_id.strip().replace('\n', ''), 
                    re.sub("\s+", " ", truth.strip().replace('\n', ''))
                )
                fw_truth.write(truth_text)


def _get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("sb_test_paths", nargs="*", 
        help="Path or paths to wer_test*.txt files."
    )
    parser.add_argument("--out-kaldi-path", "-okp", nargs="*", default=[],
        help="Path where the output file will be saved in kaldi's text format (utt_id w1 w2...).")
    return parser

def _parse_args(args=None):
    if args is None:
        parser = _get_parser()
        args = parser.parse_args()
    else:
        args.sb_test_paths += getattr(args, "exps", [])
    if len(args.out_kaldi_path) == 0:
        kaldi_filename = lambda p: os.path.join(os.path.dirname(p), os.path.basename(p).split(".")[0]+".kaldi")
        args.out_kaldi_path = [kaldi_filename(p) for p in args.sb_test_paths]
    kaldi_truth_filename = lambda p: os.path.join(os.path.dirname(p), os.path.basename(p).split(".")[0]+"_truth.kaldi")
    out_ground_truth_kaldi_path = [kaldi_truth_filename(p) for p in args.sb_test_paths]
    assert len(args.sb_test_paths) == len(args.out_kaldi_path), f"{len(args.sb_test_paths)=}  {len(args.out_kaldi_path)=}"
    assert len(args.sb_test_paths) > 0, f"Length of input and output paths is {len(args.sb_test_paths)}."
    for in_path, out_path, truth_out_path in zip(
          args.sb_test_paths, 
          args.out_kaldi_path,
          out_ground_truth_kaldi_path
        ):
        convert_to_kaldi_text(in_path, out_path, truth_out_path, DEFAULT_METRIC_THRESHOLD)

if __name__ == '__main__':
    _parse_args()