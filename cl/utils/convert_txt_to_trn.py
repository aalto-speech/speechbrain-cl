import argparse
import re, os, glob
from cl.info.find_anomalies import FIRST_EXAMPLE_LINE, SEPARATOR

def convert_to_trn(path: str, out_dir: str = None):
    # `path` is a valid path to a wer_test*.txt file
    # as produced by speechbrain.
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not locate {path}. Aborting...")
    with open(path, 'r', encoding='utf-8') as fr:
        # Before the FIRST_EXAMPLE_LINE line, we have general information 
        # and instructions on how to read the file so we don't care.
        lines = fr.readlines()[FIRST_EXAMPLE_LINE:]
        lines = "".join(lines).split(SEPARATOR)
    # `references` should be in trn format: word1 word2 ... wordN (utterance_id)
    references = []
    hypotheses = []
    for l in lines:
        utt_id = l.split(",")[0].replace("\n", "")
        l = [line for line in l.split("\n") if len(line.strip()) != 0]
        ref = re.sub("\s+", " ", l[1].replace(";", "").replace("<eps>", "")).strip()
        hyp = re.sub("\s+", " ", l[3].replace(";", "").replace("<eps>", "")).strip()
        references.append("".join([ref.replace("\n", "").strip(), " (", utt_id, ")\n"]))
        hypotheses.append("".join([hyp.replace("\n", "").strip(), " (", utt_id, ")\n"]))

    if out_dir is None:
        out_dir = os.path.dirname(path)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    ref_path = os.path.join(out_dir, "references.trn")
    hyp_path = os.path.join(out_dir, "hypotheses.trn")
    with open(ref_path, 'w', encoding='utf-8') as fw:
        fw.writelines(references)
    with open(hyp_path, 'w', encoding='utf-8') as fw:
        fw.writelines(hypotheses)

def _get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="A sequence of paths to \
        wer_test*.txt files containing speechbrain's outputs.")
    parser.add_argument("--out-dir", "-o", default=None, required=False,
        help="Directory where the references.trn and hypotheses.trn files will \
            be saved. If not specified, then the .trn files will be saved in\
            the same directory as the original files.")
    return parser

def _parse_args(args=None):
    if args is None:
        parser = _get_parser()
        args = parser.parse_args()
    if len(args.paths) == 1 and "*" in args.paths:
        for path in glob.glob(args.paths):
            print("Processing", path)
            convert_to_trn(path, args.out_dir)
        return
    paths = []
    for path in args.paths:
        if path.endswith(".txt"):
            if "*" in path:
                paths += glob.glob(path)
            else:
                paths.append(path)
        else:
            raise argparse.ArgumentTypeError(f"You should provide paths to wer_test*.txt files but found {path}.")
    for path in paths:
        print("Processing", path)
        convert_to_trn(path, args.out_dir)

if __name__ == "__main__":
    _parse_args()