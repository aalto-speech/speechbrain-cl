import argh
import os
import re
from typing import Optional


FIRST_EXAMPLE_LINE = 12
SEPARATOR = "================================================================================"
DEFAULT_METRIC_THRESHOLD = 50.0  # i.e. we will keep all entries with a WER > 50.0 in order to analyze them

def read_single(path, threshold=DEFAULT_METRIC_THRESHOLD, print_stats=True):
    assert os.path.exists(path), f"Could not locate {path=}"
    basename = os.path.basename(path)
    is_wer = False if path.endswith("cer_test.txt") else True
    separate_func = lambda x: separate_test_entry(x, is_wer)
    with open(path, 'r', encoding='utf-8') as fr:
        # Before the FIRST_EXAMPLE_LINE line, we have general information 
        # and instructions on how to read the file so we don't care.
        lines = fr.readlines()[FIRST_EXAMPLE_LINE:]
        lines = "".join(lines).split(SEPARATOR)
        lines = [e[:-2] for e in map(separate_func, lines) if e[1] >= threshold]
        if print_stats:
            print("="*60)
            print(f"Number of utts with more than {DEFAULT_METRIC_THRESHOLD} {'WER' if is_wer else 'CER'}: {len(lines)}.")
            print(f"Insertions: {sum(e[2] for e in lines)}  |  Deletions: {sum(e[3] for e in lines)}  |  Substitutions: {sum(e[4] for e in lines)}")
    return lines
        
def compare(cer_path: str, wer_path: str, threshold: float = DEFAULT_METRIC_THRESHOLD, out_path: Optional[str] = None):
    if cer_path.endswith("wer_test.txt"):
        tmp = wer_path
        wer_path = cer_path
        cer_path = tmp
    assert os.path.isfile(cer_path) and os.path.isfile(wer_path)
    cl = read_single(cer_path, threshold)
    wl = read_single(wer_path, threshold)
    cer_bad_utts = map(lambda x: x[0], cl)
    wer_bad_utts = map(lambda x: x[0], wl)
    print("="*60)
    bad_overlaps = set(cer_bad_utts).intersection(set(wer_bad_utts))
    print(f"Found {len(bad_overlaps)} common bad utterances in the two sets.")
    if out_path is not None:
        with open(out_path, 'w') as fw:
            for e in cl:
                if e in bad_overlaps:
                    raise NotImplementedError()

    
def separate_test_entry(entry: str, is_wer: bool):
    # E.g.
    # sample-001053, %WER 0.00 [ 0 / 18, 0 ins, 0 del, 0 sub ]
    # i ; ' ; l ; l ; _ ; b ; e ; _ ; r ; i ; g ; h ; t ; _ ; h ; e ; r ; e
    # = ; = ; = ; = ; = ; = ; = ; = ; = ; = ; = ; = ; = ; = ; = ; = ; = ; =
    # i ; ' ; l ; l ; _ ; b ; e ; _ ; r ; i ; g ; h ; t ; _ ; h ; e ; r ; e
    # Returns a tuple of all relevant elemnts
    entry = [e for e in entry.split("\n") if len(e.strip()) != 0]
    assert len(entry) == 4, f"Length of entry should be 4. {entry=}"
    stats, truth, _, preds = entry
    if is_wer:
        reconstruct = lambda x: re.sub("\s+", " ", x.replace(";", "")).replace("<eps>", "").strip()
    else:
        reconstruct = lambda x: x.replace(";", "").replace(" ", "").replace("<eps>", "").replace("_", " ").strip()
    utt_id = stats.split(",")[0]
    truth_reconstructed = reconstruct(truth)
    preds_reconstructed = reconstruct(preds)
    score = float(stats.split("[")[-2].split(" ")[-2].strip())
    ins = int(stats.split("ins")[-2].split(",")[-1].strip())
    dels = int(stats.split("del")[-2].split(",")[-1].strip())
    subs = int(stats.split("sub")[-2].split(",")[-1].strip())
    return utt_id, score, ins, dels, subs, truth_reconstructed, preds_reconstructed, truth, preds


# dispatching:

if __name__ == '__main__':
    parser = argh.ArghParser()
    parser.add_commands([read_single, compare])
    parser.dispatch()