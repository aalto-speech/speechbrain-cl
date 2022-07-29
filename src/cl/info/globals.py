import re

AVAILABLE_METRICS = ['PER', 'WER', 'CER']
DEFAULT_METRICS = ['WER', 'CER']  # Must be one of the above.

# List of possible colors for matplotlib
DARK_COLORS = [
    "black", 'darkgray','gray','dimgray','lightgray'
]
MPL_COLORS = [
    "black", #"pink",
    "brown", "gray", "olive", "cyan", "purple",
    # "orange", "yellow", "blue", "green", "red",
]

MPL_MARKERS = [
    # "-", "--", "o", "<", ">", "s", "p", "*", "h", "H", "x", "D"
    "o", ".", ">", "4", "8", "s", "p", "*", "h", "H", "X", "D"
]

MAPPINGS = {
    "Crdnn": "CRDNN",
    "Wer": "WER",
    "Cer": "CER",
    "Seqloss": "Sequence-to-Sequence Loss",
    "Confs": "w/ Confidences",
    "5k": "",
    "Seqctcloss": "Sequence-to-Sequence w/ CTC Loss",
    "Simple": "Ascending",
    "Base": "Ascending",
    "W2v2": "Wav2Vec 2.0",
}

STRATEGIES = [
    "WER", "SEQ", "WRD", "CHR", "TOK"
]

# Map strategy names to their abbreviations
def map_name_thesis(basename, remove_runs=True):
    if "(#runs" in basename.lower() and remove_runs:
        basename = basename.split(" (#")[0].strip()
    basename = basename.replace("_ga", "").replace("_noisy", "*").\
            replace("transfer_fixed", "trf").replace("transfer", "TR").\
            replace("adaptive_pacing", "AP").replace("seq_loss", "SEQ").\
            replace("_", "-").upper().replace("SUBSAMPLING", "SUB").\
            replace("SUBSAMPLE", "SUB").replace("BASELINE", "BASE").\
            replace("ASCENDING", "DUR").replace("-FULLASC", "").\
            replace("DESCENDING", "DUR$\\downarrow$").replace("-BASE", "").\
            replace("AP", "VPF").replace("RANDOM", "RND").\
            replace("TOKEN", "TOK").replace("WORD", "WRD").\
            replace("CHAR", "CHR").replace("LP-SUBSET", "").\
            replace("0.1", "10").replace("0.3", "30")
    basename = basename.replace("SUB", "SPF")
    basename = basename.replace("WER-TRFN", "TR-WER*")
    basename = basename.replace(" ", "")
    if "N-" in basename:
        basename = basename.replace("TRFN", "TR") + "*"
    for strategy in STRATEGIES:
        if f"{strategy}-" in basename:
            # strategy name should be always at the end
            basename = basename.replace(f"{strategy}-", "") + f"-{strategy}"
            if "*" in basename:
                basename = basename.replace("*", "") + "*"
    basename = basename.replace("TRF", "TR")
    if "TR-" in basename and "PF-" in basename:
        basename = basename.replace("TR-", "")
        basename = basename.replace("PF-", "PF-TR-")
    if "DESC-" in basename:
        basename = basename.replace("DESC-", "") + "$\\downarrow$"
    if "TRN-" in basename:
        basename = basename.replace("TRN-", "TR-") + "*"
    return basename

def map_name(model_id, name_mappings):
    if not name_mappings or not isinstance(name_mappings, dict):
        return model_id
    name = ""
    for cl in name_mappings['curriculum_mappings']:
        if cl in model_id:
            name = name_mappings['curriculum_mappings'][cl]
            break
    if name == "":
        print("Could not match model_id to a name.")
        return model_id
    for transfer_map in name_mappings['transfer_mappings']:
        if transfer_map in model_id:
            name = name_mappings['transfer_mappings'][transfer_map] + name
            break
    for subset_name in name_mappings['subset_mappings']:
        if subset_name in model_id:
            name = "".join(["(", name_mappings['subset_mappings'][subset_name], ")", " ", name])
            break
    for other_name in name_mappings['other_mappings']:
        if other_name in model_id:
            name = name_mappings['other_mappings'][other_name].replace("{name}", name)
            break
    name = re.sub("\s+", " ", name)
    return name