import re

AVAILABLE_METRICS = ['PER', 'WER', 'CER']
DEFAULT_METRICS = ['WER', 'CER']  # Must be one of the above.

# List of possible colors for matplotlib
DARK_COLORS = [
    "black", 'darkgray','gray','dimgray','lightgray'
]
MPL_COLORS = [
    "blue", "green", "red", "black", 
    "brown", "gray", "olive", "cyan", "purple",
    # "orange", "yellow", "pink", 
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