AVAILABLE_METRICS = ['PER', 'WER', 'CER']
DEFAULT_METRICS = ['WER', 'CER']  # Must be one of the above.

# List of possible colors for matplotlib
MPL_COLORS = [
    "blue", "orange", "green", "red", "yellow", "black", 
    "brown", "pink", "gray", "olive", "cyan", "purple"
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