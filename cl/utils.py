from typing import List, Callable, Optional, Union, Collection
import os
import re
import logging
import numpy as np
import torch
import subprocess

from cl import curriculum
import speechbrain as sb
from speechbrain.dataio.dataloader import SaveableDataLoader
import unicodedata


logger = logging.getLogger(__name__)

# From transcripts (relevant to lp)
WORDS_TO_REMOVE = ['\[oov\]']
SPECIAL_MARK_MATCHER = re.compile("\.\w+")


def checkpoint_wrapper_cl(func):
    """ This wrapper recovers the dataloader to where it was stopped.
    """
    def recover_if_applicable(brain, stage, epoch):
        dataloader = func(brain, stage, epoch)
        if stage != sb.Stage.TRAIN:
            return
        if not isinstance(dataloader, SaveableDataLoader):
            return dataloader
        if hasattr(brain, 'train_subset'):
            train_len = len(brain.train_subset)
        else:
            train_len = len(brain.train_set)
        if (
            brain.sorting in curriculum.CurriculumDataset.CURRICULUM_KEYS and \
            len(brain.sorting_dict) < train_len
           ) or (
            brain.sorting == "random" and brain._loaded_checkpoint is False
           ):
             # Load latest checkpoint to resume training if interrupted
            if brain.checkpointer is not None:
                old_epoch = brain.hparams.epoch_counter.current
                brain.checkpointer.recover_if_possible(
                    device=torch.device(brain.device)
                )
                # Keep the same old epoch
                brain.hparams.epoch_counter.current = old_epoch
                brain._loaded_checkpoint = True
            # brain.hparams.epoch_counter.current += 1
        return dataloader
    return recover_if_applicable

def min_max_normalize(
    c: Union[list, np.ndarray, int, float],
    minimum: Optional[float] = None,
    maximum: Optional[float] = None
) -> Union[np.ndarray, float]:
    if isinstance(c, list):
        c = np.asarray(c, dtype=np.float16)
    if not isinstance(c, np.ndarray):  # debug
        # We will get here when c is a single float or integer
        assert (minimum is not None) and (maximum is not None)
    if minimum is None:
        minimum = c.min()
    if maximum is None:
        maximum = c.max()
    return (c - minimum) / (maximum - minimum)

def confidence_normalization(
    confidences: List[float], 
    durations: Optional[List[float]] = None,
    norm_func: Callable[[Union[list, np.ndarray, int, float], float, float], np.ndarray] = min_max_normalize,
    epsilon: float = 1e-3
) -> List[float]:
    """ Maps the initial confidence values (negative log likelihoods) to a positive
        number between 0 and 1. If the new value is close to 0 then the confidence
        is bad, while values close to 1 indicate a very confident prediction.
    """
    confidences = np.asarray(confidences, dtype=np.float16)
    min_conf = confidences[np.where(np.isfinite(confidences))[0]].min()  # take the minimum number
    min_conf = min_conf + (np.sign(min_conf) * epsilon)
    if not np.isfinite(confidences).all():
        logger.warn(strip_spaces("Found non-finite number in the `confidences` array. \
            We are going to replace them with the minimum plus an epsilon."))
        confidences[~np.isfinite(confidences)] = min_conf
    # Make them positive (if they aren't already)
    confidences = np.sign(min_conf) * confidences
    if not min_conf:  # if it's zero
        logger.warn(f"Something has gone wrong since min_conf is equal to {min_conf}.")
    if durations is None:
        return norm_func(confidences)
    # Divide by duration
    confidences /= np.asarray(durations, dtype=np.float16)
    return norm_func(confidences)

def normalize_with_confs(d: dict, epsilon: float, norm_func=min_max_normalize) -> dict:
    # Step 1: Convert the dict to a list of tuples where the first elements are the ids
    #         E.g. d = [(id1, 87.23, -3.2, 8.1)]
    #                  [(id,  wer,   conf, dur)]
    #         Or, if keep_confs is False:
    #         E.g. d = [(id1, 87.23, 8.1)]
    #                  [(id,  wer,   dur)]
    sorting_list = [(k, *v) for k, v in d.items() if k != 'num_datapoints']
    # Step 2: Normalize the values into one single score
    if len(sorting_list[0]) == 4:  # then we also have durations
        # Short durations will give an extra penalty to the confidences
        # since we assume that short utterances are easier to transcribe.
        ids, wers, confs, durs = zip(*sorting_list)
        new_confs = confidence_normalization(confs, durations=durs, epsilon=epsilon)
        assert len(new_confs) == len(sorting_list)
    elif len(sorting_list[0]) == 3:  # then we have keep_confs=False
        ids, wers, durs = zip(*sorting_list)
    else:
        raise ValueError(f"Expected to find tuples of either (wer, conf, dur) \
            or just (wer, conf) but found {sorting_list[0]}.")
    # Normalize everything to [0-1]
    wers = norm_func(list(wers))
    # max_wer = max(wers) + epsilon  # avoid zeros
    try:
        return {identifier: ((wer+epsilon) * nconf, dur) for identifier, wer, dur, nconf in zip(ids, wers, durs, new_confs)}
    except NameError:
        return {identifier: (wer, dur) for identifier, wer, dur in zip(ids, wers, durs)}

def normalize_only_durs(d: dict, norm_func=min_max_normalize):
    """ Perform normalization (min-max by default) on the values of 
        a given dictionary. The dictionary values should be tuples where 
        the first elemnt is some score value and the second element is 
        the duration.
        Args:
            d: Dictionary of the form {key: (value, duration)}
        Returns:
            d: Dictionary of the form {key: (normalized_value, duration)}
    """
    ld0 = list(d.values())[0]
    assert isinstance(ld0, tuple) and len(ld0) == 2, f"Not a valid tuple: {ld0}"
    del ld0
    # Pop 'num_datapoints' so it won't interfere with the min-max scores
    num_datapoints = d.pop('num_datapoints', None)
    # E.g. if d={1: (2, 3), 2: (0.44, 1), 3: (0.5, 1.5)}
    #      then d_values_norm_iter will pass through [2/3, 0.44/1, 0.5/1.5]
    #      and so the min will be 0.5/1.5=0.33 and the max will be 2/3-0.66
    # If we don't do that then min_val and max_val will be tuples and we'll get an error
    # d_values_norm_iter = map(lambda val: val[0]/val[1], d.values())
    max_val = max(map(lambda val: val[0]/val[1], d.values()))
    min_val = min(map(lambda val: val[0]/val[1], d.values()))
    # Normalize to [0, 1] (min-max normalization by default)
    # Also divides each score with the corresponding duration
    d = {k: (norm_func(v[0]/v[1], min_val, max_val), v[1]) for k, v in d.items()}
    if num_datapoints is not None:
        d['num_datapoints'] = num_datapoints
    return d


def normalize_dict(d: dict, norm_func=min_max_normalize):
    """ Use this method to normalize the values of a dictionary when 
        there are NO confidences/durations involved.
        This simply performs a min-max normalization on the values.
    """
    # Pop 'num_datapoints' so it won't interfere with the min-max scores
    num_datapoints = d.pop('num_datapoints', None)
    max_val = max(d.values())
    min_val = min(d.values())
    # Normalize to [0, 1] (min-max normalization by default)
    # d = dict_apply(d, func=norm_func, minimum=min_val, maximum=max_val)
    d = {k: norm_func(v, min_val, max_val) for k, v in d.items()}
    if num_datapoints is not None:
        d['num_datapoints'] = num_datapoints
    return d

def strip_spaces(s: str):
    return re.sub(r"\s+", " ", s)

# Credits: https://github.com/geoph9/rust-wer/blob/master/python-equivalent/wer.py
def cer_minimal(h: Collection[Union[int, str]], r: Collection[Union[int, str]]) -> float:
    """ Calculation of Levenshtein distance.
        Works only for iterables up to 254 elements (uint8).
        O(nm) time and space complexity.
        Args:
            r : list of ints (encoded tokens) or a string
            h : list of ints (encoded tokens) or a string
        Return:
            int
    """
    # initialisation
    if len(h) == 0:
        return len(r)

    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution: np.uint8 = d[i - 1][j - 1] + 1
                insertion: np.uint8 = d[i][j - 1] + 1
                deletion: np.uint8 = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)]


def _process_text(text: str, remove_special_tokens: bool = False):
    text = re.sub("|".join(WORDS_TO_REMOVE), "", text).strip()
    if remove_special_tokens:
        # Special tokens are of the form .br, .fr, e.t.c.
        text = re.sub("\.\w+", "", text)
    text = re.sub("\s+", " ", text).strip()
    return text

def normalize_text(line, *args, **kwargs):
    # Remove special tokens:
    line = line.replace("[oov]", "")
    line = re.sub(SPECIAL_MARK_MATCHER, "", line)
    line = line.replace("<UNK>", "")
    line = line.replace("[spn]", "")
    line = line.replace("[spk]", "")
    line = line.replace("[int]", "")
    line = line.replace("[fil]", "")
    # Canonical forms of letters, see e.g. the Python docs
    # https://docs.python.org/3.7/library/unicodedata.html#unicodedata.normalize
    line = unicodedata.normalize("NFKC", line)
    # Just decide that everything will be lowercase:
    line = line.lower()
    # All whitespace to one space:
    line = " ".join(line.strip().split()) 
    # Remove all extra characters:
    line = "".join(char for char in line if char.isalpha() or char == " ")
    return line

def filelist_to_text_gen(filelist_path: str, remove_special_tokens: bool = False):
    """ Reads a filelist (a file containing paths to other text files), processes 
        each file's text and yields each single line.
    """
    assert os.path.isfile(filelist_path), f"Could not locate {filelist_path}."
    with open(filelist_path, 'r') as fr:
        for txt_path in fr:
            txt_path = txt_path.replace("\n", "").strip()
            assert os.path.isfile(txt_path), f"Could not locate {txt_path}."
            with open(txt_path, 'r') as ftxt:
                proc_txt = [_process_text(txt, remove_special_tokens) for txt in ftxt.readlines()]
                yield '\n'.join([txt for txt in proc_txt if txt not in ['', '\n']])

def wccount(filename):
    out = subprocess.Popen(['wc', '-l', filename],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT
                         ).communicate()[0]
    return int(out.partition(b' ')[0])