"""
Data preparation for Common Voice.

Authors:
 * George Karakasidis 2022

Adapted from:
 * Mirco Ravanelli, Ju-Chieh Chou, Loren Lugosch 2020
"""

import csv
import logging
import os
import random
import statistics
from shutil import copy
from typing import List
from typing import Optional

import torchaudio
from speechbrain.dataio.dataio import load_pkl
from speechbrain.dataio.dataio import save_pkl

from cl.utils.process_utils import normalize_text as _process_text
from cl.utils.process_utils import wccount


try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

logger = logging.getLogger(__name__)
OPT_FILE = "opt_cv_en_prepare.pkl"


def prepare_cv8(
    data_folder: str,
    save_folder: str,
    skip_prep: bool = False,
    min_duration_secs: float = 1.0,  # 1 second
    max_duration_secs: float = 9.0,  # 9 seconds
    train_dir: str = "train",
    dev_dir: str = "dev",
    test_dir: str = "test",
    recalculate_csvs_if_exist: bool = False,
    train_sample_perc: Optional[float] = None,
    train_csv: Optional[str] = None,
):
    """
    Prepares the csv files for the Mozilla Common Voice dataset (v8).

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    skip_prep: bool
        If True, data preparation is skipped.
    min_duration_secs: float
        Default: 1 second
        Minimum duration of each audio chunk. Small audio chunks may result in errors
        and so they will be skipped.
    max_duration_secs: float
        Default: 11 seconds
        Maximum duration of each audio chunk. Large audio chunks may result in errors.
    remove_special_tokens: bool
        Default: False
        Defines whether the special tokens of the form .br, .fr etc will be used for
        training or not.
    train_dir: str
        Default: train
    dev_dir: str
        Default: dev
    dev_dir: str
        Default: test
    recalculate_csvs_if_exist: bool
        If True then whether the train/dev/test csv files exist or not, they will
        be calculated from scratch.
        Default: False
    train_sample_perc: float (Optional)
        if provided then we will only use a subset of the trainset.
    train_csv: str (Optional)
        Path to a <train>.csv file. If provided then we are not going to
        calculate a new <train>.csv file.

    Example
    -------
    >>> data_folder = 'datasets/cv8/'
    >>> save_folder = 'cv8_prepared'
    >>> prepare_lp(data_folder, save_folder)
    """

    if skip_prep:
        return
    splits = [train_dir, dev_dir, test_dir]

    conf = {
        "splits": ",".join(splits),
        "min_duration_secs": min_duration_secs,
        "max_duration_secs": max_duration_secs,
    }

    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, probably completed in previous run.")
        return
    else:
        logger.info("Data preparation...")

    # Additional checks to make sure the data folder contains CommonVoice v1
    _check_cv_v8_folders(data_folder, splits)

    # create csv files for each split
    # In case the train_csv has already been provided, we don't need to create it.
    if train_csv is not None and os.path.isfile(train_csv):
        logger.warning(
            f"{'='*80}\nWill copy a pre-created <train>.csv file (possibly a subset?).\n{'='*80}"
        )
        copy(train_csv, os.path.join(save_folder, f"{splits[0]}.csv"))

    for split in splits:
        logger.info(
            f"=============== Processing {split} split. ==============="
        )
        # Read as tsv but save as csv
        create_csv(
            orig_tsv_file=os.path.join(data_folder, f"{split}.tsv"),
            output_csv_file=os.path.join(save_folder, f"{split}.csv"),
            clips_folder=os.path.join(data_folder, "clips"),
            min_duration_secs=min_duration_secs,
            max_duration_secs=max_duration_secs,
            ignore_if_already_exists=not recalculate_csvs_if_exist,
        )
        logger.info("=" * 60)

    # In case we need to create a fixed subset
    if train_sample_perc is not None:
        train_csv = os.path.join(save_folder, f"{splits[0]}.csv")
        sample_subset(
            csv_path=train_csv, sample_perc=train_sample_perc, out_path=train_csv
        )
    # saving options
    save_pkl(conf, save_opt)


def create_csv(
    orig_tsv_file: str,
    output_csv_file: str,
    clips_folder: str,
    min_duration_secs: float = 1.0,
    max_duration_secs: float = 9.0,
    ignore_if_already_exists: bool = True,
):
    """
    Creates the csv file given a list of wav files.
    Arguments
    ---------
    orig_tsv_file : str
        Path to the Common Voice tsv file (standard file).
    output_csv_file: str
        Path to the output csv file which will be used by SpeechBrain.
    clips_folder : str
        Path of the CommonVoice dataset (the clips directory).
    min_duration_secs: float
        Minimum allowed audio duration in seconds.
    max_duration_secs: float
        Maximum allowed audio duration in seconds.
    ignore_if_already_exists: bool
        Whether or not we should recalculate the csv files if they exist.
    Returns
    -------
    None
    """

    # if "train" in output_csv_file or "dev" in output_csv_file:
    #     ignore_if_already_exists = True
    if os.path.isfile(output_csv_file) and ignore_if_already_exists:
        logger.warning(f"Ignoring {output_csv_file} since it already exists.")
        return
    # Check if the given files exists
    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    # f = open(orig_tsv_file, "r")
    num_ignored = 0
    ignored_duration = 0.0
    number_of_audios = 0
    durations = []
    segments = []
    num_nan_files = 0
    with open(output_csv_file, "w", encoding="utf8") as csv_fw:
        csv_writer = csv.writer(
            csv_fw, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        header_row = ["ID", "start", "end", "duration", "wav", "spk_id", "wrd"]
        csv_writer.writerow(header_row)
        nb_samples = wccount(orig_tsv_file)

        with open(orig_tsv_file) as tsv_fr:
            _ = next(tsv_fr)  # first line is the header

            # Adding some prints
            msg = "Creating csv lists in %s ..." % (output_csv_file)
            logger.info(msg)

            # Start processing lines
            for line in tqdm(tsv_fr, total=nb_samples):
                spk_id, _tmp_path, sentence = line.split("\t")[:3]
                utt_id = _tmp_path.split(".")[0]
                segment = line.split("\t")[-1].replace("\n", "").strip()
                if segment != "":
                    segments.append(segment)
                mp3_path = os.path.join(clips_folder, _tmp_path)

                # Reading the signal (to retrieve duration in seconds)
                if os.path.isfile(mp3_path):
                    info = torchaudio.info(mp3_path)
                else:
                    msg = "\tError loading: %s" % (str(mp3_path))
                    # logger.error(msg)
                    continue

                duration = info.num_frames / info.sample_rate
                if not (min_duration_secs <= duration <= max_duration_secs):
                    num_ignored += 1
                    ignored_duration += duration
                    continue
                durations.append(duration)

                # Remove too short sentences (or empty):
                if len(sentence) < 3:
                    continue

                start = str(0.0)
                stop = str(duration)
                wrd = _process_text(sentence).strip()
                # Composition of the csv_line (the speaker id is the same as the utt id.)
                #           <UTT>  <START> <END>  <DURATION>     <WAV>    <SPK>  <WORDS>
                csv_line = [utt_id, start, stop, str(duration), mp3_path, spk_id, wrd]
                # Ignore rows that contain NaN values
                if any(i != i for i in csv_line) or len(wrd) == 0:
                    num_nan_files += 1
                    continue
                # Adding this line to the csv_lines list
                number_of_audios += 1
                csv_writer.writerow(csv_line)

    # Final prints
    total_duration = sum(durations)
    logger.info(f"{output_csv_file} successfully created!")
    logger.info(f"Found {len(segments)} `segments` of CV8.")
    if len(segments) > 0:
        logger.info(
            f"They probably need special handling. The first segment was: {segments[:1]}."
        )
    logger.info(f"Number of samples: {number_of_audios}.")
    logger.info(f"Total duration: {round(total_duration / 3600, 2)} hours.")
    logger.info(
        f"Median/Mean duration: {round(statistics.median(durations), 2)}/{round(total_duration/len(durations), 2)}."
    )
    logger.info(
        f"Ignored {num_ignored} audio files in total (due to duration issues) out of {number_of_audios+num_ignored}."
    )
    if num_nan_files > 0:
        logger.info(f"Ignored {num_nan_files} utterances due to nan value issues.")
    logger.info(
        f"Total duration of ignored files {round(ignored_duration / 60, 2)} minutes."
    )


def skip(splits: List[str], save_folder: str, conf: dict) -> bool:
    """
    Detect when the data preparation can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the save directory
    conf : dict
        The configuration options to ensure they haven't changed.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking csv files
    skip = True

    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split + ".csv")):
            skip = False
            break

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip


def sample_subset(csv_path: str, sample_perc: float, out_path: str):
    assert os.path.isfile(csv_path), f"{csv_path=}"
    assert isinstance(sample_perc, float) and 0 < sample_perc < 1, f"{sample_perc=}"
    new_csv_lines = []
    with open(csv_path) as f:
        header = f.__next__()
        new_csv_lines.append(header)
        lines = f.readlines()
        random.shuffle(lines)
    new_csv_lines += lines[: int(sample_perc * len(lines))]
    with open(out_path, "w") as fw:
        fw.writelines(new_csv_lines)


def _check_cv_v8_folders(data_folder: str, splits: List[str]):
    """
    Check if the data folder actually contains the eight version
    of the Common Voice dataset (English only).

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain the desired dataset.
    """

    def check_file_exists(f, *args):
        if len(args) != 0:
            f = os.path.join(f, *args)
        if not os.path.exists(f):
            err_msg = (
                "the folder %s does not exist while it is expected for the "
                "common-voice english dataset (version 1)." % f
            )
            raise FileNotFoundError(err_msg)

    check_file_exists(os.path.join(data_folder, "clips"))
    # Checking if all the splits exist
    for split in splits:
        # Expect the `cv-valid-{split}` folder and .csv file to exist
        split_tsv = os.path.join(data_folder, f"{split}.tsv")
        check_file_exists(split_tsv)
