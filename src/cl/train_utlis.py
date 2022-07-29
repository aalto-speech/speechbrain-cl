#!/usr/bin/env python3
""" Generic code to fit a SpeechBrain ASR model with different
Curriculum Learning techniques. In particular, you may choose 
one of the following methods (in your <hyperparameters>.yaml file):
Sorting methods:
    - wer: the train set is decoded and then sorted based on wer 
        (set reverse=True for descending).
    - cer: similarly for cer (set reverse=True for descending)
    - ascending: duration-based ascending sorting.
    - descending: duration-based descending sorting.
    - seq_loss: A forward pass is performed on the trainset and its
        examples are then sorted based on the seq2seq loss value.
    - seq_ctc_loss: Similarly for seq2seq+CTC loss (assuming 
        you are using CTC).
    - ctc_loss: similarly for only CTC loss.
    - no: ther <train>.csv file is read sequentially and no sorting
        is performed,
    - random: the <train>.csv file's examples are randomly shuffled.

Usage:
    - The `fit()` method creates a `Brain` model and fits it on
        the data provided in the hparams file. You need to make sure
        that you have called a "preparation" file that initializes
        the exps directory and creates the *.csv files. The `train.py` 
        and `prepare_lp.py` files contain an example for the Lahjoita
        Puhetta dataset.
    - The `dataio_prepare()` method defines the text/audio pipelines 
        and returns the required datasets (`CurriculumDataset`s).
    - The `get_tokenizer()` method checks if we are using an LM and 
        if not, it creates a new one based on the <hparams>.yaml 
        hyperparameters.
    - The `use_lm()` method simply loads a pretrained LM. Currently,
        this has only been tested with the LibriSpeech LM.
    - The `webdataio_prepare()` is used for sharded datasets. In this
        case you must have already sharded your dataset in some directory
        and you must pinpoint the location in the <hparams>.yaml file.
        This was copy-pasted from Aku Rouhe's recipe.

Authors
 * George Karakasidis 2022
"""

import os
from copy import deepcopy
from cl.base_asr_model import BaseASR
import torch
import logging
import speechbrain as sb
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.distributed import run_on_main

from cl.asr_models import ASR, ASR_Old, AsrWav2Vec2
from cl.curriculum import CurriculumDataset
from cl.methods.frequency_cl import FrequencyCL
from cl.filelist_tokenizer import FileListTokenizer
from cl.utils.process_utils import normalize_text, strip_spaces
from cl.vad import testset_pipeline_with_segments, testset_pipeline_with_force_segments


# torch.cuda.set_device(0)
logger = logging.getLogger(__name__)


def fit(hparams, run_opts, overrides, ASR_Model=ASR):

    # Create the datasets objects as well as tokenization and encoding :-D
    if hparams.get("use_shards", False):
        data = webdataio_prepare(hparams)
        train_loader_kwargs = hparams['train_loader_kwargs']
    else:
        data = dataio_prepare(hparams, run_opts['device'])
        train_loader_kwargs = hparams["dataloader_options"]

    # Trainer initialization
    asr_brain: BaseASR = ASR_Model(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        opt_class=hparams["opt_class"],
        checkpointer=hparams["checkpointer"],
        sorting=hparams["sorting"],
        train_set=data['train'],
        train_loader_kwargs=train_loader_kwargs,
        tokenizer=data['tokenizer'],
        sorting_dict=None,
        profiler=hparams.get("profiler", None),
    )
    
    # Make sure that a checkpoint doesn't already exist
    do_fixed_sort_flag = True
    if hparams['sorting'] not in CurriculumDataset.CURRICULUM_KEYS:
        do_fixed_sort_flag = False
    elif "pretrained_model_hparams" not in hparams:
        do_fixed_sort_flag = False
    elif asr_brain.checkpointer.find_checkpoint() is None:
        assert os.path.isfile(hparams["pretrained_model_hparams"])
        _save_folder = os.path.join(hparams['output_folder'], "curriculum_logs")
        sorting_dict_log = os.path.join(_save_folder, f"{asr_brain.dict_log_prefix}{hparams['sorting']}_dict-epoch=0.log")
        logger.info(f"Trying to find sorting dictionary under: {sorting_dict_log}")
        if os.path.isfile(sorting_dict_log):
            logger.info(f"Found already existing sorting_dict_log, will try to load it.")
            asr_brain.load_sorting_dict(epoch=0)
            if len(asr_brain.sorting_dict) < len(asr_brain.train_set):
                logger.error(strip_spaces("Could not load sorting dictionary since it is not full. \
                    The previous sorting process was probably stopped amidst. Will continue \
                        as though nothing happened and we will create a new sorting dictionary"))
                asr_brain.sorting_dict = None
                do_fixed_sort_flag = True
            else:
                do_fixed_sort_flag = False
    else:
        do_fixed_sort_flag = False
    if do_fixed_sort_flag:
        logger.warning(strip_spaces(f"Will use a pretrained model to initialize the dictionary. \
            The path of the model's hyperparameters is {hparams['pretrained_model_hparams']}.\
                You should make sure yourself whether the relevant hyperparameters \
                    between the models are the same."))
        # Load hyperparameters
        with open(hparams["pretrained_model_hparams"]) as fin:
            overrides2 = "\n".join([o for o in overrides.split("\n") if "sorting" not in o and "seed" not in o])
            pretrained_hparams = load_hyperpyyaml(fin, overrides2)
        # Load tokenizer
        pm_tokenizer = get_tokenizer(pretrained_hparams, run_opts["device"])
        # Handle trainset pipeline (Here we are making a full copy of the trainset)
        train_dataset = deepcopy(data['train'])
        # 3. Define text pipeline:
        @sb.utils.data_pipeline.takes("wrd")
        @sb.utils.data_pipeline.provides(
            "tokens_list", "tokens_bos", "tokens_eos", "tokens"
        )
        def text_pipeline(wrd):
            tokens_list = pm_tokenizer.sp.encode_as_ids(wrd)
            yield tokens_list
            tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
            yield tokens_bos
            tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
            yield tokens_eos
            tokens = torch.LongTensor(tokens_list)
            yield tokens
        sb.dataio.dataset.add_dynamic_item([train_dataset], text_pipeline)

        asr_temp = ASR_Model(
            modules=pretrained_hparams["modules"],
            hparams=pretrained_hparams,
            run_opts=run_opts,
            opt_class=pretrained_hparams["opt_class"],
            checkpointer=pretrained_hparams["checkpointer"],
            sorting=hparams['sorting'],  # NOTE: We are using the new sorting here
            train_set=train_dataset,
            train_loader_kwargs={"batch_size": hparams['batch_size']},
            tokenizer=pm_tokenizer,
        )
        asr_temp.hparams.default_sorting_epochs = -1
        asr_brain.hparams.default_sorting_epochs = -1
        if not os.path.isdir(_save_folder):
            os.mkdir(_save_folder)
        sorting_dict = asr_temp.create_curriculum_dict(
            train_set=asr_temp.train_set, 
            sorting_dict_save_path=sorting_dict_log,
            try_recover=False,
        )
        del train_dataset, pm_tokenizer, pretrained_hparams, asr_temp
        if not asr_brain.do_adaptive_pacing:
            asr_brain.train_set = asr_brain.train_set.filtered_sorted(
                sort_key=hparams['sorting'], 
                sorting_dict=sorting_dict,
                reverse=hparams.get("reverse", False),
            )
        else:
            # otherwise the train set should stay as it is (a curriculum base)
            pass
        asr_brain.sorting_dict = sorting_dict
        asr_brain.final_sortings = sorting_dict.copy()
    else:
        asr_brain.sorting_dict = {}
    
    if hparams.get("use_fixed_sorting", False) \
      and hparams.get("pretrained_model_hparams", None) not in [None, False]:
        logger.info("Transfer learning CL approach...")
        if asr_brain.sorting_dict is None or len(asr_brain.sorting_dict) == 0:
            logger.info("Loading the precomputed sorting dictionary for CL.")
            asr_brain.sorting_dict = asr_brain.load_sorting_dict(epoch=0)
        train_set = asr_brain.make_dataloader(
            dataset=asr_brain.train_set,
            stage=sb.Stage.TRAIN,
            **asr_brain.train_loader_kwargs
        )
        logger.info("Created a dataloader and using that.")
    else:
        train_set = asr_brain.train_set
    # Training
    # with torch.autograd.detect_anomaly():
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_set,
        data['valid'],
        train_loader_kwargs=train_loader_kwargs,
        valid_loader_kwargs=hparams["valid_dataloader_options"],
    )

    # Test
    asr_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test.txt"
    asr_brain.hparams.cer_file = hparams["output_folder"] + "/cer_test.txt"
    if hparams.get("use_shards", False):
        test_loader_kwargs = {}
    else:
        test_loader_kwargs = hparams["test_dataloader_options"]
    # Load best checkpoint (highest STOI) for evaluation
    asr_brain.evaluate(
        data[hparams["test_data_id"]],
        min_key="WER",
        test_loader_kwargs=test_loader_kwargs,
    )

# TODO: Use gender and language?

# Define custom data procedure
def dataio_prepare(hparams, device):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    data_folder = hparams["data_folder"]
    
    # from shutil import copy
    # copy(hparams['train_csv'], hparams['train_csv'].replace(".csv", "_full.csv"))
    # with open(hparams['train_csv'], 'r', encoding='utf-8') as fr:
    #    lines = fr.readlines()
    #    lines = lines[:1501]
    # with open(hparams['train_csv'], 'w', encoding='utf-8') as fw:
    #    fw.writelines(lines)


    # Get tokenizer
    tokenizer = get_tokenizer(hparams, device)

    if hparams['sorting'] in FrequencyCL.VALID_FREQUENCY_TYPES:
        train_data = FrequencyCL.from_csv(
            csv_path=hparams["train_csv"],
            replacements={"data_root": data_folder},
            frequency_type=hparams['sorting'],
            tokenizer=tokenizer,
        )
    else:
        train_data = CurriculumDataset.from_csv(
            csv_path=hparams["train_csv"],
            replacements={"data_root": data_folder},
        )

    if hparams["sorting"] in train_data.CURRICULUM_KEYS:
        # In this case the training set will change at the start of every epoch
        # by calling `on_stage_start` of the ASR model.
        logger.info(f"Curriculum learning with '{hparams['sorting']}' sorting...")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False
        train_data.sorting = hparams["sorting"]
    
    elif hparams['sorting'] in FrequencyCL.VALID_FREQUENCY_TYPES:
        # In this case the training set will change only now based on char/word/token frequencies.
        logger.info(f"Curriculum learning with '{hparams['sorting']}' sorting...")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False
        train_data = train_data.filtered_sorted(
            sort_key=hparams['sorting'],
            reverse=hparams.get("reverse", False),
            noise_percentage=hparams.get('noisy_random_percentage', None),
        )
        print(f"Lenght of sorted dataset: {len(train_data)=}")

    elif hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            noise_percentage=hparams.get('noisy_random_percentage', None),
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True,
            noise_percentage=hparams.get('noisy_random_percentage', None),
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        hparams['do_subsample'] = True
        hparams['subsampling_percentage'] = 1.
        hparams['subsampling_n_epochs'] = hparams['number_of_epochs']
        hparams['subsampling_increase_factor'] = None
        # We will shuffle only once (otherwise we would shuffle every
        # time we recovered from a checkpoint).
        hparams["dataloader_options"]["shuffle"] = False
    elif hparams["sorting"] in ["no", False]:
        hparams["dataloader_options"]["shuffle"] = False
        pass
    else:
        curr_keys = ", ".join(train_data.CURRICULUM_KEYS)
        raise NotImplementedError(
            f"Invalid 'sorting' value: {hparams['sorting']}. 'sorting'\
                must be one of: {curr_keys}, random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], 
        replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], 
        replacements={"data_root": data_folder},
    )
    if hparams.get("sort_test_set", True):
        # We also sort the test data
        test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data]
    if hparams.get("use_vad", False):
        max_dur = hparams.get("max_duration_secs", 50.)
        min_dur = hparams.get("min_duration_secs", 1.)
        if hparams.get("segment_forcefully_secs", -1.0) > min_dur:
            test_data = testset_pipeline_with_force_segments(
                test_set=test_data,
                tokenizer=tokenizer,
                overlap_duration_secs=hparams.get('overlap_duration_secs', 0.05),
                sample_rate=hparams['sample_rate'],
                bos_index=hparams['bos_index'],
                eos_index=hparams['eos_index'],
                max_duration_secs=hparams['segment_forcefully_secs'],
                min_duration_secs=min_dur,
            )
        else:
            test_data = testset_pipeline_with_segments(
                test_set=test_data,
                vad=hparams['VAD'](),
                tokenizer=tokenizer,
                sample_rate=hparams['sample_rate'],
                bos_index=hparams['bos_index'],
                eos_index=hparams['eos_index'],
                max_duration_secs=max_dur,
                min_duration_secs=min_dur,
            )
    else:
        # Default audio pipeline for the test set
        datasets.append(test_data)
    
    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes(
        "wav",
        "start",
        "end",
    )
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, end):
        info = torchaudio.info(wav)
        sr = info.sample_rate

        start = int(float(start)*sr)
        end = int(float(end)*sr)
        sig = sb.dataio.dataio.read_audio({
            'file': wav,
            'start': start,
            'stop': end,
        })
        resampled = torchaudio.transforms.Resample(
            sr, hparams["sample_rate"],
        )(sig)

        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    if hparams.get("use_lm"):
        @sb.utils.data_pipeline.takes("wrd")
        @sb.utils.data_pipeline.provides(
            "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
        )
        def text_pipeline(wrd):
            yield wrd
            tokens_list = tokenizer.encode_as_ids(wrd)
            yield tokens_list
            tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
            yield tokens_bos
            tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
            yield tokens_eos
            tokens = torch.LongTensor(tokens_list)
            yield tokens
    else:
        @sb.utils.data_pipeline.takes("wrd")
        @sb.utils.data_pipeline.provides(
            "tokens_list", "tokens_bos", "tokens_eos", "tokens"
        )
        def text_pipeline(wrd):
            tokens_list = tokenizer.sp.encode_as_ids(wrd)
            yield tokens_list
            tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
            yield tokens_bos
            tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
            yield tokens_eos
            tokens = torch.LongTensor(tokens_list)
            yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "tokens_bos", "tokens_eos", "tokens"],
    )
    return {"train": train_data, "valid": valid_data, "test": test_data, "tokenizer": tokenizer}

def webdataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.


    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Dictionary containing "train", "valid", and "test" keys mapping to 
        WebDataset datasets dataloaders for them.
    """
    import webdataset as wds

    tok_kwargs = {}
    tok_kwargs['bos_id'] = hparams['bos_index']
    tok_kwargs['eos_id'] = hparams['eos_index']

    if os.path.isfile(hparams.get("lp_filelist", "")):
        logger.info("Creating a tokenizer based on a filelist (more text data).")
        # Better keep `remove_special_tokens` as False
        tok_kwargs['remove_special_tokens'] = hparams.get('remove_special_tokens', False)
        tokenizer = FileListTokenizer(hparams, **tok_kwargs)
    else:
        logger.info("Creating simple SentencePiece tokenizer...")
        # defining tokenizer and loading it
        tokenizer = SentencePiece(
            model_dir=hparams["save_folder"],
            vocab_size=hparams["output_neurons"],
            annotation_train=hparams["train_csv"],
            annotation_read="wrd",
            model_type=hparams["token_type"],
            character_coverage=hparams["character_coverage"],
            **tok_kwargs,
            # pad_id=hparams.get('pad_index', -1),
        )

    def tokenize(sample):
        text = sample["wrd"]
        text = normalize_text(text)
        sample["wrd"] = text
        fulltokens = torch.LongTensor(
                [hparams["bos_index"]] + tokenizer.sp.encode(text) + [hparams["eos_index"]]
        )
        sample["tokens"] = fulltokens[1:-1]
        sample["tokens_bos"] = fulltokens[:-1]
        sample["tokens_eos"] = fulltokens[1:]
        return sample
    
    traindata = (
            wds.WebDataset(hparams["trainshards"])
            .decode()
            .rename(wrd="transcript.txt", sig="audio.pth")
            .map(tokenize)
            .repeat()
            .then(
                sb.dataio.iterators.dynamic_bucketed_batch,
                **hparams["dynamic_batch_kwargs"]
            )
    )
    validdata = (
            wds.WebDataset(hparams["validshards"])
            .decode()
            .rename(wrd="transcript.txt", sig="audio.pth")
            .map(tokenize)
            .batched(
                batchsize=hparams["valid_batch_size"], 
                collation_fn=sb.dataio.batch.PaddedBatch,
                partial=True
            )
    )

    return {"train": traindata, "valid": validdata, "tokenizer": tokenizer}

def get_tokenizer(hparams, device, annotation_read="wrd"):
    if _use_lm(hparams, device):
        return hparams['tokenizer']
    # tok_kwargs = {
    #     'bos_id': hparams['bos_index'],
    #     'eos_id': hparams['eos_index'],
    # }
    tok_kwargs = {}
    if hparams.get("use_wav2vec2", False) or hparams['bos_index'] != hparams['eos_index']:
        tok_kwargs['bos_id'] = hparams['bos_index']
        tok_kwargs['eos_id'] = hparams['eos_index']

    if os.path.isfile(hparams.get("lp_filelist", "")):
        logger.info("Creating a tokenizer based on a filelist (more text data).")
        # Better keep `remove_special_tokens` as False
        tok_kwargs['remove_special_tokens'] = hparams.get('remove_special_tokens', False)
        tokenizer = FileListTokenizer(hparams, **tok_kwargs)
    else:
        logger.info("Creating simple SentencePiece tokenizer...")
        # defining tokenizer and loading it
        tokenizer = SentencePiece(
            model_dir=hparams["save_folder"],
            vocab_size=hparams["output_neurons"],
            annotation_train=hparams["train_csv"],
            annotation_read=annotation_read,
            model_type=hparams["token_type"],
            character_coverage=hparams["character_coverage"],
            **tok_kwargs,
            # pad_id=hparams.get('pad_index', -1),
        )
    return tokenizer

def _use_lm(hparams, device):
    if hparams.get("use_lm", False):
        if "pretrainer" not in hparams or\
            "tokenizer" not in hparams or\
            "pretrained_lm_tokenizer_path" not in hparams:  # This is probably not needed
            raise KeyError(strip_spaces("You should provide 'pretrainer', 'tokenizer' \
                and 'pretrained_lm_tokenizer_path' (the last is not really \
                mandatory so if that's what you are missing maybe you should \
                change this if statement.)"))
        # We download the pretrained LM from HuggingFace (or elsewhere depending on
        # the path given in the YAML file). The tokenizer is loaded at the same time.
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(device=device)
        return True
    return False
