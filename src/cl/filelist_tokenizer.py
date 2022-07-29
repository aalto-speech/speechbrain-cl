#!/usr/bin/env python3
""" Train a SentencePience model on the whole Lahjoita Puhetta dataset.
This corresponds to roughly 1500 hours of spoken data.

The FileListTokenizer class is adapted from
`speechbrain/speechbrain/tokenizers/SentencePiece.py`

Authors
 * George Karakasidis 2021

"""

import os
import logging
import re
import torch
import sentencepiece as spm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import merge_char
from speechbrain.utils.distributed import run_on_main
from cl.utils.process_utils import _process_text, filelist_to_text_gen

# torch.cuda.set_device(0)
logger = logging.getLogger(__name__)


class FileListTokenizer:
    """This function train a SentencePiece model and saved it in the corresponding
    directory.
    """
    def __init__(self, 
        hparams, 
        bos_id=-1, 
        eos_id=-1, 
        pad_id=-1,
        unk_id=0,
        character_coverage=1.0,
        split_by_whitespace=True,
        char_format_input=False, 
        user_defined_symbols=None,
        max_sentencepiece_length=10,
        remove_special_tokens=False,
        output_folder=None,
    ):
        if isinstance(hparams, str):  # path
            with open(hparams) as fin:  # then load into a dictionary
                hparams = load_hyperpyyaml(fin)
        self.hparams = hparams
        self.vocab_size = str(self.hparams["output_neurons"])
        self.model_type = self.hparams["token_type"]
        self.user_defined_symbols = user_defined_symbols
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.max_sentencepiece_length = max_sentencepiece_length
        self.character_coverage = character_coverage
        self.char_format_input = char_format_input
        self.split_by_whitespace = split_by_whitespace
        self.remove_special_tokens = remove_special_tokens or self.hparams.get('remove_special_tokens', False)
        # 1. Get filelist containing paths to the transcript files
        #    E.g. /path/to/lp-trns.txt
        #    The file shall be in the format:
        #      /path/to/transcript1.txt
        #      /path/to/transcript2.txt
        #      ...
        #      /path/to/transcriptN.txt
        #    Each file may contain multiple lines of text. We will combine them.
        lp_trans_path = self.hparams.get("lp_filelist")
        if lp_trans_path is None:
            raise ValueError("`lp_filelist` parameter missing from the yaml file.")
        if output_folder is None:
            output_folder = self.hparams.get("spm_folder", self.hparams['save_folder'])
        
        # 2. Now create a .txt file of all transcripts combined
        self.text_file: str = os.path.join(output_folder, "lp-train-complete-sp.txt")# self.hparams["train_csv"].replace(".csv", "_sp.txt")
        # 3. Define tokenizer's model file
        self.prefix_model_file = os.path.join(
            output_folder, 
            self.vocab_size + "_" + self.model_type
        )
        if not os.path.isfile(self.prefix_model_file+".model"):
            if not os.path.isfile(self.text_file):
                run_on_main(self.filelist_to_text, kwargs={"filelist_path": lp_trans_path})
            # 4. Now train a bpe model
            run_on_main(self._train_BPE)
        
        # Create SentencePiece model
        logger.info("==== Loading Tokenizer ===")
        logger.info("Tokenizer path: " + self.prefix_model_file + ".model")
        logger.info("Tokenizer vocab_size: " + str(self.vocab_size))
        logger.info("Tokenizer type: " + self.model_type)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.prefix_model_file + ".model")
    
    def _check_coverage_from_bpe(self, *args, **kwargs):
        raise NotImplementedError("This method hasn't been implemented in the "+
            "FileListTokenizer. If you need it, you may simply copy-paste the "+
            "default speechbrain SentencePiece tokenizer's method.")

    def filelist_to_text(self, filelist_path: str):
        """ Converts a filelist of paths to transcripts to a single text file of 
            all transcripts.
        """
        with open(self.text_file, 'w') as fw:
            for txt in filelist_to_text_gen(filelist_path, self.remove_special_tokens):
                fw.write(txt)

    def _train_BPE(self):
        """Train tokenizer with unsupervised techniques (BPE, Unigram) using
        SentencePiece Library. If you use "char" mode, the SentencePiece
        creates a char dict so the vocab_size attribute is not needed.

        Adjusted from speechbrain/speechbrain/tokenizers/SentencePiece.py
        """
        query = (
            "--input="
            + self.text_file
            + " --model_prefix="
            + self.prefix_model_file
            + " --model_type="
            + self.model_type
            + " --bos_id="
            + str(self.bos_id)
            + " --eos_id="
            + str(self.eos_id)
            + " --pad_id="
            + str(self.pad_id)
            + " --unk_id="
            + str(self.unk_id)
            + " --max_sentencepiece_length="
            + str(self.max_sentencepiece_length)
            + " --character_coverage="
            + str(self.character_coverage)
        )
        if self.model_type not in ["char"]:
            # include vocab_size
            query += " --vocab_size=" + self.vocab_size
        if self.hparams.get("user_defined_symbols", None) is not None:
            query += " --user_defined_symbols=" + self.hparams["user_defined_symbols"]
        elif self.user_defined_symbols is not None:
            query += " --user_defined_symbols=" + self.user_defined_symbols
        if not self.split_by_whitespace:
            query += " --split_by_whitespace=false"
        # Train tokenizer
        spm.SentencePieceTrainer.train(query)

    def __call__(
        self, batch, batch_lens=None, ind2lab=None, task="encode",
    ):
        """ Copy-pasted from the original file (check top).
        The reason this is copy-pasted is not to require changes to the current code.
        Ofc this could lead to issues if the main branch of speechbrain makes some 
        changes here, so please keep an eye.
        =============================================================================
        This __call__ function implements the tokenizer encoder and decoder
        (restoring the string of word) for BPE, Regularized BPE (with unigram),
        and char (speechbrain/nnet/RNN.py).
        Arguments
        ----------
        batch : tensor.IntTensor or list
            List if ( batch_lens = None and task = "decode_from_list")
            Contains the original labels. Shape: [batch_size, max_length]
        batch_lens : tensor.LongTensor
            Containing the relative length of each label sequences. Must be 1D
            tensor of shape: [batch_size]. (default: None)
        ind2lab : dict
            Dictionary which maps the index from label sequences
            (batch tensor) to string label.
        task : str
            ("encode", "decode", "decode_from_list)
            "encode": convert the batch tensor into sequence of tokens.
                the output contain a list of (tokens_seq, tokens_lens)
            "decode": convert a tensor of tokens to a list of word sequences.
            "decode_from_list": convert a list of token sequences to a list
                of word sequences.
        """
        if task == "encode" and ind2lab is None:
            raise ValueError("Tokenizer encoder must have the ind2lab function")

        if task == "encode":
            # Convert list of words/chars to bpe ids
            bpe = []
            max_bpe_len = 0
            batch_lens = (batch_lens * batch.shape[1]).int()
            for i, utt_seq in enumerate(batch):
                tokens = [
                    ind2lab[int(index)] for index in utt_seq[: batch_lens[i]]
                ]
                if self.char_format_input:
                    (words_list,) = merge_char([tokens])
                    sent = " ".join(words_list)
                else:
                    sent = " ".join(tokens)
                bpe_encode = self.sp.encode_as_ids(sent)
                bpe.append(bpe_encode)
                # save the longest bpe sequence
                # it help to compute the relative length of each utterance
                if len(bpe_encode) > max_bpe_len:
                    max_bpe_len = len(bpe_encode)
            # Create bpe tensor
            bpe_tensor = torch.zeros(
                (batch.shape[0], max_bpe_len), device=batch.device
            )
            bpe_lens = torch.zeros((batch.shape[0]), device=batch.device)
            for i, bpe_utt in enumerate(bpe):
                bpe_tensor[i, : len(bpe_utt)] = torch.Tensor(bpe_utt)
                bpe_lens[i] = len(bpe_utt) / max_bpe_len
            return bpe_tensor, bpe_lens
        elif task == "decode_from_list":
            # From list of hyps (not padded outputs)
            # do decoding
            return [self.sp.decode_ids(utt_seq).split(" ") for utt_seq in batch]
        elif task == "decode":
            # From a batch tensor and a length tensor
            # find the absolute batch lengths and do decoding
            batch_lens = (batch_lens * batch.shape[1]).int()
            return [
                self.sp.decode_ids(
                    utt_seq[: batch_lens[i]].int().tolist()
                ).split(" ")
                for i, utt_seq in enumerate(batch)
            ]