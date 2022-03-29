import re
from collections import Counter
import logging
from typing import Union, Callable, Dict, List, Optional, Tuple
from cl.curriculum import CurriculumDataset
from cl.filelist_tokenizer import FileListTokenizer
from cl import utils
from speechbrain.dataio.dataset import FilteredSortedDynamicItemDataset
from speechbrain.tokenizers.SentencePiece import SentencePiece


logger = logging.getLogger(__name__)

class InvalidFrequencyType(Exception): pass

class FrequencyCL(CurriculumDataset):
    VALID_FREQUENCY_TYPES = ['char', 'word', 'token']
    # Valid types:
    #   1. char: Uses character level frequency.
    #   2. word: Uses word level frequency.
    #   3. token: Token level frequency (e.g. BPE).
    DEFAULT_FREQUENCY_TYPE = "char"

    def __init__(self, *args, **kwargs):
        self.frequency_type: str = kwargs.pop("frequency_type", self.DEFAULT_FREQUENCY_TYPE)
        # NOTE: self.precomputed_freqs should change at every epoch
        #       when using a pacing function (since the frequencies of the
        #       characters will change. BUT, we may also provide the frequencies
        #       of the whole training set and avoid the re-calculation).
        self.precomputed_freqs: Dict[str, int] = kwargs.pop("precomputed_freqs", {})
        if self.frequency_type == "token":
            assert "tokenizer" in kwargs and \
                (isinstance(kwargs["tokenizer"], SentencePiece) or \
                    isinstance(kwargs["tokenizer"], FileListTokenizer)), "You should provide a valid tokenizer."
        assert self.frequency_type in self.VALID_FREQUENCY_TYPES, f"{self.frequency_type} not in {self.VALID_FREQUENCY_TYPES}."
        self.tokenizer: Union[SentencePiece, FileListTokenizer] = kwargs.pop("tokenizer", None)
        super().__init__(*args, **kwargs)
        self.calculate_frequency: Callable
        if self.frequency_type == "char":
            self.calculate_frequency = self._calculate_char_frequency
        elif self.frequency_type == "word":
            self.calculate_frequency = self._calculate_word_frequency
        elif self.frequency_type == "token":
            self.calculate_frequency = self._calculate_token_frequency
        else:
            raise InvalidFrequencyType(f"Invalid type of frequency calculator: {self.frequency_type}")

    def filtered_sorted(self,
        key_min_value: Optional[dict] = {},
        key_max_value: Optional[dict] ={},
        key_test: Optional[dict] = {},
        sort_key: Optional[str] = None,
        reverse: bool = False,
        select_n: int = None,
        sorting_dict: Optional[dict] = None,
        hparams: Optional[dict] = None,
        noise_percentage: Optional[float] = None,
    ) -> FilteredSortedDynamicItemDataset:
        if sort_key in self.VALID_FREQUENCY_TYPES:
            filtered_sorted_ids = self._curriculum_filtered_ids(sorting_dict, reverse, select_n)
            if isinstance(noise_percentage, float) and 0.0 < noise_percentage <= 1.0:
                # logger.info(f"{filtered_sorted_ids[:10]=}")
                filtered_sorted_ids = CurriculumDataset.add_random_noise(filtered_sorted_ids, noise_percentage)
                logger.info("Added some random noise among the easy examples.")
                # logger.info(f"{filtered_sorted_ids[:10]=}")
            filtered_trainset = FilteredSortedDynamicItemDataset(self, filtered_sorted_ids)
            return filtered_trainset
        return super().filtered_sorted(
            key_min_value, key_max_value, key_test, sort_key, reverse, select_n, 
            sorting_dict, hparams, noise_percentage
        )
    
    def _curriculum_filtered_ids(
        self,
        sorting_dict: Dict[str, Union[float, Tuple[float, float], Tuple[float, float, float]]],
        reverse: bool = False,
        select_n: Optional[int] = None,
        debug: bool = False,
        epsilon: float = None,
    ) -> List[str]:
        del epsilon  # not used
        select_n = select_n or (getattr(self, "current_epoch_n_datapoints", None) or len(sorting_dict))
        sorting_dict = sorting_dict or {}
        if len(sorting_dict) == len(self) and len(sorting_dict) >= len(self.precomputed_freqs):
            self.precomputed_freqs = sorting_dict.copy()
            logger.warn("Using the sorting dictionary as precomputed freqs. This may lead to errors when\
                calculating the frequency (since the sorting dictionary contains tuples).")
        select_n = round(select_n)
        reverse_freqs = self.calculate_frequency()
        reverse_freqs = utils.normalize_dict(reverse_freqs)
        filtered_sorted_ids: list = [utt_id for utt_id in sorted(
            reverse_freqs,
            key=lambda x: reverse_freqs[x],
            reverse=not reverse)
        ][:select_n]
        if debug:
            # Make sure that the sorting was successfull (debugging).
            for i, j in zip(filtered_sorted_ids[:-1], filtered_sorted_ids[1:]):
                if reverse:
                    assert sorting_dict[i]>=sorting_dict[j], f"i:{i}, j:{j}, di: {sorting_dict[i]}, dj: {sorting_dict[j]}" 
                else:
                    assert sorting_dict[i]<=sorting_dict[j], f"i:{i}, j:{j}, di: {sorting_dict[i]}, dj: {sorting_dict[j]}" 
            # Make sure that we have the valid batch ids
            for i, data_id in enumerate(self.data_ids):
                assert data_id in sorting_dict.keys(), f"Could not locate {data_id}."
        return filtered_sorted_ids

    @staticmethod
    def clean_str(text):
        return re.sub(r'[-\(\)\"#\/@;:<>\{\}\-=~|\.\?]', '', text).strip()
    
    def _calculate_char_frequency(self):
        # Higher score -> easier example
        reverse_sorting_dict = {}
        with self.output_keys_as(['wrd', 'id']) as dataset_trans:
            if len(self.precomputed_freqs) != len(dataset_trans):
                characters = ""
                for item in dataset_trans:
                    characters += self.clean_str(item['wrd'])
                self.precomputed_freqs = Counter(characters).pop(' ')
            for item in dataset_trans:
                trans = self.clean_str(item['wrd'])
                easyness = sum([self.precomputed_freqs[char] for char in trans])
                reverse_sorting_dict[item['id']] = easyness
        return reverse_sorting_dict

    def _calculate_word_frequency(self):
        # Higher score -> easier example
        reverse_sorting_dict = {}
        with self.output_keys_as(['wrd', 'id']) as dataset_trans:
            if len(self.precomputed_freqs) != len(dataset_trans):
                words = []
                for item in dataset_trans:
                    words += self.clean_str(item['wrd']).split()
                self.precomputed_freqs = Counter(words)
            for item in dataset_trans:
                words = self.clean_str(item['wrd']).split()
                easyness = sum([self.precomputed_freqs[w] for w in words])
                reverse_sorting_dict[item['id']] = easyness
        return reverse_sorting_dict

    def _calculate_token_frequency(self):
        # Higher score -> easier example
        reverse_sorting_dict = {}
        tokenize = self.tokenizer.sp.encode_as_ids if hasattr(self.tokenizer, 'sp') else self.tokenizer.encode_as_ids
        with self.output_keys_as(['wrd', 'id']) as dataset_trans:
            if len(self.precomputed_freqs) != len(dataset_trans):
                tokens = []
                for item in dataset_trans:
                    tokens += tokenize(item['wrd'])
                self.precomputed_freqs = Counter(tokens)
            for item in dataset_trans:
                tokens = tokenize(item['wrd'])
                easyness = sum([self.precomputed_freqs[w] for w in tokens])
                reverse_sorting_dict[item['id']] = easyness
        return reverse_sorting_dict
        
    pass
