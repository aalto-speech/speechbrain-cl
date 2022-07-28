import copy
import re
from collections import Counter
import logging
from typing import Union, Callable, Dict, List, Optional, Tuple
from cl.curriculum import CurriculumDataset
from cl.filelist_tokenizer import FileListTokenizer
from cl import utils
from speechbrain.dataio.dataio import load_data_csv
from speechbrain.dataio.dataset import FilteredSortedDynamicItemDataset
from speechbrain.tokenizers.SentencePiece import SentencePiece


logger = logging.getLogger(__name__)

class InvalidFrequencyType(Exception): pass
class NoTokenizerError(Exception): pass

class FrequencyCL(CurriculumDataset):
    # Valid types:
    #   1. char: Uses character level frequency.
    #   2. word: Uses word level frequency.
    #   3. token: Token level frequency (e.g. BPE).
    DEFAULT_FREQUENCY_TYPE = "char"
    VALID_FREQUENCY_TYPES = ['char', 'word', 'token']

    def __init__(self, 
      *args,
      frequency_type=None, 
      tokenizer=None, 
      precomputed_freqs=None, 
    ):
        self.frequency_type: str = frequency_type
        # NOTE: self.precomputed_freqs should change at every epoch
        #       when using a pacing function (since the frequencies of the
        #       characters will change. BUT, we may also provide the frequencies
        #       of the whole training set and avoid the re-calculation).
        self.precomputed_freqs: Dict[str, int] = precomputed_freqs or []
        self.tokenizer: Union[SentencePiece, FileListTokenizer] = tokenizer
        self._initialize()
        super().__init__(*args)

    def _initialize(self):
        self.calculate_frequency: Callable
        if self.frequency_type == "char":
            self.calculate_frequency = self._calculate_char_frequency
        elif self.frequency_type == "word":
            self.calculate_frequency = self._calculate_word_frequency
        elif self.frequency_type == "token":
            self.calculate_frequency = self._calculate_token_frequency
            if self.tokenizer is None:
                raise NoTokenizerError(f"The 'token' CL method requires you to provide a tokenizer.")
            assert isinstance(self.tokenizer, SentencePiece) or \
                isinstance(self.tokenizer, FileListTokenizer), "You should provide a valid tokenizer."
        else:
            raise InvalidFrequencyType(f"Invalid type of frequency calculator: {self.frequency_type}")

    def filtered_sorted(self,
        key_min_value: Optional[dict] = {},
        key_max_value: Optional[dict] ={},
        key_test: Optional[dict] = {},
        sort_key: Optional[str] = None,
        reverse: Optional[bool] = False,
        select_n: Optional[int] = None,
        sorting_dict: Optional[dict] = None,
        hparams: Optional[dict] = None,
        noise_percentage: Optional[float] = None,
    ):
        if sort_key in self.VALID_FREQUENCY_TYPES:
            filtered_sorted_ids = self._curriculum_filtered_ids(sorting_dict, reverse, select_n)
            if isinstance(noise_percentage, float) and 0.0 < noise_percentage <= 1.0:
                # logger.info(f"{filtered_sorted_ids[:10]=}")
                filtered_sorted_ids = CurriculumDataset.add_random_noise(filtered_sorted_ids, noise_percentage)
                logger.info("Added some random noise among the easy examples.")
                # logger.info(f"{filtered_sorted_ids[:10]=}")
            filtered_trainset = FilteredSortedFrequencyCL(
                self, filtered_sorted_ids,
                frequency_type=self.frequency_type,
                tokenizer=self.tokenizer,
                precomputed_freqs=self.precomputed_freqs,
            )
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
        sorting_dict = sorting_dict or self.sorting_dict
        select_n = select_n or (getattr(self, "current_epoch_n_datapoints", None) or len(self))
        sorting_dict = sorting_dict or {}
        if len(sorting_dict) == len(self) and len(sorting_dict) >= len(self.precomputed_freqs):
            self.precomputed_freqs = sorting_dict.copy()
            logger.warn("Using the sorting dictionary as precomputed freqs. This may lead to errors when\
                calculating the frequency (since the sorting dictionary contains tuples).")
        select_n = round(select_n)
        reverse_freqs = self.get_frequencies()
        filtered_sorted_ids: list = [utt_id for utt_id in sorted(
            reverse_freqs,
            key=lambda x: reverse_freqs[x],
            reverse=not reverse)
        ][:select_n]
        print(f"5 first examples: {filtered_sorted_ids[:5]}")
        print(f"5 last examples: {filtered_sorted_ids[-5:]}")
        if debug:
            # Make sure that the sorting was successfull (debugging).
            for i, j in zip(filtered_sorted_ids[:-1], filtered_sorted_ids[1:]):
                if not reverse:
                    assert sorting_dict[i]>=sorting_dict[j], f"i:{i}, j:{j}, di: {sorting_dict[i]}, dj: {sorting_dict[j]}" 
                else:
                    assert sorting_dict[i]<=sorting_dict[j], f"i:{i}, j:{j}, di: {sorting_dict[i]}, dj: {sorting_dict[j]}" 
            # Make sure that we have the valid batch ids
            for i, data_id in enumerate(self.data_ids):
                assert data_id in sorting_dict.keys(), f"Could not locate {data_id}."
        return filtered_sorted_ids

    def get_frequencies(self):
        if hasattr(self, 'reverse_freqs'):
            return self.reverse_freqs
        reverse_freqs = self.calculate_frequency()
        return utils.process_utils.normalize_dict(reverse_freqs)

    def get_scores(self):
        reverse_freqs = self.get_frequencies()
        max_freq = max(reverse_freqs.values())
        scores = {k: max_freq-v for k, v in reverse_freqs.items()}
        return scores

    @classmethod
    def from_csv(
        cls, csv_path, replacements={}, dynamic_items=[], output_keys=[], **kwargs
    ):
        """Load a data prep CSV file and create a Dataset based on it."""
        data = load_data_csv(csv_path, replacements)
        return cls(data, dynamic_items, output_keys, **kwargs)
        

    @staticmethod
    def clean_str(text):
        s = re.sub(r'[!-\(\)\"#\/@;:<>\{\}\-=~|\.\?]', ' ', text.lower()).strip()
        return re.sub("\s+", " ", s)
    
    def _calculate_char_frequency(self):
        # Higher score -> easier example
        reverse_sorting_dict = {}
        with self.output_keys_as(['wrd', 'id']) as dataset_trans:
            if len(self.precomputed_freqs) < len(dataset_trans):
                characters = ""
                for item in dataset_trans:
                    characters += self.clean_str(item['wrd'])
                self.precomputed_freqs = Counter(characters)
                self.precomputed_freqs.pop(' ')
            for item in dataset_trans:
                trans = self.clean_str(item['wrd']).replace(" ", "")
                easyness = sum([self.precomputed_freqs[char] for char in trans]) / len(trans)
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
                easyness = sum([self.precomputed_freqs[w] for w in words]) / len(words)
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
                easyness = sum([self.precomputed_freqs[w] for w in tokens]) / len(tokens)
                reverse_sorting_dict[item['id']] = easyness
        return reverse_sorting_dict
        
class FilteredSortedFrequencyCL(FrequencyCL):
    """Possibly filtered, possibly sorted DynamicItemDataset.

    Shares the static data (reference).
    Has its own dynamic_items and output_keys (deepcopy).
    """

    def __init__(self, 
      from_dataset, 
      data_ids, 
      frequency_type=None, 
      tokenizer=None, 
      precomputed_freqs=None
    ):
        self.frequency_type: str = frequency_type
        # NOTE: self.precomputed_freqs should change at every epoch
        #       when using a pacing function (since the frequencies of the
        #       characters will change. BUT, we may also provide the frequencies
        #       of the whole training set and avoid the re-calculation).
        self.precomputed_freqs: Dict[str, int] = precomputed_freqs
        self.tokenizer: Union[SentencePiece, FileListTokenizer] = tokenizer
        super()._initialize()
        self.data = from_dataset.data
        self.data_ids = data_ids
        self.pipeline = copy.deepcopy(from_dataset.pipeline)
