#!/usr/bin/python
from re import T
from typing import Dict, List
import logging
import time
import random
from abc import ABC, abstractmethod
import speechbrain as sb
import torch
from speechbrain.dataio.dataset import (
    DynamicItemDataset, 
    FilteredSortedDynamicItemDataset
)
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.metric_stats import wer_details_for_batch
from speechbrain.dataio.dataio import load_pkl, save_pkl
from . import utils


strip_spaces = lambda s: utils.strip_spaces(s)


logger = logging.getLogger(__name__)


class BaseCurriculum(ABC):
    def __init__(self, sorting_method, brain, sorting_dict=None, **kwargs):
        self.brain = brain
        self.sorting_method = sorting_method
        # Try loading the sorting dictionary if nothing was provided
        assert isinstance(sorting_dict, dict)
        self.sorting_dict = sorting_dict
        self._is_normalized = False
    
    @abstractmethod
    def _process_batch(self, batch, step=None):
        # Must be implemented by the subclasses
        return
    
    def is_fully_loaded(self, train_set):
        if self.sorting_dict.get('num_datapoints') == len(train_set):
            logger.info(strip_spaces(f"Seems like the curriculum dictionary is fully loaded. \
                Length of dictionary: {self.sorting_dict.get('num_datapoints')}, \
                    Trainset Length: {len(train_set)}."))
            return True
        return False
    
    def min_max_normalize(self, epsilon=1e-3):
        if self._is_normalized is True:
            return
        if isinstance(list(self.sorting_dict.values())[0], tuple):
            self.sorting_dict = utils.normalize_with_confs(self.sorting_dict, epsilon)
        else:
            self.sorting_dict = utils.normalize_dict(self.sorting_dict)
        self._is_normalized = True

    @staticmethod
    def update_average(new_val, avg, step=None):
        """Update running average of the loss and/or wer/cer.
        Arguments
        ---------
        new_val : torch.tensor
            detached tensor, a single float value (either loss or wer/cer).
        avg : float
            current running average.
        Returns
        -------
        avg : float
            The updated average loss or wer/cer.
        """
        if torch.isfinite(new_val) and step is not None:
            avg -= avg / step
            avg += float(new_val) / step
        return avg


class MetricCurriculum(BaseCurriculum):
    def __init__(self, sorting_method, brain, sorting_dict=None, keep_confs=False, **kwargs):
        super(MetricCurriculum, self).__init__(sorting_method, brain, sorting_dict, **kwargs)
        if self.sorting_method == 'cer':
            # convert the sequence of words to a sequence of characters.
            # this way we will get CER instead of WER
            self.postprocess = lambda word_seq: [list('_'.join(w)) for w in word_seq]
        elif self.sorting_method == 'wer':
            # Do nothing
            self.postprocess = lambda word_seq: word_seq
        else:
            raise Exception("This sorting method has not been implemented yet:", self.sorting_method)
        if keep_confs is True:
            # This is useful in cases where we want to use the confidence
            # scores as an extra criterion for our sorting.
            # The confidences are normalized by the duration of the audios.
            # There is no option to use the raw confidences.
            logger.info("Scores will be normalized by confidences.")
            self.get_metric = lambda wer, conf, dur: (wer, conf, dur)
        else:
            # No reason to return the confidences since they won't be used for
            # the final sorting
            self.get_metric = lambda wer, conf, dur: wer
    
    def _process_batch(self, batch, avg, step=None):
        if batch.id[0] in self.sorting_dict:
            return
        predictions: list = self.brain.compute_forward(batch, sb.Stage.VALID)
        predicted_tokens = [h[0] for h in predictions[2]]
        tokens, tokens_lens = batch.tokens
        # Decode token terms to words
        predicted_words: list = self.postprocess(self.brain.tokenizer(
            predicted_tokens, task="decode_from_list"
        ))
        # Convert indices to words
        target_words = undo_padding(tokens, tokens_lens)
        target_words = self.postprocess(self.brain.tokenizer(target_words, task="decode_from_list"))
        details: List[dict] = wer_details_for_batch(batch.id, target_words, predicted_words, True)
        # t.set_postfix(
        #     wer=f"[{', '.join([str(round(det['WER'], 1)) for det in details])}]",
        #     confidences=f"[{', '.join([str(round(p.item(), 1)) for p in predictions[3]])}]",
        # )
        loss_equivalent = torch.sum(torch.Tensor([details[dp_id]['WER'] / tokens_lens[dp_id] for dp_id in range(len(batch.id))]))
        avg = self.update_average(loss_equivalent, avg, step=step)
        for dp_id in range(len(batch.id)):
            # `predictions[3]` denotes the confidence scores returned by `compute_forward`.
            self.sorting_dict[batch.id[dp_id]] = self.get_metric(
                round(details[dp_id]['WER'],4), 
                round(predictions[3][dp_id].cpu().item(), 4),
                batch.duration[dp_id].cpu().item()
            )
            logger.debug(
                f"id: {batch.id[dp_id]} - {self.sorting_method}: "
                f"{self.sorting_dict[batch.id[dp_id]]}"
            )
        return avg

class LossCurriculum(BaseCurriculum):
    def __init__(self, sorting_method, brain, sorting_dict=None, **kwargs):
        super(LossCurriculum, self).__init__(sorting_method, brain, sorting_dict, **kwargs)
        try:
            self.method = getattr(brain, f"_compute_{self.sorting_method}")
        except AttributeError:
            raise ValueError(f"""Invalid sorting method: {self.sorting_method}. 
We could not locate a brain method with the name: _compute_{self.sorting_method}""")
        if kwargs.get('keep_durs', False) is True:
            self.get_loss = lambda loss, dur: (loss, dur)  # return a tuple and keep dur
        else:
            self.get_loss = lambda loss, dur: loss  # keep a single value
    
    def _mean_reduction(self, losses: torch.Tensor, batch):
        target_lens = batch.tokens[1]
        losses /= target_lens
        return torch.mean(losses).detach()
    
    def _process_batch(self, batch, avg, step):
        if batch.id[0] in self.sorting_dict:
            return
        # Get losses for each datapoint in the batch
        # `predictions` is a tuple of a certain length (3 in our case)
        # Its content are: p_ctc, p_seq, wav_lens
        # The shape of p_seq is: (bs, N, output_neurons)
        predictions = self.brain.compute_forward(batch, sb.Stage.TRAIN)
        # Get losses for each datapoint in the batch
        loss = self.method(predictions, batch, stage=sb.Stage.TRAIN, reduction="batch").detach()
        # 'id' must also be an output key
        # batch.id[0] since we only have one datapoint here.
        avg = self.update_average(new_val=self._mean_reduction(loss, batch), avg=avg, step=step)
        # self.n_samples += len(batch.id)
        for dp_id in range(len(batch.id)):
            self.sorting_dict[batch.id[dp_id]] = self.get_loss(
                round(loss[dp_id].item(), 4), 
                batch.duration[dp_id].cpu().item()
            )
            logger.debug(
                f"id: {batch.id[dp_id]} - loss: {loss[dp_id].item()} "
                f"- duration: {batch.duration[dp_id]}"
            )
        del loss
        return avg

class JointCurriculum:
    def __init__(self, loss_method, metric_method, brain, sorting_dict=None, metric_prob=0.5, epsilon=1e-3, **kwargs):
        """ Use a mix of LossCurriculuma and MetricCurriculum.
            Given a probability (`metric_prob`) this class will either 
            calculate a score based on the metric method or based on the loss method.
            This method does not make the processing slower or more memory-hungry
            since the dictionary and the brain objects are shared across the loss
            curriculum and metric curriculum instances.
        """
        self.loss_curriculum = LossCurriculum(loss_method, brain, sorting_dict, **kwargs)
        self.metric_curriculum = MetricCurriculum(metric_method, brain, sorting_dict, **kwargs)
        self.epsilon = epsilon
        self.metric_prob = metric_prob
        self._inner_sort_dict = None
        self.is_normalized = False
        # TODO
        logger.error(f"""Loss curriculum and Metric curriculum output values in different ranges 
which means that when we will get to the point of sorting the whole dictionary, the whole sorting
will be pointless. To solve this make sure that both curriculum methods output values in the same
range (after normalization).""")
        raise
    
    @property
    def running_average(self):
        return self.loss_curriculum.running_average
    
    @property
    def sorting_dict(self):  # will also normalize the dataset
        if self.is_normalized is not True:
            self._normalize()
        if self._inner_sort_dict is None:
            # Concatenate the two dictionaries
            # NOTE: Using 2x memory here!!!
            self._inner_sort_dict = self.loss_curriculum.sorting_dict.copy()
            self._inner_sort_dict.update(self.metric_curriculum.sorting_dict)
        assert len(self._inner_sort_dict) == (len(self.loss_curriculum.sorting_dict) + len(self.metric_curriculum.sorting_dict))
        return self._inner_sort_dict
    
    def _normalize(self):
        # Make sure both dictionaries are in the range 0-1
        logger.info("Normalizing loss and metric based curriculum dictionaries to the same range.")
        self.loss_curriculum.min_max_normalize(self.epsilon)
        self.metric_curriculum.min_max_normalize(self.epsilon)
        self.is_normalized = True
    
    def is_fully_loaded(self, train_set):
        if self._inner_sort_dict.get('num_datapoints', len(self._inner_sort_dict)) == len(train_set):
            logger.info(strip_spaces(f"Seems like the curriculum dictionary is fully loaded. \
                Length of dictionary: {self.sorting_dict.get('num_datapoints')}, \
                    Trainset Length: {len(train_set)}."))
            return True
        return False
    
    def _process_batch(self, batch, avg, step):
        if random.random() <= self.metric_prob:
            return self.metric_curriculum._process_batch(batch, avg, step)
        else:
            return self.loss_curriculum._process_batch(batch, avg, step)
