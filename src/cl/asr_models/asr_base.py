#!/usr/bin/env python3
import re
import torch
import logging
import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from cl.base_asr_model import BaseASR


logger = logging.getLogger(__name__)

# Define training procedure
class ASR_Old(BaseASR):
    def __init__(self, modules=None, opt_class=None, hparams=None, run_opts=None, 
                 checkpointer=None, sorting=None, train_set=None, tokenizer=None,
                 train_loader_kwargs=None, *args, **kwargs):
        super(ASR, self).__init__(
            modules=modules,
            opt_class=opt_class,
            hparams=hparams, 
            run_opts=run_opts, 
            checkpointer=checkpointer,
            sorting=sorting,
            train_set=train_set,
            train_loader_kwargs=train_loader_kwargs,
            *args, **kwargs,
        )
        self.tokenizer = tokenizer
    
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        ## Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        x = self.modules.enc(feats.detach())
        e_in = self.modules.emb(tokens_bos)  # y_in bos + tokens
        h, _ = self.modules.dec(e_in, x, wav_lens)
        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                # Output layer for ctc log-probabilities
                logits = self.modules.ctc_lin(x)
                p_ctc = self.hparams.log_softmax(logits)
                out = p_seq, p_ctc, wav_lens
            else:
                out = p_seq, wav_lens
            # For metric-based curriculum we also need to decode in order to later get
            # the predicted tokens and calculate wer/cer.
            if self.sorting in getattr(self.train_set, "METRIC_SORTERS", []):
                p_tokens, scores = self.hparams.beam_searcher(x, wav_lens)
                out += (p_tokens, scores)
            return out
        else:
            p_tokens, scores = self.hparams.beam_searcher(x, wav_lens)
            return p_seq, wav_lens, p_tokens, scores

    def compute_objectives(
        self,
        predictions,
        batch,
        stage,
    ):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        if stage != sb.Stage.TRAIN:
            # p_seq, wav_lens, predicted_tokens = predictions
            # Needed in beam searcher
            # predicted_tokens = [h[0] for h in predicted_tokens]
            predicted_tokens = [h[0] for h in predictions[2]]

        ids = batch.id

        loss = self.compute_loss(
            predictions, 
            batch, 
            stage=stage, 
            reduction="mean", 
            weight=self.hparams.ctc_weight
        )

        if stage != sb.Stage.TRAIN:
            tokens, tokens_lens = batch.tokens
            # Decode token terms to words
            # logging.info(f"predicted tokens: {predicted_tokens}")
            predicted_words = self.tokenizer(
                predicted_tokens, task="decode_from_list"
            )

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")
            # Process predictions and truth so that they don't contain special tokens (.br, .fr etc)
            predicted_words = [re.sub("\.\w+|-", "", ' '.join(txt)).strip().split() for txt in predicted_words]
            target_words = [re.sub("\.\w+|-", "", ' '.join(txt)).strip().split() for txt in target_words]
            # import random
            # if random.random() > 0.99:
            #     print("  preds-truth pairs:", list(zip(predicted_words, target_words))[:1])

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_valid_test_stage_start(self, stage):
        """Gets called before validation or testing"""
        assert stage != sb.Stage.TRAIN
        self.cer_metric = self.hparams.cer_computer()
        self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        super().on_stage_end(stage, stage_loss, epoch)
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)
            if hasattr(self.hparams, 'cer_file'):
                with open(self.hparams.cer_file, "w") as c:
                    self.cer_metric.write_stats(c)

