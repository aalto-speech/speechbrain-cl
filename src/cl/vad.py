import os
import tempfile
import logging
import torch
import torchaudio
import speechbrain as sb
from speechbrain.pretrained import VAD
from cl.batch import VADData

logger = logging.getLogger(__name__)

# savedir = "../VAD/results/pretrained_hf/"

# VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir=savedir)

def _read_and_resample(audio_path, new_sr, sr=None):
    if sr is None:
        info = torchaudio.info(audio_path)
        sr = info.sample_rate
    sig = sb.dataio.dataio.read_audio(audio_path)
    if sr != new_sr:
        sig = torchaudio.transforms.Resample(sr, new_sr)(sig)
    return sig


# TODO: remove code repetition
def testset_pipeline_with_force_segments(
    test_set, tokenizer, sample_rate=16000,
    max_duration_secs=10.0, min_duration_secs=1.0,
    overlap_duration_secs=0.05, 
    bos_index=0, eos_index=0,
):
    min_duration_secs = max(0., min_duration_secs)
    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def test_audio_pipeline(wav, duration):
        if duration <= max_duration_secs:
            return _read_and_resample(wav, sample_rate)
        info = torchaudio.info(wav)
        sr = info.sample_rate
        # print(f"MY: {sr=} while vad's: {vad.sample_rate=}, {wav=}")
        delete_wav_after = False
        if sr != sample_rate:
            # print(f"Found different sample rate: {sr}")
            # Unfortuantely you can't pass an audio signal
            # to VAD.get_speech_segments. This means that
            # we need to save the audio signal to a temporary
            # and then delete it.
            resampled = _read_and_resample(wav, new_sr=sample_rate, sr=sr)
            wav = os.path.join(tempfile.gettempdir(), os.path.basename(wav))
            torchaudio.save(wav, resampled.reshape(1, -1), sample_rate)
            # print("saved audio at: ", wav)
            delete_wav_after = True
        sig = []
        n_clean_segments = int(duration//max_duration_secs)
        for seg_id in range(n_clean_segments):
            beginning = max(0.0, seg_id*max_duration_secs - overlap_duration_secs)
            ending = min(duration, seg_id*max_duration_secs + max_duration_secs)
            if ending - beginning <= min_duration_secs:
                continue
            segmented_signal = sb.dataio.dataio.read_audio({
                'file': wav,
                'start': int(beginning * sample_rate),
                'stop': int(ending * sample_rate),
            })
            # print("processed start=", beginning, ", end=", ending)
            sig.append(segmented_signal)
        # Handle the remainder seconds
        final_diff = duration - ending
        if final_diff > 0:
            if final_diff <= min_duration_secs:
                # append the last small part to the final segment
                segmented_signal = sb.dataio.dataio.read_audio({
                    'file': wav,
                    'start': int(beginning * sample_rate),
                    'stop': int(duration * sample_rate),  # add whatever is left
                })
                # change the previously last segmetn
                sig[-1] = segmented_signal
            else:
                # add a new segemnt
                segmented_signal = sb.dataio.dataio.read_audio({
                    'file': wav,
                    'start': int((ending-overlap_duration_secs) * sample_rate),
                    'stop': int(duration * sample_rate),
                })
                sig.append(segmented_signal)
            
        if delete_wav_after:
            os.remove(wav)
        # print(f"SIG BEFORE: {sig=}")
        return VADData(sig)
    sb.dataio.dataset.add_dynamic_item([test_set], test_audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def test_text_pipeline(wrd):
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([bos_index] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [eos_index])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
    sb.dataio.dataset.add_dynamic_item([test_set], test_text_pipeline)
    sb.dataio.dataset.set_output_keys(
        [test_set], ["id", "sig", "tokens_bos", "tokens_eos", "tokens"],
    )

    return test_set


def testset_pipeline_with_segments(
    test_set, vad, tokenizer, sample_rate=16000, 
    max_duration_secs=20.0, min_duration_secs=1.0,
    bos_index=0, eos_index=0,
):
    assert isinstance(vad, VAD), vad
    assert sample_rate == vad.sample_rate, f"{sample_rate=} === {vad.sample_rate=}"

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def test_audio_pipeline(wav, duration):
        if duration <= max_duration_secs:
            return _read_and_resample(wav, sample_rate)
        info = torchaudio.info(wav)
        sr = info.sample_rate
        # print(f"MY: {sr=} while vad's: {vad.sample_rate=}, {wav=}")
        delete_wav_after = False
        if sr != vad.sample_rate:
            # print(f"Found different sample rate: {sr}")
            # Unfortuantely you can't pass an audio signal
            # to VAD.get_speech_segments. This means that
            # we need to save the audio signal to a temporary
            # and then delete it.
            resampled = _read_and_resample(wav, vad.sample_rate, sr=sr)
            wav = os.path.join(tempfile.gettempdir(), os.path.basename(wav))
            torchaudio.save(wav, resampled.reshape(1, -1), vad.sample_rate)
            # print("saved audio at: ", wav)
            delete_wav_after = True

        # TODO: This may return multiple 'signals' for each segment.
        #       But then the batch size would not be fixed.
        #       Could lead to unexpected out-of-memory errors.
        #       Counterpoint: Even if we get an OOM error we wouldn't 
        #       really lose much since the training has already finished 
        #       at this stage. We would just have to re-decode the testset.
        # print("Using wav:", wav)
        segments = vad.get_speech_segments(wav)
        if len(segments) == 1:
            return _read_and_resample(wav, sample_rate)
        sig = []
        for beginning, ending in segments:
            if ending - beginning <= min_duration_secs:
                continue
            segmented_signal = sb.dataio.dataio.read_audio({
                'file': wav,
                'start': int(beginning * vad.sample_rate),
                'stop': int(ending * vad.sample_rate),
            })
            # print(f"Shape of segmented signal: {segmented_signal.shape}")
            sig.append(segmented_signal)
            
        if delete_wav_after:
            os.remove(wav)
        # print(f"SIG BEFORE: {sig=}")
        return VADData(sig)
    sb.dataio.dataset.add_dynamic_item([test_set], test_audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def test_text_pipeline(wrd):
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([bos_index] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [eos_index])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
    sb.dataio.dataset.add_dynamic_item([test_set], test_text_pipeline)
    sb.dataio.dataset.set_output_keys(
        [test_set], ["id", "sig", "tokens_bos", "tokens_eos", "tokens"],
    )

    return test_set
    