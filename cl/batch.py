import collections
import torch
from speechbrain.dataio.batch import (
    PaddedBatch, batch_pad_right, default_convert, PaddedData, 
    mod_default_collate, recursive_pin_memory, recursive_to
)


VADData = collections.namedtuple("VADData", ["data"])

# def padding_func(tensors: list, mode='constant', value=0):
#     if isinstance(tensors[0], VADData):
#         values_with_segments = []
#         for batch in tensors:
#             for segment in batch:
#                 assert isinstance(segment, torch.Tensor)
#                 values_with_segments.append(segment)
#         return batch_pad_right(values_with_segments, mode, value)
#     return batch_pad_right(tensors, mode, value)

class PaddedBatchVAD(PaddedBatch):
    # Worst OOP possible. TODO: Do the same without copy pasting everything from speechbrain's class.
    # The issue is that __<attr> attributes cannot be accessed since they will be given the name
    # PaddedBatch__<attr> but instead we would expect the name PaddedBatchVAD__<attr> and so we 
    # will get an AttributeError
    def __init__(
        self,
        examples,
        device="cuda:0",
        padded_keys=None,
        device_prep_keys=None,
        padding_func=batch_pad_right,
        padding_kwargs={},
        apply_default_convert=True,
        nonpadded_stack=True,
    ):
        self.__length = len(examples)
        self.__keys = list(examples[0].keys())
        self.__padded_keys = []
        self.__device_prep_keys = []
        for key in self.__keys:
            values = [example[key] for example in examples]
            # Default convert usually does the right thing (numpy2torch etc.)
            if apply_default_convert:
                values = default_convert(values)
            if (padded_keys is not None and key in padded_keys) or (
                padded_keys is None and isinstance(values[0], torch.Tensor)
            ):
                # Padding and PaddedData
                self.__padded_keys.append(key)
                padded = PaddedData(*padding_func(values, **padding_kwargs))
                setattr(self, key, padded)
            elif (padded_keys is None and isinstance(values[0], VADData)):
                self.__padded_keys.append(key)
                values_with_segments = []
                # print("WE ARE HERE")
                for batch in values:
                    for segment in batch.data:
                        assert isinstance(segment, torch.Tensor), f"{segment=}\n{values=}"
                        values_with_segments.append(segment.to(device))
                    # padded = PaddedData(*padding_func(batch.data.to(device), **padding_kwargs))
                    # setattr(self, key, padded)
                padded = PaddedData(*padding_func(values_with_segments, **padding_kwargs))
                # print("KEY:", key, "===== PADDED:", padded)
                setattr(self, key, padded)
            else:
                # Default PyTorch collate usually does the right thing
                # (convert lists of equal sized tensors to batch tensors, etc.)
                if nonpadded_stack:
                    values = mod_default_collate(values)
                setattr(self, key, values)
            if (device_prep_keys is not None and key in device_prep_keys) or (
                device_prep_keys is None and isinstance(values[0], torch.Tensor)
            ):
                self.__device_prep_keys.append(key)
            # print(f"{getattr(self, key)=}")

    def __len__(self):
        return self.__length

    def __getitem__(self, key):
        if key in self.__keys:
            return getattr(self, key)
        else:
            raise KeyError(f"Batch doesn't have key: {key}")

    def __iter__(self):
        """Iterates over the different elements of the batch.
        Example
        -------
        >>> batch = PaddedBatch([
        ...     {"id": "ex1", "val": torch.Tensor([1.])},
        ...     {"id": "ex2", "val": torch.Tensor([2., 1.])}])
        >>> ids, vals = batch
        >>> ids
        ['ex1', 'ex2']
        """
        return iter((getattr(self, key) for key in self.__keys))

    def pin_memory(self):
        """In-place, moves relevant elements to pinned memory."""
        for key in self.__device_prep_keys:
            value = getattr(self, key)
            pinned = recursive_pin_memory(value)
            setattr(self, key, pinned)
        return self

    def to(self, *args, **kwargs):
        """In-place move/cast relevant elements.
        Passes all arguments to torch.Tensor.to, see its documentation.
        """
        for key in self.__device_prep_keys:
            value = getattr(self, key)
            moved = recursive_to(value, *args, **kwargs)
            setattr(self, key, moved)
        return self

    def at_position(self, pos):
        """Fetch an item by its position in the batch."""
        key = self.__keys[pos]
        return getattr(self, key)

    @property
    def batchsize(self):
        return self.__length
