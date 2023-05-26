import math
import random
from typing import Optional, List, Iterator, Iterable

import allennlp.nn.util as nn_util
import torch
from allennlp.common.util import lazy_groups_of, shuffle_iterable
from allennlp.data import BatchSampler
from allennlp.data.data_loaders.data_collator import DefaultDataCollator
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides


@DataLoader.register("instance")
class InstanceDataLoader(DataLoader):
    """
    A very simple `DataLoader` that is mostly used for testing.
    """

    def __init__(
        self,
        instances: List[Instance],
        batch_sampler: BatchSampler = None,
        *,
        batch_size: int = None,
        shuffle: bool = False,
        batches_per_epoch: Optional[int] = None,
        vocab: Optional[Vocabulary] = None,
    ) -> None:

        if batch_sampler is not None:
            if batch_size is not None:
                raise ValueError("batch_sampler option is mutually exclusive with batch_size")
            if shuffle:
                raise ValueError("batch_sampler option is mutually exclusive with shuffle")

        self.instances = instances
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_sampler = batch_sampler
        self.batches_per_epoch = batches_per_epoch
        self.vocab = vocab

        self.cuda_device: Optional[torch.device] = None
        self._batch_generator: Optional[Iterator[TensorDict]] = None
        self.collate_fn = DefaultDataCollator()


    def __len__(self) -> int:
        if self.batches_per_epoch is not None:
            return self.batches_per_epoch

        if self.batch_sampler is not None:
            return self.batch_sampler.get_num_batches(self.instances)
        return math.ceil(len(self.instances) / self.batch_size)

    @overrides
    def __iter__(self) -> Iterator[TensorDict]:
        if self.batches_per_epoch is None:
            yield from self._iter_batches()
        else:
            if self._batch_generator is None:
                self._batch_generator = self._iter_batches()
            for i in range(self.batches_per_epoch):
                try:
                    yield next(self._batch_generator)
                except StopIteration:  # data_generator is exhausted
                    self._batch_generator = self._iter_batches()  # so refresh it
                    yield next(self._batch_generator)

    def _iter_batches(self) -> Iterator[TensorDict]:
        if self.shuffle:
            random.shuffle(self.instances)

        if self.instances is not None:
            for batch in self._instance_to_batches(self.iter_instances(), move_to_device=True):
                yield  batch
        # for batch in lazy_groups_of(self.iter_instances(), self.batch_size):
        #     tensor_dict = self.collate_fn(batch)
        #     if self.cuda_device is not None:
        #         tensor_dict = nn_util.move_to_device(tensor_dict, self.cuda_device)
        #     yield tensor_dict

    def _instance_to_batches(self,
                             instance_iterator: Iterable[Instance], move_to_device:bool):

        if move_to_device and self.cuda_device is not None:
            tensorize = lambda batch: nn_util.move_to_device(  # noqa: E731
                self.collate_fn(batch), self.cuda_device
            )
        else:
            tensorize = self.collate_fn

        if self.batch_sampler is not None:

            instance_chunks = [list(instance_iterator)]
            for instances in instance_chunks:
                batches = (
                    [instances[i] for i in batch_indices]
                    for batch_indices in self.batch_sampler.get_batch_indices(instances)
                )
                for batch in batches:
                    yield tensorize(batch)
        else:
            # Safe to assume this is not `None` when `self.batch_sampler` is `None`.
            assert self.batch_size is not None

            if self.shuffle:

                # At this point we've already loaded the instances in memory and indexed them,
                # so this won't take long.
                instance_iterator = list(instance_iterator)
                random.shuffle(instance_iterator)

            for batch in lazy_groups_of(instance_iterator, self.batch_size):
                yield tensorize(batch)
    @overrides
    def iter_instances(self) -> Iterator[Instance]:
        for instance in self.instances:
            if self.vocab is not None:
                instance.index_fields(self.vocab)
            yield instance

    @overrides
    def index_with(self, vocab: Vocabulary) -> None:
        self.vocab = vocab
        for instance in self.instances:
            instance.index_fields(self.vocab)

    @overrides
    def set_target_device(self, device: torch.device) -> None:
        self.cuda_device = device