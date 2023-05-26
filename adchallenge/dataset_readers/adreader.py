import json
import logging
import csv
from random import random

from typing import Optional, Iterable, Dict, List

from allennlp.common.file_utils import cached_path

from allennlp.data import DatasetReader, Tokenizer, TokenIndexer, Instance, Field
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField, LabelField, ListField

logger = logging.getLogger(__name__)


def get_negative_samples(passage: str, drop_rate: float = 0.3, num_negatives: int = 5):
    negatives = [["" if random() < drop_rate else token for token in passage.split()]
                 for _ in range(num_negatives)]
    return negatives


@DatasetReader.register("adreader")
class ADReSSDatasetReader(DatasetReader):
    """
    This DatasetReader returns a dataset of instances with the
    following fields:
    The output of `read` is a list of `Instance` s with the fields:
        tokens : `TextField` and
        label : `LabelField`
    # Parameters
    tokenizer: `Tokenizer`, optional (default=`WhitespaceTokenizer()`)
        Tokenizer to use to split the input sequences into words or other kinds of tokens.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    """

    def __init__(
            self, tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            num_aug: int = 1,
            num_negatives: int = 5,
            delete_rate: float = 0.3,
            is_training: bool = True,
            **kwargs
    ):
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )

        self.num_negatives = num_negatives
        self.num_aug = num_aug
        self.delete_rate = delete_rate
        self.is_training = is_training
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.label_func = {"Control": 1, "Alzheimer": 0}

    def _read(self, file_path) -> Iterable[Instance]:
        file_path = cached_path(file_path, extract_archive=True)
        with open(file_path) as f:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for row in self.shard_iterable(csv.reader(f)):
                passage, label = row[1], self.label_func.get(row[-1])

                if self.is_training and self.num_aug > 1:
                    for _ in range(self.num_aug):
                        yield self.text_to_instance(
                            passage=passage, label=label
                            # label=self.label_func.get(row[-1]),
                        )
                else:
                    yield self.text_to_instance(
                        passage=passage, label=label
                    )

    def text_to_instance(  # type: ignore
            self, passage: str, label: int = None
    ) -> Instance:
        """
        We take the passage as input, tokenize and concat them.
        # Parameters
        passage : `str`, required.
        question : `str`, required.
        label : `bool`, optional, (default = `None`).
        # Returns
        An `Instance` containing the following fields:
            tokens : `TextField`
            label : `LabelField`
        """
        fields: Dict[str, Field] = {}

        # 80% of the question length in the training set is less than 60, 512 - 4 - 60 = 448.
        passage_tokens = self.tokenizer.tokenize(passage)
        text_field = TextField(passage_tokens)
        fields["tokens"] = text_field

        if self.is_training and self.num_negatives > 0:
            neg_passages = get_negative_samples(passage, drop_rate=self.delete_rate,
                                                num_negatives=self.num_negatives)
            neg_text_field = ListField([
                TextField(self.tokenizer.tokenize(" ".join(np))) for np in neg_passages])
            fields["negatives"] = neg_text_field

        if label is not None:
            label_field = LabelField(int(label), skip_indexing=True)
            fields["label"] = label_field
        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["tokens"]._token_indexers = self.token_indexers  # type: ignore

        if self.is_training:
            for text_field in instance.fields["negatives"]:
                text_field._token_indexers = self.token_indexers
