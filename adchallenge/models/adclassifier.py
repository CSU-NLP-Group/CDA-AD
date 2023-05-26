from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.fields import MetadataField
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import BooleanAccuracy, F1Measure
from overrides import overrides


class RDrop(nn.Module):
    """
    R-Drop for classification tasks.
    Example:
        criterion = RDrop()
        logits1 = model(input)  # model: a classification model instance. input: the input data
        logits2 = model(input)
        loss = criterion(logits1, logits2, target)     # target: the target labels. len(loss_) == batch size
    Notes: The model must contains `dropout`. The model predicts twice with the same input, and outputs logits1 and logits2.
    """
    def __init__(self, kl_weight: float = 1.):
        super(RDrop, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.kld = nn.KLDivLoss()
        self._kl_weight = kl_weight

    def forward(self, logits1, logits2, target):
        """
        Args:
            logits1: One output of the classification model.
            logits2: Another output of the classification model.
            target: The target labels.
            kl_weight: The weight for `kl_loss`.
        Returns:
            loss: Losses with the size of the batch size.
        """
        bce_loss = (self.bce(logits1, target) + self.bce(logits2, target)) / 2
        kl_loss1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        kl_loss2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        kl_loss = (kl_loss1 + kl_loss2) / 2
        loss = bce_loss + self._kl_weight * kl_loss
        return loss


@Model.register("ad_classifier")
class ADReSSClassifier(Model):
    """
    This `Model` implements a basic text classifier. After embedding the text into
    a text field, we will optionally encode the embeddings with a `Seq2SeqEncoder`. The
    resulting sequence is pooled using a `Seq2VecEncoder` and then passed to
    a linear classification layer, which projects into the label space. If a
    `Seq2SeqEncoder` is not provided, we will pass the embedded text directly to the
    `Seq2VecEncoder`.

    Registered as a `Model` with name "basic_classifier".

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the input text into a `TextField`
    seq2seq_encoder : `Seq2SeqEncoder`, optional (default=`None`)
        Optional Seq2Seq encoder layer for the input text.
    seq2vec_encoder : `Seq2VecEncoder`
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
        will pool its output. Otherwise, this encoder will operate directly on the output
        of the `text_field_embedder`.
    feedforward : `FeedForward`, optional, (default = `None`)
        An optional feedforward layer to apply after the seq2vec_encoder.
    dropout : `float`, optional (default = `None`)
        Dropout percentage to use.
    num_labels : `int`, optional (default = `None`)
        Number of labels to project to in classification layer. By default, the classification layer will
        project to the size of the vocabulary namespace corresponding to labels.
    namespace : `str`, optional (default = `"tokens"`)
        Vocabulary namespace corresponding to the input text. By default, we use the "tokens" namespace.
    label_namespace : `str`, optional (default = `"labels"`)
        Vocabulary namespace corresponding to labels. By default, we use the "labels" namespace.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    """

    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            seq2vec_encoder: Seq2VecEncoder,
            seq2seq_encoder: Seq2SeqEncoder = None,
            feedforward: Optional[FeedForward] = None,
            dropout: float = None,
            margin: float = 0.1,
            alpha: float = 0.5,
            kl_weight: float = 1.,#rdrop中设置kl_weight
            use_rdrop: bool = False,
            # num_labels: int = None,
            # label_namespace: str = "labels",
            # namespace: str = "tokens",
            initializer: InitializerApplicator = InitializerApplicator(),
            **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)

        self._alpha = alpha
        self._margin = margin
        self._use_rdrop = use_rdrop

        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        if feedforward is not None:
            self._classifier_input_dim = feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        # self._label_namespace = label_namespace
        # self._namespace = namespace

        # if num_labels:
        #     self._num_labels = num_labels
        # else:
        #     self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, 1)
        self._accuracy = BooleanAccuracy()
        self._recall = F1Measure(0)
        self._rdrop_loss = RDrop(kl_weight=kl_weight)
        self._loss = torch.nn.BCEWithLogitsLoss()
        # self._loss = torch.nn.MSELoss()
        initializer(self)

    def embed_text(self, tokens: TextFieldTensors, num_wrapping_dims: int = 0):

        embedded_text = self._text_field_embedder(tokens, num_wrapping_dims=num_wrapping_dims)
        mask = get_text_field_mask(tokens, num_wrapping_dims=num_wrapping_dims)

        if num_wrapping_dims > 0:
            embedded_text = TimeDistributed._reshape_tensor(embedded_text)
            mask = TimeDistributed._reshape_tensor(mask)

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        logits = self._classification_layer(embedded_text).view(-1)

        return logits

    @staticmethod
    def make_one_hot(probs):
        onehot = torch.zeros(probs.shape[0], 2,device=probs.device)
        onehot[probs > 0.5, 1] = 1
        onehot[probs <= 0.5, 0] = 1
        return onehot
    
    def forward(  # type: ignore
            self,
            tokens: TextFieldTensors,
            negatives: TextFieldTensors = None,
            label: torch.IntTensor = None,
            metadata: MetadataField = None,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        tokens : `TextFieldTensors`
            From a `TextField`
        label : `torch.IntTensor`, optional (default = `None`)
            From a `LabelField`

        # Returns

        An output dictionary consisting of:

            - `logits` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                unnormalized log probabilities of the label.
            - `probs` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                probabilities of the label.
            - `loss` : (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.
        """

        logits = self.embed_text(tokens)
        probs = torch.sigmoid(logits)

        loss = 0.

        if not self.training:
            loss += self._loss(logits, label.float())
        else:
            if negatives is not None:
                # (batch_size * num_negatives)
                neg_logits = self.embed_text(negatives, num_wrapping_dims=1)
                # (batch_size, num_negatives)
                neg_probs = torch.sigmoid(neg_logits.view(logits.size(0), -1))
                # margin + neg - pos
                distance = self._margin + neg_probs.mean(dim=-1) - probs
                loss += self._alpha * torch.mean(F.relu(distance))

            if self._use_rdrop:
                logits2 = self.embed_text(tokens)
                loss += self._rdrop_loss(logits, logits2, label.float())
            else:
                loss += self._loss(logits, label.float())


        output_dict = {"logits": logits, "probs": probs, "loss": loss,
                       "token_ids": util.get_token_ids_from_text_field_tensors(tokens)}

        self._accuracy((probs > 0.5).long(), label)
        self._recall(ADReSSClassifier.make_one_hot(probs),label)
        return output_dict

    @overrides
    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        # predictions = output_dict["probs"]
        # if predictions.dim() == 2:
        #     predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        # else:
        #     predictions_list = [predictions]
        # classes = []
        # for prediction in predictions_list:
        # label_idx = prediction.argmax(dim=-1).item()

        # label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
        #     label_idx, str(label_idx)
        # )
        # classes.append(label_str)
        # output_dict["label"] = classes
        # tokens = []
        # for instance_tokens in output_dict["token_ids"]:
        #     tokens.append(
        #         [
        #             self.vocab.get_token_from_index(token_id.item(), namespace=self._namespace)
        #             for token_id in instance_tokens
        #         ]
        #     )
        # output_dict["tokens"] = tokens
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset),
            "recall": self._recall.get_metric(reset)["recall"]}
        return metrics

    # default_predictor = "text_classifier"
