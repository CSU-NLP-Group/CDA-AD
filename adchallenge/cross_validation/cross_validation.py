import copy
import logging
import os
from typing import Any, Dict, List

from allennlp.commands.train import TrainModel
from allennlp.common import Lazy, util as common_util
from allennlp.data import DataLoader, DatasetReader, Vocabulary
from allennlp.models.model import Model
from allennlp.training import Trainer
from allennlp.training.metrics import Average
from overrides import overrides
from allennlp.training import util as training_util

logger = logging.getLogger(__name__)


@TrainModel.register("cross_validation")
class CrossValidateModel(TrainModel):
    def __init__(self, serialization_dir: str,
                 train_data_paths: List[str],
                 val_data_paths: List[str],
                 # test_data_path: str,
                 model: Lazy[Model],
                 dataset_reader: DatasetReader,
                 data_loader: Lazy[DataLoader],
                 trainer: Lazy[Trainer],
                 validation_dataset_reader: DatasetReader = None,
                 validation_data_loader: Lazy[DataLoader] = None,
                 vocabulary: Lazy[Vocabulary] = Lazy(Vocabulary),
                 evaluate_on_test:bool = False) -> None:

        super().__init__(serialization_dir, model=None, trainer=None)

        self.serialization_dir = serialization_dir
        self.train_data_paths = train_data_paths
        self.val_data_paths = val_data_paths
        # self.test_data_path = test_data_path

        self.evaluate_on_test = evaluate_on_test
        self.data_loader = data_loader
        self.validation_data_loader = validation_data_loader
        self.dataset_reader = dataset_reader
        self.validation_dataset_reader = validation_dataset_reader

        self.model = model
        self.trainer = trainer
        self.vocabulary = vocabulary

    @overrides
    def run(self) -> Dict[str, Any]:
        metrics_by_fold = []

        val_dataset_reader = self.validation_dataset_reader or self.dataset_reader
        val_data_loader = self.validation_data_loader or self.data_loader

        # test_data_loader = None
        # if self.test_data_path is not None and self.evaluate_on_test:
        #     test_data_loader = val_data_loader.construct(
        #         reader=val_dataset_reader, data_path=self.test_data_path
        #     )

        assert len(self.train_data_paths) == len(self.val_data_paths)
        num_folds = len(self.train_data_paths)
        model_, vocabulary_ = None, None

        for fold_index, (train_path, val_path) in enumerate(zip(
                self.train_data_paths, self.val_data_paths)):

            logger.info(f"---------------------Fold {fold_index}/{num_folds - 1}-------------------------")

            serialization_dir = os.path.join(self.serialization_dir, f"fold_{fold_index}")
            data_loaders: Dict[str, DataLoader] = {
                "train": self.data_loader.construct(
                    reader=self.dataset_reader, data_path=train_path),
                "test": val_data_loader.construct(
                    reader=val_dataset_reader, data_path=val_path
                )}

            if vocabulary_ is None:
                instance_generator = (
                    instance
                    for key, data_loader in data_loaders.items()
                    for instance in data_loader.iter_instances()
                )
                vocabulary_ = self.vocabulary.construct(instances=instance_generator)

            if model_ is None:
                model_ = self.model.construct(
                    vocab=vocabulary_, serialization_dir=serialization_dir
                )

            for data_loader_ in data_loaders.values():
                data_loader_.index_with(model_.vocab)
                # test_data_loader.index_with(model_.vocab)

            if common_util.is_global_primary():
                os.makedirs(serialization_dir, exist_ok=True)

            sub_model = copy.deepcopy(model_)
            sub_trainer = self.trainer.construct(
                serialization_dir=serialization_dir,
                model=sub_model,
                data_loader=data_loaders["train"],
                # validation_data_loader=data_loaders.get("validation"),
            )
            assert sub_trainer is not None

            fold_metrics = sub_trainer.train()

            test_data_loader = data_loaders.get("test", None)
            if test_data_loader and self.evaluate_on_test:
                for metric_key, metric_value in training_util.evaluate(
                    sub_model,
                    test_data_loader,
                    sub_trainer.cuda_device,
                ).items():
                    # if metric_key in fold_metrics:
                    fold_metrics[f"test_{metric_key}"] = metric_value
                    # else:
                    #     fold_metrics[metric_key] = metric_value

            if common_util.is_global_primary():
                common_util.dump_metrics(
                    os.path.join(sub_trainer._serialization_dir, "metrics.json"),
                    fold_metrics,
                    log=True,
                )

            metrics_by_fold.append(fold_metrics)

        metrics = {}

        for metric_key, fold_0_metric_value in metrics_by_fold[0].items():
            for fold_index, fold_metrics in enumerate(metrics_by_fold):
                metrics[f"fold{fold_index}_{metric_key}"] = fold_metrics[metric_key]
            if isinstance(fold_0_metric_value, float):
                average = Average()
                for fold_metrics in metrics_by_fold:
                    average(fold_metrics[metric_key])
                metrics[f"average_{metric_key}"] = average.get_metric()

        return metrics

    def finish(self, metrics: Dict[str, Any]) -> None:
        common_util.dump_metrics(
            os.path.join(self.serialization_dir, "metrics.json"), metrics, log=True
        )
