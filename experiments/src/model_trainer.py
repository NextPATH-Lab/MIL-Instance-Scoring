"""Definition for a model trainer of multiple instance learning classifiers.

"""
### === Dependencies === #
import copy, logging, itertools # >> Shorthand for Native Python Deps
from typing import Optional, Union, Callable

import numpy as np

import torch as th
from torch import nn, optim
from torch.utils.data import DataLoader

from sklearn.metrics import get_scorer

log = logging.getLogger(__name__)

class MILTrainer:
    def __init__(self):
        return

    def compute_weighted_correlation(
            self,
            corr_mx: th.Tensor,
            weight_params: th.Tensor,
            strength: float = 1e-3,
    ):
        """Deprecate...?
        Was an idea I had before to replace pre-filtering features."""

        w_interaction = th.outer(weight_params, weight_params)
        diag_mask = 1.0 - th.eye(corr_mx.size(0), device = corr_mx.device)

        corr_penalty = th.sum(corr_mx * w_interaction * diag_mask) / 2.0

        return corr_penalty * strength

    def compute_l1_term(
            self,
            module: nn.Module,
            strength: float = 1e-3,
            classifier_only: bool = False,
            input_layer_only: bool = False
    ) -> float:
        """Compute the L1 regularization term based on some parameters.

        Args:
            module (nn.Module): The model to compute L1 norm for
            strength (float, optional): Scaling factor for L1 norm.
                Defaults to 1e-3.
            classifier_only (bool, optional): Whether to take the L1 norm only 
                of the layer named `classifier`. Defaults to False.

        Returns:
            th.Tensor: Tensor scalar of L1 norm value.
        """
        _params = []
        if (
            classifier_only and
            isinstance(getattr(module, "classifier", None), nn.Module)
        ):
            _params.append(module.classifier.parameters())

        if input_layer_only:
            _params.append(module.attention_module.get_first_layer())

        if (not input_layer_only) and (not classifier_only):
            _params.append(module.parameters())

        w = th.cat([x.view(-1) for x in itertools.chain.from_iterable(_params)])
        l1_term = strength * th.linalg.vector_norm(w, ord = 1)
        return l1_term

    def reset_weights(self, module: nn.Module) -> None:
        """Resets all resettable layers.

        Args:
            module (nn.Module): Module to reset parameters of.
        """
        for layer in module.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        return

    def train_iteration_with_grad_accumulation(
            self,
            module: nn.Module,
            dataloader: DataLoader,
            optimizer: optim.Optimizer,
            loss_func: Callable,
            loss_kwargs: Optional[dict] = None,
            batch_size: int = 1,
            on_device: str = "cpu",
            **kwargs
    ) -> float:
        """_summary_

        Args:
            dataloader (DataLoader): _description_
            optimizer (optim.Optimizer): _description_
            batch_size (int, optional): _description_. Defaults to 1.
            on_device (str, optional): _description_. Defaults to "cpu".

        Returns:
            float: _description_
        """
        module.train()
        optimizer.zero_grad() # Zero gradients at start
        loss_kwargs = {} if loss_kwargs is None else loss_kwargs

        epoch_loss = 0.
        # >> Get L1-related keywords from **kwargs
        l1_strength = kwargs.get("l1_strength", 0.0)
        l1_classifier_only = kwargs.get("l1_classifier_only", False)
        l1_attention_in_layer_only = (
            kwargs.get('l1_attention_in_layer_only', False)
        )
            
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(on_device), y.to(on_device)
            # >> Forward pass
            y_hat = module.to(on_device)(x)

            # Expandable in future to various regularization terms
            reg_term = (
                self.compute_l1_term(
                    module,
                    l1_strength,
                    l1_classifier_only, # bool flag
                    l1_attention_in_layer_only # bool flag
                )
            )

            # >>> Compute Error, back-prop
            loss = loss_func(y_hat, y, **loss_kwargs) + reg_term
            loss = loss / batch_size
            loss.backward()
            epoch_loss += loss.item()

            # >>> Accumulate gradients
            if i % batch_size == 0 or i == (len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

        return epoch_loss

    def validate(
            self,
            module: nn.Module,
            dataloader: DataLoader,
            metric: Union[str, list[str]],
            binarize: Union[bool, list[bool]] = True,
            on_device: str = "cpu"
    ) -> dict:
        """Calculate a metric available from sklearn of the model performance.

        Args:
            module (nn.Module): The model being validated
            x (list[th.Tensor]): Input tensors
            y (list[th.Tensor]): Label for each bag
            metric (str): Any string name of a scoring metric supported by 
                sklearn.metric.get_scorer.

        Returns:
            float: The corresponding metric.
        """
        metric = [metric] if isinstance(metric, str) else metric
        binarize = [binarize] if isinstance(binarize, bool) else binarize

        scores = {}
        module.eval()
        with th.no_grad():
            val_output = [
                [
                    module.to(on_device)(vx.to(on_device)),
                    y.to(on_device)
                ]
                for (vx, y) in dataloader
            ]
            preds, labels = list(zip(*val_output))

        try:
            assert len(metric) == len(binarize)
        except AssertionError:
            raise ValueError(
                "Number of metrics passed did not match binarization count."
            )

        preds = th.cat(preds, dim = 0).squeeze(1).numpy(force = True)
        labs = th.cat(labels, dim = 0).squeeze(1).numpy(force = True)

        for (_metric, _binarize) in zip(metric, binarize):
            score_func = get_scorer(_metric)._score_func
            scores[_metric] = (
                score_func(
                    labs,
                    preds > 0 if _binarize else preds
                )
            )

        return scores

    def _metrics_improved(
            self,
            metrics: dict,
            prior_metrics: dict,
            selection_method: Optional[str | list[str]],
    ) -> bool:
        """Internal method to check if performance improved.

        The model training allows multiple metrics to be tracked for selecting
        the best performing model. This method handles multiple metrics to
        determine if performance has improved.

        Args:
            metrics (dict): The dictionary of metrics (str, key) and its values
                (float, value) of most recent validation run.
            prior_metrics (dict): The dictionary of metrics (str, key) and its
                values (float, value) of the prior best validation run.
            selection_method (str | list[str]): The list of metrics to look at
                to determine if performance has improved.
        Returns:
            bool.
        """

        if len(prior_metrics) == 0:
            # initialization case for the first epoch, when no metrics
            # have been calculated yet.
            return True

        selection_method = (
            [selection_method]
            if isinstance(selection_method, str)
            else selection_method
        )
        # >> If all metrics have improved (i.e. the number of improvements is
        # >> the same as the number of metrics), the performance has improved.
        # >>> NOTE: We assume a metric has 'improved' if it is equal to, or
        # >>> greater than the prior value. The assumption is that the loss
        # >>> would have decreased with the same validation metric value,
        # >>> although this is not strictly true in all cases.
        count = 0
        for _method in selection_method:
            if metrics[_method] >= prior_metrics[_method]:
                count += 1

        if count == len(selection_method):
            return True

        return False

    def fit(
            self,
            module: nn.Module,
            data_loader_train: DataLoader,
            optimizer: optim.Optimizer,
            loss_func: Callable,
            loss_kwargs: Optional[dict] = None,
            data_loader_val: Optional[DataLoader] = None,
            val_metric: Union[str, list] = 'balanced_accuracy',
            model_selection_method: str | list[str] = "loss",
            num_epochs: int = 20,
            batch_size: int = 1,
            on_device: str = "cpu",
            binarize: bool = False,
            **kwargs
    ) -> list[float]:
        """Fit method.

        Args:
            module (nn.Module): The torch.nn.Module object being optimized.
            data_loader_train (DataLoader): The DataLoader instance for the
                training data.
            optimizer (torch.optim.Optimizer): The optimizer for module.
            loss_func (Callable): Any callable loss function.
            loss_kwargs (Optional, dict): Keyword arguments for the loss func,
                if any.
            data_loader_val (DataLoader): The DataLoader instance for validation
                data, if any.
            val_metric (str | list[str]): The metric(s) to compute on the 
                validation dataset, if any.
            model_selection_method (str | list[str]): The metric(s) to use to
                evaluate model performance.
            batch_size (int): The number of bags to process and accumulate
                gradients before optimization step.
            on_device (str): Device on which to optimize module.
            binarize (bool | list[bool]): A list with the same length as
                val_metric. Whether module logits need to be binarized prior
                to computing each metric.

        """
        # Check metric to decide best model
        if model_selection_method != "loss" and data_loader_val is None:
            model_selection_method = "loss"
        # Set up
        losses = []
        tracked_metric = np.inf if model_selection_method == "loss" else 0
        best_model = copy.deepcopy(module.state_dict())
        best_epoch = 0
        best_metrics = {}
        use_val_metrics_for_model_selection = False

        for epoch in range(num_epochs):
            # >> Full pass of data
            epoch_loss = (
                self.train_iteration_with_grad_accumulation(
                    module,
                    data_loader_train,
                    optimizer,
                    loss_func,
                    loss_kwargs,
                    batch_size,
                    on_device,
                    **kwargs
                )
            )
            losses.append(epoch_loss)

            # >> Validate
            if data_loader_val is not None:
                score_values = (
                    self.validate(
                        module,
                        data_loader_val,
                        val_metric,
                        binarize = binarize,
                        on_device = on_device
                    )
                )
                scores_kv = [
                    f"{_metric}: {_value :.5f}"
                    for _metric, _value in score_values.items()
                ]
                scores_msg = ", ".join(scores_kv)
                log.info(f"Epoch {epoch+1}, Validation: {scores_msg}")
                if (
                    isinstance(model_selection_method, list)
                    or (
                        isinstance(model_selection_method, str)
                        and model_selection_method in score_values.keys()
                    )
                ):
                    use_val_metrics_for_model_selection = True

            if model_selection_method == "loss" and epoch_loss < tracked_metric:
                log.info(
                    "\tUpdating model (min loss) %.5f -> %.5f",
                    tracked_metric,
                    epoch_loss
                )
                tracked_metric = epoch_loss
                best_model = copy.deepcopy(module.state_dict())
                best_epoch = epoch
            elif use_val_metrics_for_model_selection:
                model_improved = (
                    self._metrics_improved(
                        score_values,
                        best_metrics,
                        model_selection_method
                    )
                )

                if model_improved:
                    log.info(
                        f"\tUpdating model as all metrics in "
                        f"{model_selection_method} improved"
                    )
                    best_metrics = score_values
                    best_model = copy.deepcopy(module.state_dict())
                    best_epoch = epoch

        module.load_state_dict(best_model)
        return losses, best_epoch
