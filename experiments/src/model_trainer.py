"""Definition for a model trainer of multiple instance learning classifiers.

"""

import copy
import logging
import itertools
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
        if (classifier_only and
            isinstance(getattr(module, "classifier", None), nn.Module)):
            _params.append(module.classifier.parameters())
        if input_layer_only:
            _params.append(module.attention_module.get_first_layer())
        if (not input_layer_only) and (not classifier_only):
            _params.append(module.parameters())

        w = th.cat([x.view(-1) for x in itertools.chain.from_iterable(_params)])
        l1_term = strength * th.linalg.vector_norm(w, ord = 1)

        return l1_term

    def reset_weights(self, module: nn.Module):
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
        l1_strength = kwargs.get("l1_strength", 0.0)
        l1_classifier_only = kwargs.get("l1_classifier_only", False)
        l1_attention_in_layer_only = (
            kwargs.get('l1_attention_in_layer_only', False)
        )
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(on_device), y.to(on_device)
            y_hat = module.to(on_device)(x)

            # Expandable in future to various regularization terms
            reg_term = (
                self.compute_l1_term(
                    module, l1_strength, l1_classifier_only,
                    l1_attention_in_layer_only))

            # loss = logit_bce_loss(y_hat, y) + reg_term
            loss = loss_func(y_hat, y, **loss_kwargs) + reg_term
            loss = loss / batch_size
            loss.backward()
            epoch_loss += loss.item()

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
    ) -> float:
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
                [module.to(on_device)(vx.to(on_device)), y.to(on_device)]
                for (vx, y) in dataloader
            ]
            preds, labels = list(zip(*val_output))

        preds = th.cat(preds, dim = 0).squeeze(1).numpy(force = True)
        labs = th.cat(labels, dim = 0).squeeze(1).numpy(force = True)

        for (_metric, _binarize) in zip(metric, binarize):
            score_func = get_scorer(_metric)._score_func
            scores[_metric] = (
                score_func(
                    labs, # Labels
                    preds > 0 if _binarize else preds
                )
            ) # Binarize if needed

        return scores

    def fit(
            self,
            module: nn.Module,
            data_loader_train: DataLoader,
            optimizer: optim.Optimizer,
            loss_func: Callable,
            loss_kwargs: Optional[dict] = None,
            data_loader_val: Optional[DataLoader] = None,
            val_metric: Union[str, list] = 'balanced_accuracy',
            model_selection_method: str = "loss",
            num_epochs: int = 20,
            batch_size: int = 1,
            on_device: str = "cpu",
            binarize: bool = False,
            **kwargs
    ) -> list[float]:
        """Fit method in the spirit of sklearn with extra kwargs

        Args:
            module (nn.Module): Torch model being trained
            data_loader_train (DataLoader): Pytorch dataloader object which
                gives a 2D tensor (N x H, embedding space) or a batch of images
                (N x C x H x W, image space) and a bag label of 1x1.
            optimizer (optim.Optimizer): A PyTorch optimizer.
            loss_func (Callable): A callable loss function.
            loss_kwargs (dict): Keyword arguments for the loss function if the
                loss function takes any kwargs.
            val_metric (list[str]): A list of string names of sklearn metrics.
            model_selection_method (str): The metric (or 'loss') to use when
                selecting the best model.
            num_epochs (int): The number of epochs to train the model for.
            batch_size (int): The number of bags to include before updating
                model parameters.
            on_device (str): The device to train on.
            binarize (list[bool]): Must be the same length as
                model_selection_method. Whether to binarize the logit when
                calculating the metric.
        Returns:
            list[float]: The losses across different epochs.
        """
        # Check metric to decide best model
        if model_selection_method != "loss" and data_loader_val is None:
            model_selection_method = "loss"
        # Set up
        losses = []
        tracked_metric = np.inf if model_selection_method == "loss" else 0
        best_model = copy.deepcopy(module.state_dict())

        for epoch in range(num_epochs):
            epoch_loss = (
                self.train_iteration_with_grad_accumulation(
                    module, data_loader_train, optimizer,
                    loss_func, loss_kwargs,
                    batch_size, on_device, **kwargs))
            losses.append(epoch_loss)

            # Validate
            if data_loader_val is not None:
                score_values = (
                    self.validate(
                        module, data_loader_val, val_metric,
                        binarize = binarize, on_device = on_device))
                scores_kv = [
                    f"{_metric}: {_value :.5f}"
                    for _metric, _value in score_values.items()
                ]
                scores_msg = ", ".join(scores_kv)
                log.info(f"Epoch {epoch+1}, Validation: {scores_msg}")

            if model_selection_method == "loss" and epoch_loss < tracked_metric:
                log.info(
                    "\tUpdating model (min loss) %.5f -> %.5f",
                    tracked_metric,
                    epoch_loss
                )
                tracked_metric = epoch_loss
                best_model = copy.deepcopy(module.state_dict())
            elif (
                (not data_loader_val is None) # Validation set exists
                and (model_selection_method in score_values.keys())
            ):
                metric_value = score_values[model_selection_method]
                if metric_value > tracked_metric:
                    log.info(
                        "\tUpdating model (%s) %.5f -> %.5f",
                        model_selection_method,
                        tracked_metric,
                        metric_value
                    )
                    tracked_metric = metric_value
                    best_model = copy.deepcopy(module.state_dict())

        module.load_state_dict(best_model)

        return losses
