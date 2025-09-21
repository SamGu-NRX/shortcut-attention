"""Sanity checks for Mammoth model wrappers using the shared tiny fixtures."""

import torch

from models.gpm_model import GPMModel
from models.dgr_model import DGRModel

from .conftest import DEVICE, sample_batch


def test_gpm_model_forward(tiny_args, tiny_components):
    backbone, loss, transform, dataset = tiny_components
    model = GPMModel(backbone, loss, tiny_args, transform, dataset).to(DEVICE)
    model.begin_task(dataset)

    inputs, _ = sample_batch(dataset)
    outputs = model.forward(inputs)

    assert outputs.shape[0] == inputs.shape[0]
    assert outputs.shape[1] == dataset.N_CLASSES


def test_dgr_model_observe_returns_float(tiny_args, tiny_components):
    backbone, loss, transform, dataset = tiny_components
    model = DGRModel(backbone, loss, tiny_args, transform, dataset).to(DEVICE)

    model.begin_task(dataset)
    inputs, labels = sample_batch(dataset)
    loss_value = model.observe(inputs, labels, inputs)

    assert isinstance(loss_value, float)


def test_dgr_model_caches_previous_generator(tiny_args, tiny_components):
    backbone, loss, transform, dataset = tiny_components
    model = DGRModel(backbone, loss, tiny_args, transform, dataset).to(DEVICE)

    model.begin_task(dataset)
    inputs, labels = sample_batch(dataset)
    model.observe(inputs, labels, inputs)
    model.end_task(dataset)

    assert model.prev_generator is not None
    assert model.prev_classifier is not None
