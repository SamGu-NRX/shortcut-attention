"""Integration-style tests for Mammoth model wrappers using real adapters."""

import torch
import torch.nn as nn

from models.gpm_model import GPMModel
from models.dgr_model import DGRModel

from tests.models.conftest import (
    DEVICE,
    TinyDataset,
    sample_batch,
    run_gpm_steps,
    run_dgr_steps,
)


def test_gpm_model_updates_memory(tiny_args, tiny_components):
    backbone, loss, transform, dataset = tiny_components
    model = GPMModel(backbone, loss, tiny_args, transform, dataset).to(DEVICE)

    model.begin_task(dataset)
    run_gpm_steps(model, dataset, steps=3)
    model.end_task(dataset)

    # The wrapper should now contain learned bases for at least one layer
    assert len(model.feature_list) > 0
    assert all(feat.shape[1] > 0 for feat in model.feature_list if feat.size > 0)


def test_gpm_model_observe_returns_float(tiny_args, tiny_components):
    backbone, loss, transform, dataset = tiny_components
    model = GPMModel(backbone, loss, tiny_args, transform, dataset).to(DEVICE)

    model.begin_task(dataset)
    inputs, labels = sample_batch(dataset)
    loss_value = model.observe(inputs, labels, inputs)

    assert isinstance(loss_value, float)
    assert not torch.isnan(torch.tensor(loss_value))


def test_dgr_model_task_lifecycle(tiny_args, tiny_components):
    backbone, loss, transform, dataset = tiny_components
    model = DGRModel(backbone, loss, tiny_args, transform, dataset).to(DEVICE)

    assert (model.image_channels, model.image_height, model.image_width) == tuple(dataset.SIZE)

    model.begin_task(dataset)
    run_dgr_steps(model, dataset, steps=2)
    model.end_task(dataset)

    assert model.prev_classifier is not None
    assert model.prev_generator is not None

    model.begin_task(dataset)
    inputs, labels = sample_batch(dataset)
    loss_value = model.observe(inputs, labels, inputs)
    assert isinstance(loss_value, float)


def test_dgr_model_forward(tiny_args, tiny_components):
    backbone, loss, transform, dataset = tiny_components
    model = DGRModel(backbone, loss, tiny_args, transform, dataset).to(DEVICE)

    inputs, _ = sample_batch(dataset)
    outputs = model.forward(inputs)
    assert outputs.shape[0] == inputs.shape[0]
    assert outputs.shape[1] == dataset.N_CLASSES
