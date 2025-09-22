import pytest
import torch
from argparse import Namespace

from models.scratch_t2 import ScratchT2


def test_scratch_t2_name_and_compatibility():
    """Test that Scratch_T2 has correct name and compatibility."""
    assert ScratchT2.NAME == "scratch_t2"
    assert "class-il" in ScratchT2.COMPATIBILITY
    assert "domain-il" in ScratchT2.COMPATIBILITY
    assert "task-il" in ScratchT2.COMPATIBILITY


def test_scratch_t2_initialization(tiny_components, tiny_args):
    """Test that Scratch_T2 can be initialized properly."""
    backbone, loss, transform, dataset = tiny_components
    args = tiny_args
    args.n_epochs = 1
    args.debug_mode = True

    # Test initialization
    model = ScratchT2(backbone, loss, args, transform, dataset=dataset)

    assert model.NAME == "scratch_t2"
    assert model.task2_data is None
    assert model.task2_labels is None


def test_scratch_t2_task_handling(tiny_components, tiny_args):
    """Test that Scratch_T2 correctly handles task progression."""
    backbone, loss, transform, dataset = tiny_components
    args = tiny_args
    args.n_epochs = 1
    args.debug_mode = True

    model = ScratchT2(backbone, loss, args, transform, dataset=dataset)

    # Mock dataset for Task 1 (should be skipped)
    class MockDataset:
        def __init__(self, task_id):
            self.i = task_id
            self.train_loader = []

    # Test Task 1 (should be skipped)
    task1_dataset = MockDataset(0)
    model.begin_task(task1_dataset)
    assert model.task2_data is None  # Should remain None for Task 1

    # Test Task 2 (should collect data)
    task2_dataset = MockDataset(1)
    task2_dataset.train_loader = "mock_loader"  # Mock loader
    model.begin_task(task2_dataset)
    assert model.task2_data == "mock_loader"  # Should store Task 2 data


def test_scratch_t2_observe_returns_zero(tiny_components, tiny_args):
    """Test that observe method returns 0 (no training during task progression)."""
    backbone, loss, transform, dataset = tiny_components
    args = tiny_args
    args.n_epochs = 1
    args.debug_mode = True

    model = ScratchT2(backbone, loss, args, transform, dataset=dataset)

    # Test that observe returns 0 (no training)
    inputs = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 6, (4,))
    result = model.observe(inputs, labels, inputs)
    assert result == 0
