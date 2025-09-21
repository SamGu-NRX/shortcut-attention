import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from torch.utils.data import DataLoader, TensorDataset

IMAGE_SIZE = 32
NUM_CLASSES = 6
DEVICE = torch.device("cpu")


class TinyBackbone(nn.Module):
    """Compact convolutional network for adapter integration tests."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 16, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.pool(x)
        x = self.flatten(x)
        return self.classifier(x)


class TinyDataset:
    """Synthetic dataset that mimics the Mammoth dataset API."""

    def __init__(self, num_samples: int = 32, num_classes: int = NUM_CLASSES) -> None:
        self.N_CLASSES = num_classes
        self.N_TASKS = 2
        self.N_CLASSES_PER_TASK = num_classes // self.N_TASKS
        self.SETTING = 'class-il'
        self.SIZE = [3, IMAGE_SIZE, IMAGE_SIZE]

        generator = torch.Generator().manual_seed(0)
        self._data = torch.rand(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
        self._labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
        dataset = TensorDataset(self._data, self._labels)
        self._loader = DataLoader(dataset, batch_size=8, shuffle=True)

    def get_data_loaders(self):
        return self._loader, self._loader

    def get_offsets(self, task: int):
        start = task * self.N_CLASSES_PER_TASK
        end = min(self.N_CLASSES, (task + 1) * self.N_CLASSES_PER_TASK)
        return start, end

    def get_normalization_transform(self):
        return nn.Identity()

    def sample_batch(self, batch_size: int = 4):
        idx = torch.randint(0, self._data.size(0), (batch_size,))
        return self._data[idx], self._labels[idx]


@pytest.fixture
def tiny_dataset():
    torch.manual_seed(0)
    return TinyDataset()


@pytest.fixture
def tiny_components(tiny_dataset):
    backbone = TinyBackbone(num_classes=tiny_dataset.N_CLASSES).to(DEVICE)
    loss = nn.CrossEntropyLoss()
    transform = nn.Identity()
    return backbone, loss, transform, tiny_dataset


@pytest.fixture
def tiny_args():
    args = Namespace()

    # Core ContinualModel settings
    args.lr = 0.01
    args.optimizer = 'sgd'
    args.optim_wd = 0.0001
    args.optim_mom = 0.9
    args.optim_nesterov = False
    args.seed = 1
    args.nowand = True
    args.label_perc = 1.0
    args.num_workers = 0
    args.batch_size = 8
    args.device = 'cpu'
    args.buffer_size = 0

    # GPM configuration
    args.gpm_threshold_base = 0.97
    args.gpm_threshold_increment = 0.001
    args.gpm_activation_samples = 64

    # DGR configuration
    args.dgr_z_dim = 8
    args.dgr_vae_lr = 0.001
    args.dgr_replay_ratio = 0.5
    args.dgr_temperature = 2.0

    return args


def sample_batch(dataset: TinyDataset, batch_size: int = 4):
    inputs, labels = dataset.sample_batch(batch_size)
    return inputs.to(DEVICE), labels.to(DEVICE)


def run_gpm_steps(model, dataset: TinyDataset, steps: int = 2):
    for _ in range(steps):
        inputs, labels = sample_batch(dataset)
        model.observe(inputs, labels, inputs)


def run_dgr_steps(model, dataset: TinyDataset, steps: int = 2):
    for _ in range(steps):
        inputs, labels = sample_batch(dataset)
        model.observe(inputs, labels, inputs)
