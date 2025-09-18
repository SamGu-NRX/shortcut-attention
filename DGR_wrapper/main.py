import torch
import tqdm
import copy
import numpy as np

from models.classifier import Classifier
from models.vae import VAE
from models.utils import loss_functions as lf

# --- Device Setup ---
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print(f"CUDA is {' ' if cuda else 'not '}used")

# --- Dummy Data Loading (Mammoth will handle real data) ---
# This is a placeholder. In a real scenario, Mammoth would provide the data.
# For demonstration, let's create some random data.
image_size = 28
image_channels = 1
classes = 10

# Dummy config (mimicking what get_context_set might return)
config = {
    'size': image_size,
    'channels': image_channels,
    'classes': classes,
    'output_units': classes,
    'classes_per_context': 2,
}

# Dummy datasets (Mammoth will provide these)
# For simplicity, let's create a single dummy dataset for one context.
# In a real DGR setup, you'd have multiple contexts/tasks.
num_samples = 1000
dummy_train_data = torch.randn(num_samples, image_channels, image_size, image_size)
dummy_train_labels = torch.randint(0, classes, (num_samples,))

# Create a simple DataLoader for the dummy data
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


dummy_train_dataset = DummyDataset(dummy_train_data, dummy_train_labels)

# --- Model Specification ---
# Classifier
fc_lay = 3
fc_units = 400
fc_bn = False
fc_nl = "relu"

model = Classifier(
    image_size=config['size'], image_channels=config['channels'], classes=config['output_units'],
    depth=0,
    fc_layers=fc_lay, fc_units=fc_units, fc_bn=fc_bn, fc_nl=fc_nl, excit_buffer=True,
).to(device)

model.scenario = "class"
model.classes_per_context = config['classes_per_context']
model.replay_mode = 'generative'
model.replay_targets = 'hard'

# Generator (VAE)
g_fc_lay = 3
g_fc_units = 400
g_fc_bn = False
g_fc_nl = "relu"
g_z_dim = 100

generator = VAE(
    image_size=config['size'], image_channels=config['channels'],
    depth=0,
    fc_layers=g_fc_lay, fc_units=g_fc_units, fc_bn=g_fc_bn, fc_nl=g_fc_nl, excit_buffer=True,
    prior='standard', z_dim=g_z_dim,
    recon_loss='BCE', network_output='sigmoid'
).to(device)

# --- Optimizer Setup ---
lr = 0.001

model.optim_list = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': lr}]
model.optimizer = torch.optim.Adam(model.optim_list, betas=(0.9, 0.999))

generator.optim_list = [{'params': filter(lambda p: p.requires_grad, generator.parameters()), 'lr': lr}]
generator.optimizer = torch.optim.Adam(generator.optim_list, betas=(0.9, 0.999))

# --- Training Loop (Simplified) ---

iters = 1000
batch_size = 128
contexts = 1 # For this simplified example, we'll run one context

previous_model = previous_generator = None

for context in range(1, contexts + 1):
    print(f"\n--- Training Context {context}/{contexts} ---")

    data_loader = torch.utils.data.DataLoader(dummy_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    progress = tqdm.tqdm(range(1, iters + 1))
    for batch_index in progress:
        x, y = next(iter(data_loader))
        x, y = x.to(device), y.to(device)

        # --- Replay Data (if applicable) ---
        x_ = y_ = scores_ = None
        if previous_model is not None:
            # In a real DGR, you'd sample from previous_generator and classify with previous_model
            # For this dummy setup, we'll just create random replay data for demonstration
            x_ = torch.randn(batch_size, image_channels, image_size, image_size).to(device)
            y_ = torch.randint(0, classes, (batch_size,)).to(device)

        # --- Train Classifier ---
        loss_dict_model = model.train_a_batch(
            x, y, x_=x_, y_=y_, scores_=scores_, rnt=1./context,
            active_classes=list(range(config['output_units'])), context=context
        )

        # --- Train Generator ---
        loss_dict_generator = generator.train_a_batch(x, x_=x_, rnt=1./context)

        progress.set_description(
            f"<CLASSIFIER> | Loss: {loss_dict_model['loss_total']:.3f} | Acc: {loss_dict_model['accuracy']:.3f} "
            f"<VAE> | Loss: {loss_dict_generator['loss_total']:.3f}"
        )

    # --- Update previous models for replay ---
    previous_generator = copy.deepcopy(generator).eval()
    previous_model = copy.deepcopy(model).eval()

print("\nDGR Isolation complete. main.py created.")
