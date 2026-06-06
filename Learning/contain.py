# Imports
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Define SimpleUNet class
class SimpleUNet(nn.Module):
    ...

# 2. Noise schedule
T = 1000
beta = torch.linspace(1e-4, 0.02, T)
alpha = 1.0 - beta
alpha_hat = torch.cumprod(alpha, dim=0)

# 3. add_noise function
def add_noise(x, t):
    ...

# 4. Load dataset
transform = transforms.ToTensor()
dataset = datasets.MNIST(...)
loader = DataLoader(...)

# 5. Create model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleUNet().to(device)

# 6. Training loop
for epoch in range(epochs):
    ...

'''
    diffusion_project/
│
├── data/
│   └── (dataset files downloaded automatically)
│
├── checkpoints/
│   └── model_epoch_5.pth
│
├── model.py
├── diffusion.py
├── dataset.py
├── train.py
├── sample.py
├── config.py
├── utils.py
├── requirements.txt
└── README.md


model.py
    ↓
Defines U-Net

diffusion.py
    ↓
Defines noise process

train.py
    ↓
Trains the network

sample.py
    ↓
Generates images


'''