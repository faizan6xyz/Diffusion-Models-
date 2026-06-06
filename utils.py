# utils.py

import os
import torch


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def load_checkpoint(model, filename):
    model.load_state_dict(torch.load(filename))
    print(f"Model loaded from {filename}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
