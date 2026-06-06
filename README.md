<div align="center">

# 🌀 Diffusion Model from Scratch

**A minimal, fully-commented DDPM implementation in PyTorch — trained on MNIST in under 100 lines.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 📖 Overview

This repository implements a **Denoising Diffusion Probabilistic Model (DDPM)** from scratch, following the paper [*Ho et al. (2020)*](https://arxiv.org/abs/2006.11239). The goal is clarity over complexity — every line is commented, every design choice is explained.

The model learns to **generate handwritten digits** by reversing a gradual noising process, starting from pure Gaussian noise and iteratively denoising until a clean image emerges.

> ✨ Entire training + sampling pipeline in a single ~100-line file.

---

## 🧠 How It Works

Diffusion models work in two phases:

```
Forward process (training):   x₀ ──noise──▶ x₁ ──noise──▶ ... ──▶ xₜ  (pure noise)
Reverse process (sampling):   xₜ ──denoise──▶ ... ──▶ x₁ ──denoise──▶ x₀  (clean image)
```

**Training:** At each step, the model receives a noisy image `xₜ` and timestep `t`, and learns to predict the noise `ε` that was added. Loss = MSE between predicted and actual noise.

**Sampling:** Starting from random noise, the model repeatedly predicts and removes noise over `T` steps until a clean image is recovered.

---

## 🏗️ Architecture

| Component | Description |
|---|---|
| **Noise Schedule** | Linear `β` schedule from `1e-4` → `0.02` over `T=300` steps |
| **Forward Process** | Closed-form sampling: `xₜ = √ᾱ·x₀ + √(1-ᾱ)·ε` |
| **Backbone** | Tiny U-Net with skip connections and GroupNorm |
| **Time Conditioning** | Sinusoidal embeddings injected at the bottleneck |
| **Loss** | MSE between predicted and actual noise `‖ε - ε_θ(xₜ, t)‖²` |
| **Sampler** | Stochastic DDPM reverse process |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/faizan6xyz/Diffusion-Models-
```

### 2. Install dependencies

```bash
pip install torch torchvision
```

### 3. Train and sample

```bash
python diffusion_model.py
```

MNIST will be downloaded automatically on first run. Training for 5 epochs takes ~10 minutes on CPU, ~2 minutes on GPU.

### 4. Save generated samples

Uncomment the last line in the script:

```python
import torchvision
torchvision.utils.save_image(imgs, "samples.png", nrow=4)
```

---

## ⚙️ Configuration

All hyperparameters are at the top of `diffusion_model.py`:

| Parameter | Default | Description |
|---|---|---|
| `T` | `300` | Number of diffusion timesteps |
| `BETA_START` | `1e-4` | Starting noise variance |
| `BETA_END` | `0.02` | Ending noise variance |
| `EPOCHS` | `5` | Training epochs |
| `BATCH` | `128` | Batch size |
| `LR` | `2e-4` | Adam learning rate |

---

## 📁 Project Structure

```
diffusion-model/
│
├── diffusion_model.py     # Full model: schedule, U-Net, training, sampling
├── README.md              # This file
└── samples.png            # Generated images (after running the script)
```

---

## 📚 Key Concepts

<details>
<summary><b>What is ᾱ (alpha bar)?</b></summary>

`ᾱₜ = ∏ αₛ` for `s = 1..t` is the cumulative product of `(1 - βₜ)`. It lets us jump directly from a clean image `x₀` to any noise level `t` in a single operation — no need to step through each intermediate timestep during training.

</details>

<details>
<summary><b>Why sinusoidal time embeddings?</b></summary>

The U-Net needs to know *which* timestep it's denoising (step 1 vs step 299 require very different predictions). Sinusoidal embeddings encode `t` as a vector of sines and cosines at exponentially-spaced frequencies — the same idea used for positional encoding in Transformers — giving each timestep a unique, smooth fingerprint.

</details>

<details>
<summary><b>What are skip connections in the U-Net?</b></summary>

During downsampling, spatial detail is lost. Skip connections carry full-resolution feature maps from the encoder directly to the corresponding decoder layer, allowing the model to produce sharp, spatially-accurate noise predictions.

</details>

<details>
<summary><b>Why add noise during sampling?</b></summary>

The DDPM reverse process is stochastic — a small amount of fresh noise is added at each denoising step (except the very last). This is what allows the model to generate diverse samples from the same model; removing it gives DDIM, a deterministic sampler.

</details>

---

## 📄 Reference

```bibtex
@article{ho2020ddpm,
  title   = {Denoising Diffusion Probabilistic Models},
  author  = {Jonathan Ho and Ajay Jain and Pieter Abbeel},
  journal = {NeurIPS},
  year    = {2020},
  url     = {https://arxiv.org/abs/2006.11239}
}
```

---

## 📝 License

MIT — free to use, modify, and distribute. See [LICENSE](LICENSE) for details.
