import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

T          = 300    # Total diffusion timesteps — how many steps to corrupt/recover an image
BETA_START = 1e-4   # Starting noise variance (very small — barely any noise at step 0)
BETA_END   = 0.02   # Ending noise variance (larger — almost pure noise at step T)
EPOCHS     = 5      # Number of full passes over the training dataset
BATCH      = 128    # Number of images processed per gradient update
LR         = 2e-4   # Adam learning rate
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# β_t: how much new noise is added at each timestep t (linearly increasing)
betas     = torch.linspace(BETA_START, BETA_END, T).to(DEVICE)
# α_t = 1 - β_t: how much of the signal is preserved at each step
alphas    = 1.0 - betas
# ᾱ_t = cumulative product of all α up to t
# This lets us jump directly to any noise level without stepping through each t:
#   noisy_image = sqrt(ᾱ_t) * x0  +  sqrt(1 - ᾱ_t) * noise
alpha_bar = torch.cumprod(alphas, dim=0)

def q_sample(x0, t):
    """
    Forward (noising) process: corrupt a clean image x0 to timestep t in one shot.
    Thanks to the reparameterisation trick we skip the intermediate steps.
    Returns the noisy image AND the noise that was added (needed for the loss).
    """
    ab    = alpha_bar[t][:, None, None, None]   # reshape to (B,1,1,1) for broadcasting
    noise = torch.randn_like(x0)                # sample ε ~ N(0, I)
    # Closed-form: x_t = √ᾱ·x0 + √(1-ᾱ)·ε
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

# ── Sinusoidal time embedding ─────────────────────────────────────────────────
# The U-Net needs to know *which* timestep it's denoising, so we encode t as a
# vector of sines and cosines at different frequencies — the same idea used for
# positional encoding in transformers.
class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half  = dim // 2
        # Exponentially spaced frequencies: low freqs capture broad trends,
        # high freqs capture fine-grained timestep differences
        freqs = torch.exp(-torch.arange(half) * (8.0 / half))
        self.register_buffer("freqs", freqs)    # saved with model, not a parameter

    def forward(self, t):           # t: integer tensor of shape (B,)
        args = t[:, None].float() * self.freqs[None]    # (B, dim//2)
        # Concatenate sin and cos so the embedding is unique for each t
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)

# ── Tiny U-Net ────────────────────────────────────────────────────────────────
# U-Net = encoder (downsample) + bottleneck + decoder (upsample) with skip connections.
# Skip connections carry fine-grained spatial info from the encoder to the decoder,
# helping the model predict accurate pixel-level noise.
class UNet(nn.Module):
    def __init__(self, ch=64, emb_dim=128):
        super().__init__()
        self.emb  = SinusoidalEmb(emb_dim)         # time → vector
        self.proj = nn.Linear(emb_dim, ch)          # project time emb to channel width

        # A reusable double-conv block: Conv → Norm → Activation × 2
        # GroupNorm stabilises training; SiLU (swish) is smooth and works well here
        def blk(ci, co): return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU())

        # Encoder: progressively increase channels, decrease spatial size
        self.d1  = blk(1,    ch)       # input:  (B, 1,   28, 28) → (B, ch,   28, 28)
        self.d2  = blk(ch,   ch * 2)  # after pool: (B, ch, 14, 14) → (B, 2ch, 14, 14)

        # Bottleneck: deepest representation where time embedding is injected
        self.mid = blk(ch*2, ch * 2)

        # Decoder: upsample back to original resolution, using skip connections
        # ch*4 input = upsampled (2ch) + skip from encoder (2ch) concatenated
        self.u1  = blk(ch*4, ch)       # (B, 4ch, 14, 14) → (B, ch, 14, 14)
        self.u2  = blk(ch*2, ch)       # (B, 2ch, 28, 28) → (B, ch, 28, 28)

        self.out  = nn.Conv2d(ch, 1, 1)        # 1×1 conv → back to 1-channel output
        self.down = nn.MaxPool2d(2)            # halves spatial dimensions
        self.up   = nn.Upsample(scale_factor=2, mode="nearest")  # doubles spatial dims

    def forward(self, x, t):
        # Encode the timestep and reshape to (B, ch, 1, 1) so it broadcasts spatially
        e  = self.proj(self.emb(t))[:, :, None, None]

        # ── Encoder (save activations as skip connections) ──
        s1 = self.d1(x)             # full resolution features
        s2 = self.d2(self.down(s1)) # half resolution features

        # ── Bottleneck (add time info here) ──
        h  = self.mid(s2) + e       # adding e injects timestep knowledge

        # ── Decoder (upsample + concatenate skip connections) ──
        h  = self.u1(torch.cat([self.up(h), s2], dim=1))  # skip from d2
        h  = self.u2(torch.cat([self.up(h), s1], dim=1))  # skip from d1

        return self.out(h)  # predicted noise, same shape as input x

# ── Data loading ──────────────────────────────────────────────────────────────
# MNIST: 28×28 grayscale handwritten digits.
# We rescale pixel values from [0,1] to [-1,1] so the data matches the range
# of Gaussian noise used in the diffusion process.
loader = DataLoader(
    datasets.MNIST(".", download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Lambda(lambda x: x * 2 - 1)  # [0,1] → [-1,1]
                   ])),
    batch_size=BATCH, shuffle=True)

# ── Training ──────────────────────────────────────────────────────────────────
model = UNet().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    total = 0
    for x, _ in loader:             # labels (_) are unused — this is unconditional
        x  = x.to(DEVICE)

        # Sample a random timestep t for each image in the batch
        # Training on all t uniformly makes the model learn every noise level
        t  = torch.randint(0, T, (x.size(0),), device=DEVICE)

        # Corrupt the images and get the noise that was added
        xt, noise = q_sample(x, t)

        # The model predicts the noise; loss = how wrong that prediction is
        # This is the core DDPM objective: minimise E[||ε - ε_θ(x_t, t)||²]
        loss = F.mse_loss(model(xt, t), noise)

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()

    print(f"Epoch {epoch}/{EPOCHS}  loss={total/len(loader):.4f}")

# ── Sampling (DDPM reverse process) ──────────────────────────────────────────
# To generate images we reverse the process: start from pure noise x_T ~ N(0,I)
# and iteratively denoise, stepping from t=T down to t=0.
@torch.no_grad()
def sample(n=16):
    # Start from pure Gaussian noise — this is x_T
    x = torch.randn(n, 1, 28, 28, device=DEVICE)

    for t in reversed(range(T)):    # T-1, T-2, ..., 1, 0
        # Create a batch of the same timestep value
        tv   = torch.full((n,), t, device=DEVICE, dtype=torch.long)

        # Ask the model: "what noise was added to reach this x at step t?"
        pred = model(x, tv)

        ab      = alpha_bar[t]
        ab_prev = alpha_bar[t - 1] if t > 0 else torch.tensor(1.0)

        # Estimate the clean image x0 from the current noisy x and predicted noise
        x0   = (x - (1 - ab).sqrt() * pred) / ab.sqrt()
        x0   = x0.clamp(-1, 1)     # keep estimate in valid range

        # Direction pointing towards x_t (the predicted noise component)
        dir_ = (1 - ab_prev).sqrt() * pred

        # Compute x_{t-1}: the slightly-less-noisy image
        x    = ab_prev.sqrt() * x0 + dir_

        # Add a small amount of stochastic noise at every step except the last
        # This is what makes DDPM stochastic (vs DDIM which is deterministic)
        if t > 0:
            x += betas[t].sqrt() * torch.randn_like(x)

    # Rescale from [-1, 1] back to [0, 1] for display
    return (x.clamp(-1, 1) + 1) / 2

imgs = sample(16)
print("Sampled tensor shape:", imgs.shape)  # (16, 1, 28, 28)
# Uncomment to save a grid of generated images:
# import torchvision; torchvision.utils.save_image(imgs, "samples.png", nrow=4)