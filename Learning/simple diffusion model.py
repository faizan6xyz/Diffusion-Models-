import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Config ──────────────────────────────────────────────────────────────────
T          = 300          # diffusion timesteps
BETA_START = 1e-4
BETA_END   = 0.02
EPOCHS     = 5
BATCH      = 128
LR         = 2e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

betas      = torch.linspace(BETA_START, BETA_END, T).to(DEVICE)
alphas     = 1.0 - betas
alpha_bar  = torch.cumprod(alphas, dim=0)           # ᾱ_t

def q_sample(x0, t):
    ab = alpha_bar[t][:, None, None, None]           # (B,1,1,1)
    noise = torch.randn_like(x0)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-torch.arange(half) * (8.0 / half))
        self.register_buffer("freqs", freqs)

    def forward(self, t):                            # t: (B,)
        args = t[:, None].float() * self.freqs[None] # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)

# ── Tiny U-Net ───────────────────────────────────────────────────────────────
class UNet(nn.Module):
    def __init__(self, ch=64, emb_dim=128):
        super().__init__()
        self.emb  = SinusoidalEmb(emb_dim)
        self.proj = nn.Linear(emb_dim, ch)

        def blk(ci, co): return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU())

        self.d1  = blk(1,    ch)        # 28×28
        self.d2  = blk(ch,   ch * 2)   # 14×14
        self.mid = blk(ch*2, ch * 2)
        self.u1  = blk(ch*4, ch)        # 14×14  (skip from d2)
        self.u2  = blk(ch*2, ch)        # 28×28  (skip from d1)
        self.out = nn.Conv2d(ch, 1, 1)
        self.down = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t):
        e  = self.proj(self.emb(t))[:, :, None, None]   # (B,ch,1,1)
        s1 = self.d1(x)                                  # (B,ch,28,28)
        s2 = self.d2(self.down(s1))                      # (B,2ch,14,14)
        h  = self.mid(s2) + e                            # inject time
        h  = self.u1(torch.cat([self.up(h), s2], 1))    # (B,ch,14,14)
        h  = self.u2(torch.cat([self.up(h), s1], 1))    # (B,ch,28,28)
        return self.out(h)

# ── Training ─────────────────────────────────────────────────────────────────
loader = DataLoader(
    datasets.MNIST(".", download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                                                 transforms.Lambda(lambda x: x * 2 - 1)])),
    batch_size=BATCH, shuffle=True)

model = UNet().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    total = 0
    for x, _ in loader:
        x  = x.to(DEVICE)
        t  = torch.randint(0, T, (x.size(0),), device=DEVICE)
        xt, noise = q_sample(x, t)
        loss = F.mse_loss(model(xt, t), noise)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    print(f"Epoch {epoch}/{EPOCHS}  loss={total/len(loader):.4f}")

# ── Sampling (DDPM reverse) ──────────────────────────────────────────────────
@torch.no_grad()
def sample(n=16):
    x = torch.randn(n, 1, 28, 28, device=DEVICE)
    for t in reversed(range(T)):
        tv   = torch.full((n,), t, device=DEVICE, dtype=torch.long)
        pred = model(x, tv)
        ab   = alpha_bar[t];  ab_prev = alpha_bar[t-1] if t > 0 else torch.tensor(1.0)
        x0   = (x - (1-ab).sqrt() * pred) / ab.sqrt()
        x0   = x0.clamp(-1, 1)
        dir_ = (1 - ab_prev).sqrt() * pred
        x    = ab_prev.sqrt() * x0 + dir_
        if t > 0:
            x += betas[t].sqrt() * torch.randn_like(x)
    return (x.clamp(-1,1) + 1) / 2          # → [0,1]

imgs = sample(16)
print("Sampled tensor shape:", imgs.shape)  # (16, 1, 28, 28)
# Save: torchvision.utils.save_image(imgs, "samples.png", nrow=4)