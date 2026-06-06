import torch

T = 1000

beta = torch.linspace(1e-4, 0.02, T)
alpha = 1.0 - beta
alpha_hat = torch.cumprod(alpha, dim=0)

def add_noise(images, t):
    noise = torch.randn_like(images)

    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]

    noisy_images = (
        sqrt_alpha_hat * images +
        sqrt_one_minus_alpha_hat * noise
    )

    return noisy_images, noise
