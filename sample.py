import torch
import matplotlib.pyplot as plt

from model import SimpleUNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleUNet().to(device)
model.load_state_dict(torch.load("diffusion_model.pth"))
model.eval()

with torch.no_grad():
    noise = torch.randn(1, 1, 28, 28).to(device)

    generated = model(noise)

    image = generated.squeeze().cpu()

    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()
