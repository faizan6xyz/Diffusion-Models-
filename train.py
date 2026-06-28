import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleUNet
from diffusion import add_noise, T
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.ToTensor()
dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
epochs = 5
for epoch in range(epochs):
    for images, _ in loader:
        images = images.to(device)
        t = torch.randint(
            0,
            T,
            (images.shape[0],),
            device=device
        )
        noisy_images, noise = add_noise(images, t)
        predicted_noise = model(noisy_images)
        loss = criterion(predicted_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
torch.save(model.state_dict(), "diffusion_model.pth")
print("Model saved.")
