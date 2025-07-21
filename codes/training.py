# model.py
import torch
import torch.nn as nn
import os

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = SimpleNN().to(device)
    x = torch.randn(32, 10).to(device)
    out = model(x)
    print("Output shape:", out.shape)

    # Guardar el modelo en la carpeta 'models'
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/last_model.pth")
    print("Modelo guardado en 'models/last_model.pth'")