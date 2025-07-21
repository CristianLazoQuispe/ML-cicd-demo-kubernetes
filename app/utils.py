# app/utils.py
import torch
from app.model import SimpleNN

def load_model(model_path: str = "models/last_model.pth", device="cuda" if torch.cuda.is_available() else "cpu"):
    model = SimpleNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
