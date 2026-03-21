# src/client.py

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import OrderedDict
from src.models import get_model

# --- config ---
NUM_CLASSES  = 38
BATCH_SIZE   = 32
LOCAL_EPOCHS = 3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_data(client_id: int):
    """Load this client's partition from data/clients/client_X/"""
    train_dir = f"data/clients/client_{client_id}/train"
    val_dir   = f"data/clients/client_{client_id}/val"

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds   = datasets.ImageFolder(val_dir,   transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    return train_loader, val_loader


def train(model, loader, epochs):
    """Local training — same concept as AGRIFOLD's per-client training loop"""
    model.train()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def evaluate(model, loader):
    """Local evaluation — returns loss + accuracy"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), correct / total


class CropDiseaseClient(fl.client.NumPyClient):
    """
    Flower NumPyClient — this is your equivalent of AGRIFOLD's run_fl_inner.sh
    when task_id > 0 (i.e. client role).

    Three methods Flower requires:
      get_parameters → send weights TO server
      set_parameters → receive weights FROM server
      fit            → local training
      evaluate       → local evaluation
    """

    def __init__(self, client_id: int):
        self.client_id   = client_id
        self.model       = get_model(NUM_CLASSES).to(DEVICE)
        self.train_loader, self.val_loader = load_data(client_id)

    def get_parameters(self, config):
        # Extract model weights as list of numpy arrays — sent to server
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # Load weights received from server into local model
        params_dict  = zip(self.model.state_dict().keys(), parameters)
        state_dict   = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # 1. Receive global model weights from server
        self.set_parameters(parameters)
        # 2. Train locally for LOCAL_EPOCHS
        train(self.model, self.train_loader, epochs=LOCAL_EPOCHS)
        # 3. Return updated weights + number of training samples
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Evaluate global model on this client's local val set
        self.set_parameters(parameters)
        loss, accuracy = evaluate(self.model, self.val_loader)
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}


def client_fn(cid: str) -> CropDiseaseClient:
    """Factory function — Flower calls this to create each client by ID"""
    return CropDiseaseClient(client_id=int(cid))