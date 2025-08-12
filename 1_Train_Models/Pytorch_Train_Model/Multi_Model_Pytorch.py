import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -----------------------------
# Device Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Data Utilities
# -----------------------------
def load_data(npy_path: str, test_size: float, batch_size: int, random_state: int):
    data = np.load(npy_path, allow_pickle=True)
    X = data[:, :, :12].astype(np.float32)
    y = data[:, 0, -1].astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    # Convert to torch tensors
    train_ds = TensorDataset(
        torch.from_numpy(X_train).to(device),
        torch.from_numpy(y_train).to(device)
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test).to(device),
        torch.from_numpy(y_test).to(device)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# -----------------------------
# Model Definitions
# -----------------------------
class MultiDNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(True),
            nn.Linear(32, 64), nn.ReLU(True),
            nn.Linear(64, 128), nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4, 256), nn.ReLU(True),
            nn.Dropout(0.2), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, sensors, seq_len, features) => flatten per sensor
        outs = []
        for i in range(x.size(1)):
            sensor = x[:, i].reshape(x.size(0), -1)
            outs.append(self.shared(sensor))
        cat = torch.cat(outs, dim=1)
        return self.classifier(cat)

class MultiCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3), nn.ReLU(True), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3), nn.ReLU(True), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3), nn.ReLU(True), nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (((300 - 2* (3-1)) // 2 // 2 // 2)), 256),
            nn.ReLU(True), nn.Dropout(0.2), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, sensors, seq_len, features) -> permute to channels first then process each sensor
        outs = []
        for i in range(x.size(1)):
            sensor = x[:, i].permute(0, 2, 1)  # (batch, features, seq_len)
            outs.append(self.conv(sensor))
        # flatten and concat
        feats = [o.flatten(1) for o in outs]
        cat = torch.cat(feats, dim=1)
        return self.fc(cat)

# You can add more models (e.g., LSTM, DenseNet) following the same pattern

# -----------------------------
# Training & Evaluation
# -----------------------------

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_model(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch)
            total_loss += criterion(preds, y_batch).item()
            correct += (preds.argmax(1) == y_batch).sum().item()
    acc = correct / len(loader.dataset)
    return total_loss / len(loader), acc

# -----------------------------
# Main
# -----------------------------
def main(args):
    # Data
    train_loader, test_loader = load_data(
        args.data_path, args.test_size, args.batch_size, args.random_state
    )

    # Model Selection
    if args.model == 'dnn':
        model = MultiDNN(input_dim=300*12, num_classes=args.num_classes)
    elif args.model == 'cnn':
        model = MultiCNN(in_channels=12, num_classes=args.num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model.to(device)

    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_model(model, test_loader, criterion)
        print(f"Epoch {epoch}: Train Loss={loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        scheduler.step()

    # Save
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HAR Multimodal Models")
    parser.add_argument('--model', type=str, choices=['dnn','cnn'], default='dnn')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='model.pth')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--test_size', type=float, default=0.5)
    parser.add_argument('--random_state', type=int, default=1004)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=7)
    args = parser.parse_args()
    main(args)
