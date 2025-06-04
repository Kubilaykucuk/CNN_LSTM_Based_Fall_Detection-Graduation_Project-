import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# ===========================
# CONFIG
# ===========================
CSV_PATH = "dataset\combined_all_data.csv"  # Replace with your CSV file
SEQUENCE_LENGTH = 5
BATCH_SIZE = 32
NUM_EPOCHS = 100
MODEL_PATH = "model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# EARLY STOPPING CLASS
# ===========================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ===========================
# MODEL DEFINITION
# ===========================
class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_features, num_classes=3):
        super(CNN_LSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2,
                            batch_first=True)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.permute(0, 2, 1)  # (batch, reduced_seq_len, channels)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    
# ===========================
# DATA PREPROCESSING
# ===========================
df = pd.read_csv(CSV_PATH)
df.sort_values(by=["ObjectID", "Timestamp"], inplace=True)

# Feature selection
features = ["X(m)", "Y(m)", "Z(m)", "Vx(m/s)", "Vy(m/s)", "Vz(m/s)",
            "Vr(m/s)", "range_sc(m)", "azimuth_sc(rad)",
            "Z_diff", "X_diff", "Y_diff"]
label_col = "Posture"

# Drop rows with any NaN or inf values in features or labels
df = df.replace([np.inf, -np.inf], np.nan)
df.dropna(subset=features + [label_col], inplace=True)

# Normalize
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Sequence building
X_data, y_data = [], []
for obj_id, group in df.groupby("ObjectID"):
    group = group.reset_index(drop=True)
    for i in range(len(group) - SEQUENCE_LENGTH + 1):
        seq = group.loc[i:i+SEQUENCE_LENGTH-1, features].values
        label = group.loc[i+SEQUENCE_LENGTH-1, label_col]
        X_data.append(seq)
        y_data.append(label)

X_data = np.array(X_data)
y_data = np.array(y_data).astype(np.int64)

# Downsample majority classes
from collections import Counter
from sklearn.utils import resample

class_counts = Counter(y_data)
min_class = min(class_counts, key=class_counts.get)
min_count = class_counts[min_class]

X_balanced = []
y_balanced = []

for cls in np.unique(y_data):
    idxs = np.where(y_data == cls)[0]
    sampled_idxs = resample(idxs, replace=False, n_samples=min_count, random_state=42)
    X_balanced.append(X_data[sampled_idxs])
    y_balanced.append(y_data[sampled_idxs])

X_data_bal = np.concatenate(X_balanced)
y_data_bal = np.concatenate(y_balanced)

# Shuffle the balanced dataset
from sklearn.utils import shuffle
X_data_bal, y_data_bal = shuffle(X_data_bal, y_data_bal, random_state=42)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X_data_bal, y_data_bal, test_size=0.2, stratify=y_data_bal, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Shuffle the balanced dataset
from sklearn.utils import shuffle
X_data_bal, y_data_bal = shuffle(X_data_bal, y_data_bal, random_state=42)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X_data_bal, y_data_bal, test_size=0.2, stratify=y_data_bal, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ===========================
# CLASS WEIGHTS AND LOSS
# ===========================
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# ===========================
# LOAD MODEL IF EXISTS
# ===========================
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = torch.load(MODEL_PATH)
else:
    model = CNN_LSTM_Model(num_features=len(features)).to(DEVICE)

# ===========================
# TRAINING LOOP
# ===========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
early_stopper = EarlyStopping(patience=15, min_delta=0.0005)

train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Early stopping
    early_stopper(avg_val_loss)
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break

    # Save model
    torch.save(model, MODEL_PATH)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='x')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("loss_graph.png", dpi=300)
plt.close()
print("✅ Saved training loss graph as loss_graph.png")