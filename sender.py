"""
sender.py

This script runs a radar data stream from MATLAB, applies a CNN-LSTM model to classify object movement,
clusters the results, tracks object IDs, and sends real-time predictions to a TCP server.
"""

import subprocess
import json
import os
import csv
import time
import socket
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict, deque
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

# === Constants ===
SEQUENCE_LENGTH = 5
features = [
    "X(m)", "Y(m)", "Z(m)", "Vx(m/s)", "Vy(m/s)", "Vz(m/s)",
    "Vr(m/s)", "range_sc(m)", "azimuth_sc(rad)",
    "Z_diff", "X_diff", "Y_diff"
]
label_map = {0: "Nothing", 1: "Falling", 2: "Walking"}

TRACK_MEMORY = 10  # kaç frame boyunca ID’yi tutacağımız
track_history = {}  # track_id -> (centroid, last_seen_frame)
frame_counter = 0

# === Model Setup ===
MODEL_PATH = "model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_features, num_classes=3):
        super(CNN_LSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    
def load_model(model_path):
    model = torch.load(model_path, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model(MODEL_PATH)

# === TCP Client Setup ===
SERVER_IP = os.getenv("SERVER_IP", "127.0.0.1")  # fallback IP
SERVER_PORT = 8000
tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_client.connect((SERVER_IP, SERVER_PORT))

object_position_history = defaultdict(lambda: deque(maxlen=10))
STATIC_MOVEMENT_THRESHOLD = 0.3  # meters

# === Buffers ===
scaler = MinMaxScaler()
pred_buffers = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
pred_results = {}
track_centroids = {}
next_track_id = 0

# === Helpers ===
def compute_differences(df):
    df['Prev_Z'] = df.groupby('ObjectID')['Z(m)'].shift(1)
    df['Z_diff'] = df['Z(m)'] - df['Prev_Z']
    df['X_diff'] = df.groupby('ObjectID')['X(m)'].diff().abs()
    df['Y_diff'] = df.groupby('ObjectID')['Y(m)'].diff().abs()
    return df

def match_clusters_to_tracks(cluster_centroids, track_centroids, max_dist=1.5):
    matches = {}
    unmatched_clusters = list(range(len(cluster_centroids)))

    if not track_centroids:
        # İlk karede cluster ID = track ID olarak eşleştir (0,1,2,3...)
        matches = {i: i for i in range(len(cluster_centroids))}
        return matches, []

    cluster_centroids = np.array(cluster_centroids)
    track_ids = list(track_centroids.keys())
    track_positions = np.array(list(track_centroids.values()))

    distances = cdist(cluster_centroids, track_positions)

    used_tracks = set()
    used_clusters = set()

    for cluster_idx, row in enumerate(distances):
        min_idx = np.argmin(row)
        if row[min_idx] < max_dist:
            track_id = track_ids[min_idx]
            if track_id not in used_tracks:
                matches[cluster_idx] = track_id
                used_clusters.add(cluster_idx)
                used_tracks.add(track_id)

    unmatched_clusters = [i for i in range(len(cluster_centroids)) if i not in used_clusters]
    return matches, unmatched_clusters

# === Launch MATLAB Process ===
matlab_command = [
    "matlab", "-batch",
    r"run('path_to_radar_code.m')"
]
print("Launching MATLAB radar stream...")
proc = subprocess.Popen(matlab_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

try:
    for line in proc.stdout:
        line = line.strip()
        if not (line.startswith("{") or line.startswith("[")):
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                if all(not isinstance(v, (list, dict)) for v in data.values()):
                    data = [data]
            df = pd.DataFrame(data)
            df.rename(columns={
                "X_m_": "X(m)", "Y_m_": "Y(m)", "Z_m_": "Z(m)",
                "Vx_m_s_": "Vx(m/s)", "Vy_m_s_": "Vy(m/s)", "Vz_m_s_": "Vz(m/s)",
                "Vr_m_s_": "Vr(m/s)", "range_sc_m_": "range_sc(m)", "azimuth_sc_rad_": "azimuth_sc(rad)"
            }, inplace=True)
        except json.JSONDecodeError:
            continue

        df = compute_differences(df)
        df = df.replace([np.inf, -np.inf], np.nan)

        df["Z_diff"] = df["Z_diff"].fillna(0)
        df["X_diff"] = df["X_diff"].fillna(0)
        df["Y_diff"] = df["Y_diff"].fillna(0)
        df.dropna(subset=[
            "X(m)", "Y(m)", "Z(m)", "Vx(m/s)", "Vy(m/s)", "Vz(m/s)",
            "Vr(m/s)", "range_sc(m)", "azimuth_sc(rad)"
        ], inplace=True)
        for frame_id in sorted(df["Frame"].unique()):
            frame_df = df[df["Frame"] == frame_id].copy()
            frame_df["X_unscaled"] = df.loc[frame_df.index]["X(m)"]
            frame_df["Y_unscaled"] = df.loc[frame_df.index]["Y(m)"]
            frame_df[features] = scaler.fit_transform(frame_df[features])

            # Model prediction for each ObjectID
            predictions = []
            for obj_id, obj_group in frame_df.groupby("ObjectID"):
                pred_buffers[obj_id].append(obj_group[features].values[0])
                
                # Save current position for static detection
                object_position_history[obj_id].append((obj_group["X_unscaled"].values[0], obj_group["Y_unscaled"].values[0]))

                if len(pred_buffers[obj_id]) == SEQUENCE_LENGTH:
                    seq = np.stack(pred_buffers[obj_id])
                    X_input = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32).to(DEVICE)

                    with torch.no_grad():
                        output = model(X_input)
                        pred = torch.argmax(output, dim=1).item()
                        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    
                    # === Check if object is static ===
                    positions = np.array(object_position_history[obj_id])
                    if len(positions) >= 2:
                        max_movement = np.max(np.linalg.norm(positions - positions[0], axis=1))
                        if max_movement < STATIC_MOVEMENT_THRESHOLD:
                            pred = 0  # Force label to "Nothing"

                    # Fetch unscaled X/Y
                    x_val = float(obj_group["X_unscaled"].values[0])
                    y_val = float(obj_group["Y_unscaled"].values[0])

                    pred_results[obj_id] = {
                        "label": label_map[pred],
                        "confidence": round(float(probs[pred]), 2),
                        "feature": obj_group[features].values[0],
                        "x": x_val,
                        "y": y_val
                    }
            # Apply clustering on predicted objects
            if not pred_results:
                continue

            positions = np.array([[v["x"], v["y"]] for v in pred_results.values()])

            feats = np.array([v["feature"] for v in pred_results.values()])
            db = DBSCAN(eps=20.0, min_samples=1).fit(positions)
            labels = db.labels_

            cluster_centroids_list = []
            cid_to_idx = {}
            cluster_preds = {}
            obj_ids = list(pred_results.keys())

            for cid in set(labels):
                idxs = np.where(labels == cid)[0]
                cluster_positions = positions[idxs]
                cluster_feats = feats[idxs]
                centroid = np.mean(cluster_positions, axis=0)
                
                cluster_centroids_list.append(centroid)
                cid_to_idx[cid] = len(cluster_centroids_list) - 1
                
                cluster_preds[cid] = [
                    {
                        "feature": feats[i],
                        "x": float(positions[i][0]),
                        "y": float(positions[i][1]),
                        "label": pred_results[obj_ids[i]]['label'],
                        "confidence": pred_results[obj_ids[i]]['confidence']
                    }
                    for i in idxs
                ]

            # === Takip eşleşmesi ===
            matches, unmatched = match_clusters_to_tracks(cluster_centroids_list, track_centroids)
            new_centroids = {}
            track_data = []

            used_track_ids = set(matches.values())
            reverse_matches = {v: k for k, v in matches.items()}

            for cid, obj_list in cluster_preds.items():
                if cid in matches:
                    track_id = matches[cid]
                    centroid = cluster_centroids_list[cid]
                else:
                    # Daha önce var olan track ile benzerse eşleştir, değilse yeni track verme
                    centroid = np.mean([[obj["x"], obj["y"]] for obj in obj_list], axis=0)
                    distances = {
                        tid: np.linalg.norm(centroid - np.array(pos))
                        for tid, pos in track_centroids.items()
                        if tid not in used_track_ids
                    }
                    if distances:
                        closest_tid = min(distances, key=distances.get)
                        if distances[closest_tid] < 1.0:
                            track_id = closest_tid
                            used_track_ids.add(track_id)
                        else:
                            track_id = next_track_id
                            next_track_id += 1
                    else:
                        track_id = next_track_id
                        next_track_id += 1

                new_centroids[track_id] = centroid

                for obj in obj_list:
                    track_data.append({
                        "id": int(cid),
                        "x": obj["x"],
                        "y": obj["y"],
                        "label": obj["label"],
                        "confidence": obj["confidence"]
                    })

            track_centroids = new_centroids

            for tid, pos in new_centroids.items():
                track_history[tid] = (pos, frame_counter)
            
            if track_data:
                json_msg = str(track_data).replace("'", '"')
                tcp_client.sendall(json_msg.encode())

            track_centroids = {
                tid: pos for tid, (pos, last_seen) in track_history.items()
                if frame_counter - last_seen <= TRACK_MEMORY
            }
            frame_counter += 1
            pred_results.clear()
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\U0001F6D1 Interrupted by user.")
except Exception as e:
    print(f"❌ Runtime error: {e}")
finally:
    tcp_client.close()
    proc.terminate()