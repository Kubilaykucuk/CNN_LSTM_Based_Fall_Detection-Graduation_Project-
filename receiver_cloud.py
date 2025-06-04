"""
receiver_cloud.py

This script listens for posture tracking data over TCP,
displays real-time visuals using Tkinter + Matplotlib,
and publishes fall and movement events to ThingsBoard using MQTT.
"""

import socket
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import matplotlib.cm as cm
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import paho.mqtt.client as mqtt

# === MQTT Configuration ===
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "your_thingsboard_token")
BROKER = os.getenv("BROKER", "demo.thingsboard.io")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
TOPIC = os.getenv("TOPIC", "v1/devices/me/telemetry")

mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(ACCESS_TOKEN)
mqtt_client.connect(BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()
print("✅ MQTT connected to ThingsBoard.")


# === TCP Configuration ===
TCP_HOST = os.getenv("RECEIVER_HOST", "0.0.0.0")
TCP_PORT = int(os.getenv("RECEIVER_PORT", 8000))
MAX_TRAIL = 100
MAX_FALLS = 4
FALL_DIST_THRESH = 0.5  # meters

# === Buffers & State ===
id_to_color = {}
fall_points = deque(maxlen=20)
fall_points_per_id = defaultdict(list)
fall_markers = []
track_lines = {}
track_dots = {}
track_labels = {}
track_history = defaultdict(lambda: deque(maxlen=MAX_TRAIL))
fall_clusters_per_id = defaultdict(list)
fall_points_per_id = defaultdict(lambda: deque(maxlen=100))

def is_similar_point(p1, p2, threshold=FALL_DIST_THRESH):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < threshold

def get_color_for_id(track_id):
    if track_id not in id_to_color:
        cmap = cm.get_cmap('tab20')
        color_index = len(id_to_color) % 20
        id_to_color[track_id] = cmap(color_index)
    return id_to_color[track_id]

def update_plot(ax, tracks, current_filter):
    seen_ids = set()
    filter_id = current_filter.get()

    if tracks:
        x_vals = [t['x'] for t in tracks]
        max_abs = max(abs(min(x_vals)), abs(max(x_vals))) + 10
        ax.set_xlim(-max_abs, max_abs)

    for tid in track_lines:
        visible = (filter_id == "ALL" or str(tid) == filter_id)
        track_lines[tid].set_visible(visible)
        track_dots[tid].set_visible(visible)
        track_labels[tid].set_visible(visible)

    for t in tracks:
        track_id = t['id']
        x, y = t['x'], t['y']
        label = t['label']
        confidence = t['confidence']
        seen_ids.add(track_id)

        if filter_id != "ALL" and int(filter_id) != track_id:
            continue

        color = get_color_for_id(track_id)

        # === FALL POINT LOGIC ===
        if label == "Falling":
            new_point = (x, y)
            updated = False
            for i, (fp, count) in enumerate(fall_clusters_per_id[track_id]):
                if is_similar_point(fp, new_point):
                    new_avg = tuple(np.mean([fp, new_point], axis=0))
                    fall_clusters_per_id[track_id][i] = (new_avg, count + 1)
                    updated = True
                    if count + 1 >= MAX_FALLS:
                        fall_points.append(new_avg)
                        fall_points_per_id[track_id].append(new_avg)

                        # === SEND FALL POINT TO CLOUD ===
                        fall_data = {
                            "type": "fall",
                            "id": track_id,
                            "x": round(new_avg[0], 2),
                            "y": round(new_avg[1], 2),
                            "label": label,
                            "confidence": confidence
                        }
                        mqtt_client.publish(TOPIC, json.dumps({f"fall_{track_id}": fall_data}))
                    break
            if not updated:
                fall_clusters_per_id[track_id].append((new_point, 1))

        track_history[track_id].append((x, y))
        x_vals, y_vals = zip(*track_history[track_id])

        if track_id not in track_lines:
            line, = ax.plot(x_vals, y_vals, '--', color=color, alpha=0.7)
            dot, = ax.plot([x], [y], 'o', color=color, markersize=10)
            text = ax.text(x + 0.5, y + 0.5, f"ID:{track_id} {label} ({confidence})", fontsize=9)
            track_lines[track_id] = line
            track_dots[track_id] = dot
            track_labels[track_id] = text
        else:
            track_lines[track_id].set_data(x_vals, y_vals)
            track_dots[track_id].set_data([x], [y])
            track_labels[track_id].set_position((x + 0.5, y + 0.5))
            track_labels[track_id].set_text(f"ID:{track_id} {label} ({confidence})")

        # === SEND TRACK TO CLOUD ===
        track_data = {
            "type": "track",
            "id": track_id,
            "x": round(x, 2),
            "y": round(y, 2),
            "label": label,
            "confidence": confidence
        }
        mqtt_client.publish(TOPIC, json.dumps({f"track_{track_id}": track_data}))

    for tid in list(track_history):
        if tid not in seen_ids and track_history[tid]:
            if filter_id != "ALL" and int(filter_id) != tid:
                continue
            x_vals, y_vals = zip(*track_history[tid])
            if tid in track_lines:
                track_lines[tid].set_data(x_vals, y_vals)
            if tid in track_dots:
                track_dots[tid].set_data([x_vals[-1]], [y_vals[-1]])

    for marker in fall_markers:
        marker.remove()
    fall_markers.clear()

    if filter_id == "ALL":
        for x, y in fall_points:
            marker, = ax.plot(x, y, 'x', color='black', markersize=12, markeredgewidth=2)
            fall_markers.append(marker)
    else:
        fid = int(filter_id)
        for x, y in fall_points_per_id[fid]:
            marker, = ax.plot(x, y, 'x', color='black', markersize=12, markeredgewidth=2)
            fall_markers.append(marker)

def run_ui(conn):
    root = tk.Tk()
    root.title("Radar Tracker UI")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = int(screen_width * 0.75)
    window_height = int(screen_height * 0.75)
    root.geometry(f"{window_width}x{window_height}")

    current_filter = tk.StringVar(value="ALL")

    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=0)

    fig, ax = plt.subplots(figsize=(window_width / 100, window_height / 130))
    ax.set_xlim(-60, 60)
    ax.set_ylim(0, 10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Real-Time Radar Posture Tracking")
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    control_canvas = tk.Canvas(root, width=200)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=control_canvas.yview)
    scrollable_frame = ttk.Frame(control_canvas)
    scrollable_frame.bind("<Configure>", lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all")))
    control_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    control_canvas.configure(yscrollcommand=scrollbar.set)
    control_canvas.grid(row=0, column=1, sticky="ns", padx=(0, 5), pady=10)
    scrollbar.grid(row=0, column=2, sticky="ns", pady=10)

    inner_frame = scrollable_frame
    ttk.Label(inner_frame, text="Select Object ID:").pack(anchor="n", pady=(0, 10))
    id_buttons = {}
    selected_button = [None]

    def make_callback(oid):
        def on_click():
            current_filter.set(oid)
            if selected_button[0] is not None:
                selected_button[0].configure(style="TButton")
            id_buttons[oid].configure(style="Selected.TButton")
            selected_button[0] = id_buttons[oid]
        return on_click

    def show_all():
        current_filter.set("ALL")
        if selected_button[0] is not None:
            selected_button[0].configure(style="TButton")
        selected_button[0] = None

    ttk.Button(inner_frame, text="Show All", command=show_all).pack(anchor="s", pady=(20, 10))

    style = ttk.Style()
    style.configure("TButton", font=("Segoe UI", 10), padding=6)
    style.configure("Selected.TButton", background="#4287f5", foreground="white", font=("Segoe UI", 10, "bold"))

    def update_ids(tracks):
        active_ids = set(str(t['id']) for t in tracks if -60 <= t['x'] <= 60)
        existing_ids = set(id_buttons.keys())
        for obj_id in sorted(active_ids):
            if obj_id not in id_buttons:
                btn = ttk.Button(inner_frame, text=f"Object {obj_id}", width=20, command=make_callback(obj_id))
                btn.pack(pady=3)
                id_buttons[obj_id] = btn

    def update_loop():
        nonlocal ax
        try:
            data = conn.recv(4096)
            if not data:
                root.after(100, update_loop)
                return
            buffer = data.decode()
            while "[" in buffer and "]" in buffer:
                start = buffer.find("[")
                end = buffer.find("]") + 1
                json_chunk = buffer[start:end]
                buffer = buffer[end:]
                try:
                    tracks = json.loads(json_chunk)
                    update_plot(ax, tracks, current_filter)
                    update_ids(tracks)
                    canvas.draw()
                except json.JSONDecodeError:
                    continue
        except:
            pass
        root.after(100, update_loop)

    update_loop()
    root.mainloop()

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((TCP_HOST, TCP_PORT))
        s.listen()
        print(f"📡 Listening for incoming track data on {TCP_HOST}:{TCP_PORT}")
        conn, addr = s.accept()
        print(f"🔗 Connection from {addr}")
        run_ui(conn)

if __name__ == "__main__":
    main()