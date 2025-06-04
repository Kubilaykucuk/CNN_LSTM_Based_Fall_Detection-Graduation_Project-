"""
receiver.py

This script acts as a TCP server that receives real-time radar-based posture tracking data
and visualizes it using a live Tkinter GUI embedded with Matplotlib.
It allows object-based filtering and fall detection visualization.
"""

import socket
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict, deque
import matplotlib.cm as cm

# === Configuration ===
HOST = os.getenv("RECEIVER_HOST", "0.0.0.0")
PORT = int(os.getenv("RECEIVER_PORT", 8000))
MAX_TRAIL = 100
FALL_DIST_THRESH = 0.5
MAX_FALLS = 4

# === State ===
id_to_color = {}
fall_points = deque(maxlen=20)          # Global fall marker for ALL view
fall_points_per_id = defaultdict(list)  # Store all falls per object ID
fall_markers = []
track_lines = {}
track_dots = {}
track_labels = {}
track_history = defaultdict(lambda: deque(maxlen=MAX_TRAIL))
fall_clusters_per_id = defaultdict(list)
fall_points_per_id = defaultdict(lambda: deque(maxlen=100))

def is_similar_point(p1, p2, threshold=FALL_DIST_THRESH):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < threshold

# === Color Assignment ===
def get_color_for_id(track_id):
    if track_id not in id_to_color:
        cmap = cm.get_cmap('tab20')
        color_index = len(id_to_color) % 20
        id_to_color[track_id] = cmap(color_index)
    return id_to_color[track_id]

# === Plot Setup ===
def setup_plot():
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(-60, 60)
    ax.set_ylim(0, 10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Real-Time Radar Posture Tracking")
    ax.grid(True)
    return fig, ax

# === Plot Update ===
def update_plot(ax, tracks, current_filter):
    seen_ids = set()
    filter_id = current_filter.get()

    if tracks:
        x_vals = [t['x'] for t in tracks]
        max_abs = max(abs(min(x_vals)), abs(max(x_vals))) + 10
        ax.set_xlim(-max_abs, max_abs)

    # Gizlemeden önce tüm objeleri görünmez yap
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

        # Bu ID'yi çizmeyeceksek geç
        if filter_id != "ALL" and int(filter_id) != track_id:
            continue

        color = get_color_for_id(track_id)
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

# === GUI Setup ===
def run_ui(conn):
    root = tk.Tk()
    root.title("Radar Tracker UI")

    # Dynamically scale window to 75% of screen
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = int(screen_width * 0.75)
    window_height = int(screen_height * 0.75)
    root.geometry(f"{window_width}x{window_height}")

    current_filter = tk.StringVar(value="ALL")

    # === Grid Config ===
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=0)

    # === Plot Setup ===
    fig, ax = plt.subplots(figsize=(window_width / 100, window_height / 130))
    ax.set_xlim(-60, 60)
    ax.set_ylim(0, 10)  # Y starts from 0
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Real-Time Radar Posture Tracking")
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    # === Button Panel ===
    control_canvas = tk.Canvas(root, width=200)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=control_canvas.yview)
    scrollable_frame = ttk.Frame(control_canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: control_canvas.configure(
            scrollregion=control_canvas.bbox("all")
        )
    )

    control_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    control_canvas.configure(yscrollcommand=scrollbar.set)

    control_canvas.grid(row=0, column=1, sticky="ns", padx=(0, 5), pady=10)
    scrollbar.grid(row=0, column=2, sticky="ns", pady=10)

    # bu frame'e butonlar eklenecek
    inner_frame = scrollable_frame

    ttk.Label(inner_frame, text="Select Object ID:").pack(anchor="n", pady=(0, 10))
    id_buttons = {}
    selected_button = [None]  # For button highlight tracking

    def make_callback(oid):
        def on_click():
            current_filter.set(oid)
            if selected_button[0] is not None:
                selected_button[0].configure(style="TButton")
            id_buttons[oid].configure(style="Selected.TButton")
            selected_button[0] = id_buttons[oid]
        return on_click

    # Show All button
    def show_all():
        current_filter.set("ALL")
        if selected_button[0] is not None:
            selected_button[0].configure(style="TButton")
        selected_button[0] = None

    ttk.Button(inner_frame, text="Show All", command=show_all).pack(anchor="s", pady=(20, 10))

    # === Button Styling ===
    style = ttk.Style()
    style.configure("TButton", font=("Segoe UI", 10), padding=6)
    style.configure("Selected.TButton", background="#4287f5", foreground="white", font=("Segoe UI", 10, "bold"))

    # === Dynamic Update ===
    def update_ids(tracks):
        active_ids = set(str(t['id']) for t in tracks if -60 <= t['x'] <= 60)
        existing_ids = set(id_buttons.keys())

        # Add new buttons
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

# === TCP Server Listener ===
def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Listening for data on {HOST}:{PORT}...")
        conn, addr = s.accept()
        print(f"Connection established with {addr}")
        run_ui(conn)

if __name__ == "__main__":
    main()