# Plot results saved by sim.py
import sys
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


# Config
input_file = r"./notrack_control_errorplot/data/leader_random_reach5.npz"

fig_size = (9*0.95, 5*0.95)          # figure size in inches (width, height)
arrow_len = 0.25
world_size = 2
do_autoscale_world = True   # fit the world plot to the paths taken (ignores world_size)
autoscale_margin = 0.1      # fractional padding around the paths
time_label_interval = 10    # seconds between time labels on the leader's trail
do_visualize_lidar_fov = False

colors = {0: "#FF960D", 1: "#777777", 2: "#2BFF60", 3: "#002FFF", 4: "#F7052D", 5: "#00CE0A"}


# Load
data = np.load(input_file)
t = data["t"]
poses = data["poses"]                # (steps, vehicles, 3)
vehicle_ids = list(data["vehicle_ids"])
follower_ids = list(data["follower_ids"])
sim_dt = float(data["sim_dt"])
lidar_fov = float(data["lidar_fov"])
metrics = [("est", "Estimator Error"), ("trk", "Tracking Error"), ("lat", "Lateral Deviation")]


def world_z(veh_id, base=0) -> float:
    """Draw order for the world view: lower vehicle ids on top"""
    return base + len(vehicle_ids) - veh_id


# One window: world view on the left, error metrics stacked on the right
fig = plt.figure(figsize=fig_size)
gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1])
ax = fig.add_subplot(gs[:, 0])
ax_metrics = [fig.add_subplot(gs[0, 1])]
ax_metrics += [fig.add_subplot(gs[i, 1], sharex=ax_metrics[0]) for i in (1, 2)]


# Metrics
for ax_m, (key, title) in zip(ax_metrics, metrics):
    for col, k in enumerate(follower_ids):
        ax_m.plot(t, data[key][:, col], color=colors[k], label=f"veh {k}", zorder=len(follower_ids) - k + 2)
    ax_m.set_title(title)
    ax_m.set_ylabel("Error (m)")
    ax_m.grid(alpha=0.3)
    ax_m.tick_params(labelbottom=(ax_m is ax_metrics[-1]))
ax_metrics[-1].set_xlabel("Time (s)")


# World view
for i, vid in enumerate(vehicle_ids):
    ax.plot(poses[:, i, 0], poses[:, i, 1], linestyle='-', alpha=0.3, color=colors[vid], zorder=world_z(vid))

for i, vid in enumerate(vehicle_ids):
    x, y, theta = poses[-1, i]

    if do_visualize_lidar_fov:
        ax.add_patch(Wedge((x, y), 2.5, math.degrees(theta - lidar_fov/2), math.degrees(theta + lidar_fov/2),
                           color=colors[vid], alpha=0.05, zorder=world_z(vid)))

    ax.plot(x, y, marker='o', linestyle="", color=colors[vid], label=f"veh {vid}", zorder=world_z(vid, 10))
    ax.annotate("", xytext=(x, y), xy=(x + arrow_len*math.cos(theta), y + arrow_len*math.sin(theta)),
                arrowprops=dict(arrowstyle="->"), zorder=world_z(vid, 10))

# Time labels along the leader's trail
leader = vehicle_ids.index(0)
label_every = int(round(time_label_interval / sim_dt))
for step in range(label_every, len(t) + 1, label_every):
    mx, my, _ = poses[step - 1, leader]
    ax.plot(mx, my, marker='o', markersize=4, linestyle="", color='k', zorder=world_z(0, 20))
    ax.annotate(f"{step * sim_dt:g}s", (mx, my), xytext=(4, 4), textcoords='offset points', fontsize=8, zorder=world_z(0, 20))

if do_autoscale_world:
    pts = poses[:, :, :2].reshape(-1, 2)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2
    half = max((hi - lo).max() / 2, 0.5) * (1 + autoscale_margin)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
else:
    ax.set_xlim(-world_size, world_size)
    ax.set_ylim(-world_size, world_size)
ax.set_aspect('equal', adjustable='box')
ax.set_title("Simulator")
ax.legend(loc='best', fontsize=7, framealpha=0.7)

# Error summary
for key, title in metrics:
    print(f"{title} (m)")
    for col, k in enumerate(follower_ids):
        v = data[key][:, col]
        print(f"  veh {k}: max {np.nanmax(v):.4f}  median {np.nanmedian(v):.4f}  mean {np.nanmean(v):.4f}")
    allv = data[key]
    print(f"  overall: max {np.nanmax(allv):.4f}  median {np.nanmedian(allv):.4f}  mean {np.nanmean(allv):.4f}")

fig.tight_layout()
plt.show()
