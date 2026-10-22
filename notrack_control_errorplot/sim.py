# A simulator for off-track convoy operations.  Runs the sim and saves results to a file; see plot.py for plotting.
import time
import math
from collections import deque
from enum import IntEnum

import numpy as np

from vehicle import Vehicle
from ekf import EKF
from mpc import MPC

class LEADER_MODES(IntEnum):
    MANUAL = 0
    CIRCLE = 1
    RANDOM = 2


# Config
num_followers = 4
following_distance = 1.0
sit_time = 2

leader_mode = LEADER_MODES.RANDOM
leader_delta = math.pi/12
leader_vel = 0.8
leader_period = 4.0
leader_max_angle = np.pi/10

sim_duration = 60  # seconds of simulated time
sim_dt = 0.1
output_file = r"./notrack_control_errorplot/data/leader_random_reach5.npz"  # where results are saved for plot.py
max_past_poses = 10000
seed = 2025

q_x = 0.01
q_y = 0.01
q_theta = 0.01
q_v = 0.01

r_lidar_x = 0.05
r_lidar_y = 0.05
r_pozyx_r = 0.05

lidar_fov = math.pi * 160/180

do_enable_crosstalk = True
do_debug_print = False

do_drop_lidar = False
do_drop_pozyx = False





# Setup
vehicles = {
    0: Vehicle(id=0, x=0, y=-1, theta=0, v=0),
    # 1: Vehicle(id=1, x=-0.5, y=-1, theta=0, v=0),
    # 2: Vehicle(id=2, x=-1, y=-1, theta=0, v=0),
}
for i in range(num_followers):
    k = i+1
    vehicles[k] = Vehicle(id=k, x=-following_distance*k, y=-1, theta=0, v=0)

filters = {
    # 1: EKF(time=time.time(), veh_id=1, target_id=0),
    # 2: EKF(time=time.time(), veh_id=2, target_id=1),
}
for i in range(num_followers):
    k = i+1
    filters[k] = EKF(time=0.0, veh_id=k, target_id=i, x0 = -following_distance*k,
                     qx=q_x, qy=q_y, qtheta=q_theta, qv=q_v)

controllers = {
    # 1: MPC(),
    # 2: MPC(),
}
for i in range(num_followers):
    k = i+1
    controllers[k] = MPC()

inputs = {
    0: {'delta': 0.0, 'vel': 0.0}, 
    # 1: {'delta': 0.0, 'vel': 0.0},  
    # 2: {'delta': 0.0, 'vel': 0.0},  
}
for i in range(num_followers):
    k = i+1
    inputs[k] = {'delta': 0.0, 'vel': 0.0}



# More Setup
# Independent streams so sensor noise draws (which depend on FOV) never shift the random leader's commands
sensor_seed, leader_seed = np.random.SeedSequence(seed).spawn(2)
nprandom = np.random.default_rng(sensor_seed)
leader_random = np.random.default_rng(leader_seed)

trails = {id: deque(maxlen=10000) for id in vehicles}

# Point behind the leader's start along its initial heading; used only for lateral deviation, never plotted
_lead = vehicles[0]
_back = following_distance * num_followers + 1.0
path_prefix = np.array([[_lead.x - _back*math.cos(_lead.theta), _lead.y - _back*math.sin(_lead.theta)]])





leader_manual = {'delta': 0.0, 'vel': 0.0}  # manual mode needs a live window, so it just idles

follower_ids = list(filters.keys())
history = {
    "t": [],
    "poses": [],  # per step: (x, y, theta) for each vehicle, ordered by id
    "est": {k: [] for k in follower_ids},
    "trk": {k: [] for k in follower_ids},
    "lat": {k: [] for k in follower_ids},
}







# Helpers

def get_lidar_meas(veh_id_center, veh_id_target) -> tuple:
    """Generate a noisy lidar centroid"""
    if do_drop_lidar:
        return None, None

    veh_target = vehicles[veh_id_target]
    veh_center = vehicles[veh_id_center]

    dx = veh_target.x - veh_center.x
    dy = veh_target.y - veh_center.y

    # Check if target is within FOV before adding noise
    bearing = math.atan2(dy, dx) - veh_center.theta
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))  # wrap to [-pi, pi]

    if abs(bearing) > lidar_fov / 2:
        return None, None

    dx += nprandom.normal(0, r_lidar_x)
    dy += nprandom.normal(0, r_lidar_y)

    # Rotate into follower body frame
    c, s = math.cos(-veh_center.theta), math.sin(-veh_center.theta)
    x_meas = c*dx - s*dy
    y_meas = s*dx + c*dy

    return x_meas, y_meas

def get_pozyx_meas(veh_id_center, veh_id_target) -> float:
    """Generate a noisy lidar centroid"""
    if do_drop_pozyx:
        return None
    
    veh_target = vehicles[veh_id_target]
    veh_center = vehicles[veh_id_center]

    dx = veh_target.x - veh_center.x
    dy = veh_target.y - veh_center.y
    r = math.sqrt(dx**2 + dy**2) + nprandom.normal(0,r_pozyx_r)
    return r




def wrap_angle(a) -> float:
    return math.atan2(math.sin(a), math.cos(a))

def true_relative(veh_id_center, veh_id_target) -> tuple:
    """True pose of the target in the center vehicle's body frame"""
    veh_target = vehicles[veh_id_target]
    veh_center = vehicles[veh_id_center]

    dx = veh_target.x - veh_center.x
    dy = veh_target.y - veh_center.y
    c, s = math.cos(-veh_center.theta), math.sin(-veh_center.theta)
    return c*dx - s*dy, s*dx + c*dy, wrap_angle(veh_target.theta - veh_center.theta)

LAT_WINDOW = 200            # Segments searched on each side of the last matched segment
_lat_progress = {}          # Last matched segment index per follower

def lateral_deviation(point, trail, key=None) -> float:
    """Min distance from a point to the polyline of a trail, searching only near the previous match
    so a path that loops back on itself doesn't match a far-away (in time) portion"""
    if len(trail) < 1:
        return float('nan')

    # Hidden extension of the path backwards from its start so followers behind the start measure cross-track
    pts = np.vstack([path_prefix, np.asarray(trail)])
    a, b = pts[:-1], pts[1:]

    # Restrict to a window of segments around the last match
    last = _lat_progress.get(key, 0)
    lo, hi = max(0, last - LAT_WINDOW), min(len(a), last + LAT_WINDOW + 1)
    a, b = a[lo:hi], b[lo:hi]

    ab = b - a
    ap = np.asarray(point) - a
    denom = np.einsum('ij,ij->i', ab, ab)
    u = np.divide(np.einsum('ij,ij->i', ap, ab), denom, out=np.zeros_like(denom), where=denom > 1e-12)
    closest = a + np.clip(u, 0, 1)[:, None] * ab
    d = np.linalg.norm(closest - point, axis=1)
    i = int(np.argmin(d))
    _lat_progress[key] = lo + i
    return float(d[i])

def compute_metrics() -> tuple:
    """Per-follower estimator error, tracking error, and lateral deviation"""
    est, trk, lat = {}, {}, {}
    for k, ekf in filters.items():
        tx, ty, ttheta = true_relative(ekf.veh_id, ekf.target_id)

        # Estimator error over position components
        est[k] = math.hypot(ekf.x[0]-tx, ekf.x[1]-ty)

        # Tracking error: distance to the target point computed from true state
        saved_target = controllers[k].target
        gx, gy, _ = controllers[k].get_target_point(tx, ty, ttheta, d=following_distance)
        controllers[k].target = saved_target
        trk[k] = math.hypot(gx, gy)

        # Lateral deviation from the leader's trail
        lat[k] = lateral_deviation((vehicles[k].x, vehicles[k].y), trails[0], key=k)
    return est, trk, lat

def get_leader_inputs(t):
    period = leader_period
    t = t % period
    
    # if t < 10:
    #     return {'delta': math.pi/6, 'vel': 0.5}
    # elif t < 20:
    #     return {'delta': math.pi/6, 'vel': 0.5}

    if leader_mode == LEADER_MODES.CIRCLE:
        return {'delta': leader_delta, 'vel': leader_vel}
    elif leader_mode == LEADER_MODES.RANDOM and t < 0.2:
        return {'delta': leader_random.uniform(-leader_max_angle, leader_max_angle), 'vel': leader_random.uniform(0,0.8)}


# Main loop

try:
    # Simulated clock: runs as fast as possible, not tied to wall time
    start_time = 0.0  # simulated time, independent of the wall clock so runs are reproducible
    wall_start = time.time()
    current_dt = sim_dt
    num_steps = int(round(sim_duration / sim_dt))

    for step in range(1, num_steps + 1):
        current_time = start_time + step * sim_dt

        if True:

            # State Estimate Updates
            for ekf in filters.values():
                if ekf is not None:
                    if do_enable_crosstalk:
                        delta, vel = inputs[ekf.veh_id].values()
                        delta_t, vel_t = inputs[ekf.target_id].values()
                    else: 
                        delta = None
                        vel = None
                        delta_t = None
                        vel_t = None
                    
                    # Lidar update
                    x_meas_lidar, y_meas_lidar = get_lidar_meas(ekf.veh_id, ekf.target_id)
                    ekf.update_lidar(x_meas_lidar, y_meas_lidar, current_time, delta=delta, vel=vel, delta_t=delta_t, vel_t=vel_t)

                    # POZYX update
                    r_meas_pozyx = get_pozyx_meas(ekf.veh_id, ekf.target_id)
                    ekf.update_pozyx(r_meas_pozyx, current_time, delta=delta, vel=vel, delta_t=delta_t, vel_t=vel_t)

                    # Final predict update
                    ekf.update_predict(current_time, delta=delta, vel=vel, delta_t=delta_t, vel_t=vel_t)


            # Control inputs
            
            ## Leader control
            if leader_mode == LEADER_MODES.MANUAL:
                inputs[0] = leader_manual
            else:
                new_input = get_leader_inputs(current_time - start_time)
                if new_input is not None:
                    inputs[0] = new_input

            ## Controller Updates
            for id,mpc in controllers.items():
                if mpc is not None:
                    x_targ, y_targ, theta_targ, vel_targ = filters[id].x
                    vel, delta = mpc.solve(x_targ, y_targ, theta_targ, d=following_distance)
                    inputs[id] = {"delta": delta, "vel": vel}

            if current_time - start_time < sit_time:
                    for id,input in inputs.items():
                        inputs[id] = {"delta": 0, "vel": 0.005}
            

            # True dynamics
            for veh in vehicles.values():
                delta, vel = inputs[veh.id].values()
                veh.step(vel, delta, current_dt)

            # Record poses
            for id, veh in vehicles.items():
                trails[id].append((veh.x, veh.y))

            # Record metrics
            est, trk, lat = compute_metrics()
            history["t"].append(current_time - start_time)
            history["poses"].append([(veh.x, veh.y, veh.theta) for veh in vehicles.values()])
            for k in follower_ids:
                history["est"][k].append(est[k])
                history["trk"][k].append(trk[k])
                history["lat"][k].append(lat[k])

            # Progress indicator
            frac = step / num_steps
            elapsed = time.time() - wall_start
            eta = elapsed / frac - elapsed
            bar = '#' * int(30 * frac)
            print(f"\r[{bar:<30}] {100*frac:5.1f}%  sim {step*sim_dt:6.1f}/{sim_duration}s  elapsed {elapsed:5.1f}s  eta {eta:5.1f}s", end='', flush=True)
            if step == num_steps:
                print()




            if do_debug_print:
                
                print("VEHICLES")
                print(f"time: id | true x | true y | true theta | true v")
                for id, veh in vehicles.items():
                    print(f"{current_time}s: {id} | {veh.x} | {veh.y} | {veh.theta} | {veh.v}")

                print("\nFILTERS")
                print(f"time: id | ekf x | ekf y | ekf theta | ekf v")
                for id, ekf in filters.items():
                    print(f"{current_time}s: {id} | {ekf.x[0]} | {ekf.x[1]} | {ekf.x[2]} | {ekf.x[3]}")
                          
                


except KeyboardInterrupt:
    print("\nInterrupted, saving results so far")

except Exception as e:
    raise

finally:
    print("Shutting down sim")

# Save results
if history["t"]:
    np.savez(output_file,
        t=np.array(history["t"]),
        poses=np.array(history["poses"]),  # (steps, vehicles, 3)
        vehicle_ids=np.array(list(vehicles.keys())),
        follower_ids=np.array(follower_ids),
        est=np.array([history["est"][k] for k in follower_ids]).T,  # (steps, followers)
        trk=np.array([history["trk"][k] for k in follower_ids]).T,
        lat=np.array([history["lat"][k] for k in follower_ids]).T,
        sim_dt=sim_dt,
        lidar_fov=lidar_fov,
    )
    print(f"Saved results to {output_file}")
