# Path-following controllers for the Convoy sim.
# Every controller shares the same interface:
#     ctrl.set_path(path)                  path: (N, 2) array of waypoints, in order
#     delta = ctrl.compute(x, y, theta, v) (x, y, theta) is the rear-axle pose
import math
import numpy as np
from scipy.optimize import minimize


def wrap(a):
    """Wrap an angle to [-pi, pi]"""
    return np.arctan2(np.sin(a), np.cos(a))


class PathController:
    """Basic path controller that keeps track of the path and the nearest waypoint index"""

    def __init__(self, window=200):
        self.window = window
        self.path = None
        self.idx = 0

    def set_path(self, path):
        self.path = np.asarray(path, dtype=float)
        self.idx = 0

        # Heading of each segment
        d = np.diff(self.path, axis=0)
        seg_heading = np.arctan2(d[:, 1], d[:, 0])
        self.heading = np.append(seg_heading, seg_heading[-1]) # last waypoint reuses the final segment's heading

        # Cumulative arc length s_i along the path
        seg_len = np.hypot(d[:, 0], d[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(seg_len)])

        # change in heading per unit arc length
        dtheta = wrap(np.diff(self.heading))
        ds = np.maximum(np.append(seg_len[1:], seg_len[-1]), 1e-9)
        self.kappa = np.append(dtheta / ds, 0.0)

    def nearest(self, px, py):
        """Index of the closest forward waypoint"""
        start = self.idx
        seg = self.path[start:start + self.window]
        dists = np.hypot(seg[:, 0] - px, seg[:, 1] - py)
        self.idx = start + int(np.argmin(dists))
        return self.idx


class Stanley(PathController):
    """Steers the front axle onto the path"""

    def __init__(self, k=1.5, ks=0.3, L=0.33, delta_max=math.pi/10, window=200):
        super().__init__(window)
        self.k = k
        self.ks = ks
        self.L = L
        self.delta_max = delta_max
        self.e = 0.0

    def compute(self, x, y, theta, v) -> float:
        fx = x + self.L * math.cos(theta)
        fy = y + self.L * math.sin(theta)

        i = self.nearest(fx, fy)
        px, py = self.path[i]
        theta_p = self.heading[i]

        psi_e = wrap(theta_p - theta)
        self.e = (fx - px) * math.sin(theta_p) - (fy - py) * math.cos(theta_p)  # >0: car right of path

        delta = psi_e + math.atan2(self.k * self.e, self.ks + v)
        return max(-self.delta_max, min(self.delta_max, delta))


class PathMPC(PathController):
    """Optimizes the next N steering commands at a fixed speed
    so the predicted tracking point stays on the path and aligned with it.
    track_point = "rear": rear axle on the path, body heading along the path.
    track_point = "front": front axle on the path, front wheel direction (theta + delta) along the path."""

    def __init__(self, N=10, dt=0.1, v_ref=0.8, L=0.33, delta_max=math.pi/10,
                 w_p=1.0, w_theta=0.5, w_delta=0.0, w_ddelta=0.1, window=200,
                 track_point="rear"):
        super().__init__(window)
        if track_point not in ("rear", "front"):
            raise ValueError("track_point must be 'rear' or 'front'")
        self.front = (track_point == "front")
        self.N = N
        self.dt = dt
        self.v_ref = v_ref          # speed for the reference spacing and the rollout
        self.L = L
        self.delta_max = delta_max

        # Weights
        self.w_p = w_p
        self.w_theta = w_theta
        self.w_delta = w_delta
        self.w_ddelta = w_ddelta

        self.u_prev = None          # last optimal sequence, for the warm start
        self.delta_last = 0.0       # last applied command, delta_{-1} in the smoothness term
        self.pred_world = None      # last predicted trajectory in world frame (for plotting/debug)
        self.e = 0.0                # signed lateral error of the rear axle right now

    # Model
    def rollout(self, u):
        """Predicted body-frame states after each command, starting from (0, 0, 0)"""
        step = self.v_ref * self.dt
        theta_after = np.cumsum(step / self.L * np.tan(u))
        theta_before = np.concatenate([[0.0], theta_after[:-1]])
        x = np.cumsum(step * np.cos(theta_before))
        y = np.cumsum(step * np.sin(theta_before))
        return x, y, theta_after

    # Cost
    def cost(self, u, rx, ry, rth, dff):
        x, y, th = self.rollout(u)

        if self.front:
            # Move the tracked point to the front axle
            x = x + self.L * np.cos(th)
            y = y + self.L * np.sin(th)
            dir_ = th + u
        else:
            dir_ = th

        # Lateral error
        e = -(x - rx) * np.sin(rth) + (y - ry) * np.cos(rth)
        th_err = wrap(dir_ - rth)
        du = np.diff(np.concatenate([[self.delta_last], u]))

        return (self.w_p * np.sum(e**2)
                + self.w_theta * np.sum(th_err**2)
                + self.w_delta * np.sum((u - dff)**2)
                + self.w_ddelta * np.sum(du**2))

    # Main
    def compute(self, x, y, theta, v) -> float:
        # Locate tracked point on the path
        c, s = math.cos(theta), math.sin(theta)
        tx, ty = (x + self.L * c, y + self.L * s) if self.front else (x, y)
        i0 = self.nearest(tx, ty)
        px, py = self.path[i0]
        self.e = (tx - px) * math.sin(self.heading[i0]) - (ty - py) * math.cos(self.heading[i0])

        # Generate reference points spaced by the distance traveled per step, by arc length
        targets = self.s[i0] + self.v_ref * self.dt * np.arange(1, self.N + 1)
        ref_idx = np.minimum(np.searchsorted(self.s, targets), len(self.s) - 1)
        ref = self.path[ref_idx]
        ref_th = self.heading[ref_idx]
        dff = np.arctan(self.L * self.kappa[ref_idx])     # feedforward steering for each ref point

        # Transform to the body frame (origin at the rear axle)
        dx, dy = ref[:, 0] - x, ref[:, 1] - y
        rx = c * dx + s * dy
        ry = -s * dx + c * dy
        rth = wrap(ref_th - theta)

        # Previous plan shifted forward one step, last value repeated
        if self.u_prev is None:
            u0 = np.clip(dff, -self.delta_max, self.delta_max)
        else:
            u0 = np.append(self.u_prev[1:], self.u_prev[-1])

        # Optimize the steering sequence within the steering limits
        bounds = [(-self.delta_max, self.delta_max)] * self.N
        res = minimize(self.cost, u0, args=(rx, ry, rth, dff), bounds=bounds, method="L-BFGS-B")
        u = res.x

        # Save the predicted trajectory
        bx, by, _ = self.rollout(u)
        self.pred_world = np.column_stack([x + c * bx - s * by, y + s * bx + c * by])

        # Apply only the first command and keep the rest for next time
        self.u_prev = u
        self.delta_last = float(u[0])
        return self.delta_last