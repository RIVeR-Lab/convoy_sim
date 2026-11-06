import math
import numpy as np

class Stanley:
    def __init__(self, k=1.5, ks=0.3, L=0.33, delta_max=math.pi/10, window=200):
        self.k = k
        self.ks = ks
        self.L = L
        self.delta_max = delta_max
        self.window = window
        self.idx = 0
        self.path = None

    def set_path(self, path):        # (N, 2) array
        self.path = np.asarray(path)
        self.idx = 0

    def wrap(self, a):
        return math.atan2(math.sin(a), math.cos(a))

    def compute(self, x, y, theta, v) -> float:
        # front axle point
        f_axel_pt = [x + self.L*math.cos(theta), y + self.L*math.sin(theta)]

        # nearest waypoint in path
        best_i = 0
        best_d = float('inf')
        for i, pt in enumerate(self.path[self.idx:self.idx + self.window]):
            curr_d = np.linalg.norm(f_axel_pt - pt)
            if curr_d< best_d:
                best_i = i
                best_d = curr_d

        self.idx = self.idx + best_i
        nearest = self.path[self.idx]

        # theta_p from segment
        if self.idx == len(self.path)-1:
           theta_p = math.atan2(nearest[1] - self.path[self.idx-1][1], nearest[0] - self.path[self.idx - 1][0])
        else: 
            theta_p = math.atan2(self.path[self.idx + 1][1] - nearest[1], self.path[self.idx + 1][0] - nearest[0])

        psi_e = self.wrap(theta_p - theta)
        e = (f_axel_pt[0] - nearest[0])*math.sin(theta_p) + -1*(f_axel_pt[1] - nearest[1])*math.cos(theta_p)

        delta = psi_e +  math.atan2(self.k*e, self.ks + v)
        if delta > self.delta_max:
            delta = self.delta_max
        elif delta < -1*self.delta_max:
            delta = -1*self.delta_max

        return delta