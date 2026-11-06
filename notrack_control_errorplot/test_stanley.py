import numpy as np
from path_control import Stanley

path = np.column_stack([np.linspace(0, 10, 1001), np.zeros(1001)])

for y0 in [-0.5, 0.5]:
    ctrl = Stanley(delta_max=10)          # unclamped, to see the raw value
    ctrl.set_path(path)
    d = ctrl.compute(0.0, y0, 0.0, 0.8)
    print(f"y0={y0:+.1f}  idx={ctrl.idx}  delta={d:+.3f}")

    ctrl = Stanley()                      # default clamp at pi/10
    ctrl.set_path(path)
    print(f"          clamped delta={ctrl.compute(0.0, y0, 0.0, 0.8):+.3f}")