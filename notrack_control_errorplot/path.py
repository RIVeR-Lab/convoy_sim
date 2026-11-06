import math
import numpy as np

def generatePath(length, A=1.0, lam=8.0, ds=0.01) -> list[list[int]]:
    s = np.arange(0, length + ds, ds)
    x = s
    y = -1 + A*(1 - np.cos((2*np.pi*s)/lam))

    return np.column_stack([x, y])