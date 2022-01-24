# Utility methods

import numpy as np

def numerical(vector):
    positions = []
    vals = []
    for (position, elt) in enumerate(vector):
        if np.isinf(np.abs(elt)):
            positions.append(position)
            vals.append("inf")
        if np.isnan(elt):
            positions.append(position)
            vals.append("nan")
    return len(positions) != 0, positions, vals