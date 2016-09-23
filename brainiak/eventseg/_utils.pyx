from libc.math cimport log
import numpy as np
cimport numpy as np

def masked_log(x):
    """log(x) for all x <= 0 and -Inf otherwise"""
    y = np.empty(x.shape, dtype=x.dtype)
    lim = x.shape[0]
    for i in range(lim):
      if x[i] >= 0:
        y[i] = log(x[i])
      else:
        y[i] = float('-inf')
    return y

