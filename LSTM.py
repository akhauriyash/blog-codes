import random, math
import numpy as np
def randomarr(x, y, *args):
    np.random.seed(0)
    return (np.random.rand(*args))*(y-x) +x
def sig(x):
    return (1/(1+np.exp(-x)))
def sigderiv(values):
    return values*(1-values)
def tanhderiv(values):
    return (1-values**2)
