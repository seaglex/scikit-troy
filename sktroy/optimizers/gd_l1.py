'''
gradient descend - iteration do not change the sign of weights
inspired by the OWLQN algorithm
'''

import numpy as np


def fmin_owlqn(f, x0, fprime, T, eps, max_iteration):
    old_x = x0
    old_y = f(x0)
    x_hists = [old_x]
    y_hists = [old_y]
    for itr in range(max_iteration):
        grad = fprime(old_x)
        new_x = old_x - eps * (grad + T * np.sign(old_x))
        for n in range(len(old_x)):
            if old_x[n] == 0:
                if grad[n] + T >= 0 and grad[n] - T <= 0:
                    new_x[n] = 0
            elif new_x[n] * old_x[n] < 0:
                new_x[n] = 0
        new_y = f(new_x)
        x_hists.append(new_x)
        y_hists.append(new_y)
        old_x = new_x
        old_y = new_y
    return old_x, old_y, x_hists, y_hists

