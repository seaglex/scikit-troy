'''
conjugate gradient descent
'''
import numpy as np


def line_search_armijo(x0, f, p, f_old_val, fprime_old_val, alpha_init=1, beta=0.1):
    '''
    f(x + α*p) <= f(x) + β▽f(x)ᵀ(α*p)
    '''
    tao = 0.5
    dim = max(p.shape)
    min_delta_val = beta * np.dot(p.reshape(1, dim), fprime_old_val.reshape(dim, 1))
    alpha = alpha_init
    for itr in range(30):
        f_val = f(x0 + alpha*p)
        if f_val <= f_old_val + alpha * min_delta_val:
            return alpha, f_val
        alpha *= tao
    return 0, f_old_val


def fmin_cgd(f, x0, fprime, max_iteration):
    old_x = x0
    old_y = f(x0)
    x_hists = [old_x]
    y_hists = [old_y]
    old_grad = fprime(old_x)
    old_p = -old_grad
    for itr in range(max_iteration):
        alpha, f_val = line_search_armijo(old_x, f, old_p, old_y, old_grad)
        new_x = old_x + alpha * old_p
        new_y = f_val
        new_grad = fprime(new_x)
        beta = np.dot(new_grad, new_grad-old_grad) / np.dot(old_grad, old_grad)
        new_p = -new_grad + beta * old_p
        x_hists.append(new_x)
        y_hists.append(new_y)
        old_x = new_x
        old_y = new_y
        old_grad = new_grad
        old_p = new_p
    return old_x, old_y, x_hists, y_hists
