'''
Proximal gradient method
'''

def fmin_pgm_l1(f, x0, fprime, T, eps, max_iteration):
    old_x = x0
    old_y = f(x0)
    x_hists = [old_x]
    y_hists = [old_y]
    for itr in range(max_iteration):
        grad = fprime(old_x)
        new_x = old_x - eps * grad
        for n in range(len(old_x)):
            if new_x[n] > eps * T:
                new_x[n] -= eps * T
            elif new_x[n] < -eps * T:
                new_x[n] += eps * T
            else:
                new_x[n] = 0
        new_y = f(new_x)
        x_hists.append(new_x)
        y_hists.append(new_y)
        old_x = new_x
        old_y = new_y
    return old_x, old_y, x_hists, y_hists

