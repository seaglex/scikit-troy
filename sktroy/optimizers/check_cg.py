import numpy as np

dim = 3

A = np.asarray([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
v = np.asarray([1, 2, 1]).reshape(dim, 1)

r_hist = [v]
p_hist = [v]
a_hist = []
b_hist = []
last_r = v
last_p = v.copy()
for n in range(dim):
    a = np.dot(last_r.transpose(), last_r) / np.dot(last_p.transpose(), np.dot(A, last_p))
    r = last_r - a * np.dot(A, last_p)
    print("r_k+1 * r_k:", np.dot(last_r.transpose(), r))
    b = np.dot(r.transpose(), r) / np.dot(last_r.transpose(), last_r)
    p = r + b * last_p
    r_hist.append(r)
    print("p_k+1 vs. p0-p_k in A:", end=' ')
    for q in p_hist:
        print(np.dot(p.transpose(), np.dot(A, q)), end=' ')
    print()
    p_hist.append(p)
    a_hist.append(a)
    b_hist.append(b)
    last_r = r
    last_p = p
print(last_r)
