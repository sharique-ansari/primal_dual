import numpy as np
import random


def main():
    beta = random.uniform(0.0001, 0.9999)
    gamma = random.uniform(0.000001, 0.9999)
    f = open('cons2.txt')
    A = []
    for l in f:
        A.append([np.float(i) for i in l.split(' ')])
    A = np.array(A)

    f = open('cons3.txt')
    b = []
    for l in f:
        b.append(np.float(l))
    b = np.array(b)

    f = open('cons4.txt')
    c = []
    for l in f:
        c.append(np.float(l))
    c = np.array(c)

    # Problem statement in standard form
    A = np.array(A)
    B = np.array(b)
    C = np.array(c)

    row, col = A.shape

    B = np.expand_dims(B, axis=1)
    C = np.expand_dims(C, axis=1)
    eps = 1.0e-8
    e = np.ones((col, 1))
    x0, s0 = np.ones((col, 1)), np.ones((col, 1))
    y0 = np.zeros((len(B), 1))
    k = 0

    while (True):
        # calculation of dual residuals
        rp = B - A.dot(x0)
        rd = C - A.transpose().dot(y0) - s0
        mu = np.true_divide(x0.transpose().dot(s0), len(s0))

        # Check our termination
        if (np.linalg.norm(rp) or np.linalg.norm(rd) or np.linalg.norm(mu)) < eps:
            break
        else:
            # calculating diagonal matrix of s and x

            S = np.diagflat(s0)
            S_inv = np.linalg.inv(S)
            X = np.diagflat(x0)
            temp = A.dot(S_inv)
            temp2 = X.dot(A.transpose())
            M = temp.dot(temp2)
            asdh = X.dot(rd)
            sdfas = e * 0.3
            temp = X.dot(rd) - e * (gamma * mu)
            asdf = A.dot(S_inv)
            sdasd = asdf.dot(temp)
            r = B + sdasd

            # using dy to calculate ds and dx
            try:
                dy = np.linalg.inv(M).dot(r)
                ds = rd - A.transpose().dot(dy)
                dx = -x0 + np.linalg.inv(S).dot(gamma * mu * e - X.dot(ds))
            except:
                break

            # Step size selection
            alpha_p, alpha_d = 1, 1

            if dx[dx < 0].size != 0:
                alpha_d = np.min(x0[dx < 0] / -dx[dx < 0])
            if ds[ds < 0].size != 0:
                alpha_p = np.min(s0[ds < 0] / -ds[ds < 0])

            alpha = min(1, 0.95 * alpha_d, 0.95 * alpha_p)

            x0 = x0 + dx * alpha
            y0 = y0 + dy * alpha
            s0 = s0 + ds * alpha
    print(x0)
    print(C.T@x0)


if __name__ == "__main__":
    main()
