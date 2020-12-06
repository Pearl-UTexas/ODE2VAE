"""
This script comes from the RTRBM code by Ilya Sutskever from
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
"""

# usage
# python3 gen_bouncing_ball_video.py num_frames num_seq


import sys
from typing import Optional

import hickle as hkl  # type: ignore
import matplotlib  # type: ignore

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from numpy import (
    arange,
    array,
    dot,
    exp,
    float32,
    linalg,
    meshgrid,
    ndarray,
    rand,
    randn,
    sqrt,
    zeros,
)


def shape(A):
    if isinstance(A, ndarray):
        return np.shape(A)
    else:
        return A.shape()


def size(A):
    if isinstance(A, ndarray):
        return np.size(A)
    else:
        return A.size()


det = linalg.det


def new_speeds(m1, m2, v1, v2):
    new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2


def norm(x):
    return sqrt((x ** 2).sum())


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


def bounce_n(
    time_horizon: int = 128,
    n_balls: int = 2,
    radius: Optional[np.ndarray] = None,
    mass: Optional[np.ndarray] = None,
    init_position: Optional[np.ndarray] = None,
    size: float = 10,
):
    if radius is None:
        radius = array([1.2] * n_balls)
    if mass is None:
        mass = array([1] * n_balls)
    X = zeros((time_horizon, n_balls, 2), dtype="float")
    v = randn(n_balls, 2)
    v = v / norm(v) * 0.5
    good_config = False
    x = init_position if init_position is not None else 2 + np.rand(n_balls, 2) * 8
    while not good_config:
        x = 2 + rand(n_balls, 2) * 8
        good_config = True
        for i in range(n_balls):
            for z in range(2):
                if x[i][z] - radius[i] < 0:
                    good_config = False
                if x[i][z] + radius[i] > size:
                    good_config = False

        for i in range(n_balls):
            for j in range(i):
                if norm(x[i] - x[j]) < radius[i] + radius[j]:
                    good_config = False

    eps = 0.5
    for t in range(time_horizon):
        for i in range(n_balls):
            X[t, i] = x[i]

        for mu in range(int(1 / eps)):

            for i in range(n_balls):
                x[i] += eps * v[i]

            for i in range(n_balls):
                for z in range(2):
                    if x[i][z] - radius[i] < 0:
                        v[i][z] = abs(v[i][z])  # want positive
                    if x[i][z] + radius[i] > size:
                        v[i][z] = -abs(v[i][z])  # want negative

            for i in range(n_balls):
                for j in range(i):
                    if norm(x[i] - x[j]) < radius[i] + radius[j]:
                        w = x[i] - x[j]
                        w = w / norm(w)

                        v_i = dot(w.transpose(), v[i])
                        v_j = dot(w.transpose(), v[j])

                        new_v_i, new_v_j = new_speeds(mass[i], mass[j], v_i, v_j)

                        v[i] += w * (new_v_i - v_i)
                        v[j] += w * (new_v_j - v_j)

    return X


def ar(x, y, z):
    return z / 2 + arange(x, y, z, dtype="float")


def matricize(X, resolution, radius, size=10):
    """I think this is converting ball positions to a video, but I'm not sure."""
    time_horizon, n_balls = shape(X)[0:2]

    A = zeros((time_horizon, resolution, resolution), dtype="float")

    [I, J] = meshgrid(
        ar(0, 1, 1.0 / resolution) * size, ar(0, 1, 1.0 / resolution) * size
    )

    for t in range(time_horizon):
        for i in range(n_balls):
            A[t] += exp(
                -(
                    (((I - X[t, i, 0]) ** 2 + (J - X[t, i, 1]) ** 2) / (radius[i] ** 2))
                    ** 4
                )
            )

        A[t][A[t] > 1] = 1
    return A


def bounce_vec(resolution, n_balls=2, time_horizon=128, radius=None, mass=None):
    if radius is None:
        radius = array([1.2] * n_balls)
    x = bounce_n(time_horizon, n_balls, radius, mass)
    V = matricize(x, resolution, radius)
    return V.reshape(time_horizon, resolution ** 2), x


# make sure you have this folder
logdir = "./sample"


def show_sample(V):
    T = len(V)
    res = int(sqrt(shape(V)[1]))
    for t in range(T):
        plt.imshow(V[t].reshape(res, res), cmap=matplotlib.cm.Greys_r)
        # Save it
        fname = logdir + "/" + str(t) + ".png"
        plt.savefig(fname)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        T = 50
        N = 400
    elif len(sys.argv) == 3:
        T = int(sys.argv[1])
        N = int(sys.argv[2])
    else:
        raise ValueError("number of input parameters is wrong! exiting...")
    print("T={:d}, N={:d}".format(T, N))
    res = 32

    dat = zeros((N, T, res * res), dtype=float32)
    x = zeros((N, T, 3, 2), dtype=float32)  # ball locations
    print(dat.shape)
    for i in range(N):
        print(i)
        dat[i, :, :], x[i, :, :, :] = bounce_vec(
            resolution=res, n_balls=3, time_horizon=T
        )
    hkl.dump(dat, "training.hkl", mode="w", compression="gzip")
    hkl.dump(x, "training_locs.hkl", mode="w", compression="gzip")

    Nv = int(N / 20)
    dat = zeros((Nv, T, res * res), dtype=float32)
    x = zeros((Nv, T, 3, 2), dtype=float32)  # ball locations
    for i in range(Nv):
        print(i)
        dat[i, :, :], x[i, :, :, :] = bounce_vec(
            resolution=res, n_balls=3, time_horizon=T
        )
    hkl.dump(dat, "val.hkl", mode="w", compression="gzip")
    hkl.dump(x, "val_locs.hkl", mode="w", compression="gzip")

    Nt = int(N / 20)
    dat = zeros((Nt, T, res * res), dtype=float32)
    x = zeros((Nt, T, 3, 2), dtype=float32)  # ball locations
    for i in range(Nt):
        print(i)
        dat[i, :, :], x[i, :, :, :] = bounce_vec(
            resolution=res, n_balls=3, time_horizon=T
        )
    hkl.dump(dat, "test.hkl", mode="w", compression="gzip")
    hkl.dump(x, "test_locs.hkl", mode="w", compression="gzip")

    # show one video
    # show_sample(dat[1])
    # ffmpeg -start_number 0 -i %d.png -c:v libx264 -pix_fmt yuv420p -r 30 sample.mp4
