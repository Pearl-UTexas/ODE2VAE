"""
This script comes from the RTRBM code by Ilya Sutskever from
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
"""

# usage
# python3 gen_bouncing_ball_video.py num_frames num_seq


import logging
from pathlib import Path
from typing import Optional

import fire
import hickle as hkl  # type: ignore
import matplotlib  # type: ignore
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
    sqrt,
    zeros,
)
from numpy.random import rand, randn


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


def norm(x: np.ndarray) -> np.ndarray:
    return sqrt((x ** 2).sum())


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + exp(-x))


def bounce_n(
    time_horizon: int = 128,
    n_balls: int = 2,
    radius: Optional[np.ndarray] = None,
    mass: Optional[np.ndarray] = None,
    init_position: Optional[np.ndarray] = None,
    size: float = 10,
    noise: float = 0.0,
    step_size: float = 0.5,
) -> np.ndarray:
    if radius is None:
        radius = array([1.2] * n_balls)
    if mass is None:
        mass = array([1] * n_balls)
    X = zeros((time_horizon, n_balls, 2), dtype="float")
    v = randn(n_balls, 2)
    v = v / norm(v) * 0.5
    good_config = False
    position = init_position if init_position is not None else 2 + rand(n_balls, 2) * 8
    while not good_config:
        position = 2 + rand(n_balls, 2) * 8
        good_config = True
        for i in range(n_balls):
            for z in range(2):
                if position[i][z] - radius[i] < 0:
                    good_config = False
                if position[i][z] + radius[i] > size:
                    good_config = False

        for i in range(n_balls):
            for j in range(i):
                if norm(position[i] - position[j]) < radius[i] + radius[j]:
                    good_config = False

    for time_horizon in range(time_horizon):
        X[time_horizon] = position

        for _ in range(int(1 / step_size)):

            position += step_size * v + np.random.normal(
                loc=0.0, scale=noise, size=position.shape
            )

            for i in range(n_balls):
                for z in range(2):
                    if position[i][z] - radius[i] < 0:
                        v[i][z] = abs(v[i][z])  # want positive
                    if position[i][z] + radius[i] > size:
                        v[i][z] = -abs(v[i][z])  # want negative

            for i in range(n_balls):
                for j in range(i):
                    if norm(position[i] - position[j]) < radius[i] + radius[j]:
                        w = position[i] - position[j]
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

    for time_horizon in range(time_horizon):
        for i in range(n_balls):
            A[time_horizon] += exp(
                -(
                    (
                        (
                            (I - X[time_horizon, i, 0]) ** 2
                            + (J - X[time_horizon, i, 1]) ** 2
                        )
                        / (radius[i] ** 2)
                    )
                    ** 4
                )
            )

        A[time_horizon][A[time_horizon] > 1] = 1
    return A


def bounce_vec(
    resolution, n_balls=2, time_horizon=128, radius=None, mass=None, noise: float = 0.0
):
    if radius is None:
        radius = array([1.2] * n_balls)
    x = bounce_n(time_horizon, n_balls, radius, mass, noise=noise)
    V = matricize(x, resolution, radius)
    return V.reshape(time_horizon, resolution ** 2), x


# make sure you have this folder
logdir = "./sample"


def show_sample(V):
    time_horizon = len(V)
    res = int(sqrt(shape(V)[1]))
    for time_horizon in range(time_horizon):
        plt.imshow(V[time_horizon].reshape(res, res), cmap=matplotlib.cm.Greys_r)
        # Save it
        fname = logdir + "/" + str(time_horizon) + ".png"
        plt.savefig(fname)


def main(
    time_horizon: int = 20,
    n_videos: int = 10000,
    n_balls: int = 3,
    resolution: int = 32,
    noise: float = 0.0,
    outdir: Path = Path("balls"),
):
    matplotlib.use("Agg")

    logging.info(f"time_horizon={time_horizon:d}, n_videos={n_videos:d}")

    frames = zeros((n_videos, time_horizon, resolution * resolution), dtype=float32)
    positions = zeros((n_videos, time_horizon, 3, 2), dtype=float32)  # ball locations
    logging.info(f"Frames shape={frames.shape}")
    for i in range(n_videos):
        logging.info(f"Making training video {i} of {n_videos}")
        frames[i], positions[i] = bounce_vec(
            resolution=resolution,
            n_balls=n_balls,
            time_horizon=time_horizon,
            noise=noise,
        )
    hkl.dump(frames, outdir / "training.hkl", mode="w", compression="gzip")
    hkl.dump(positions, outdir / "training_locs.hkl", mode="w", compression="gzip")

    Nv = int(n_videos / 20)
    frames = zeros((Nv, time_horizon, resolution * resolution), dtype=float32)
    positions = zeros((Nv, time_horizon, n_balls, 2), dtype=float32)  # ball locations
    for i in range(Nv):
        logging.info(f"Making validation video {i} of {Nv}")
        frames[i], positions[i] = bounce_vec(
            resolution=resolution,
            n_balls=n_balls,
            time_horizon=time_horizon,
            noise=noise,
        )
    hkl.dump(frames, outdir / "val.hkl", mode="w", compression="gzip")
    hkl.dump(positions, outdir / "val_locs.hkl", mode="w", compression="gzip")

    Nt = int(n_videos / 20)
    frames = zeros((Nt, time_horizon, resolution * resolution), dtype=float32)
    positions = zeros((Nt, time_horizon, n_balls, 2), dtype=float32)  # ball locations
    for i in range(Nt):
        logging.info(f"Making testing video {i} of {Nt}")
        frames[i], positions[i] = bounce_vec(
            resolution=resolution, n_balls=n_balls, time_horizon=time_horizon
        )
    hkl.dump(frames, outdir / "test.hkl", mode="w", compression="gzip")
    hkl.dump(positions, outdir / "test_locs.hkl", mode="w", compression="gzip")

    # show one video
    # show_sample(dat[1])
    # ffmpeg -start_number 0 -i %d.png -c:v libx264 -pix_fmt yuv420p -r 30 sample.mp4


if __name__ == "__main__":
    fire.Fire(main)
