import os
from pathlib import Path

import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("agg")
from .utils import MyDataset


def load(dir, file) -> np.ndarray:
    if (dir / (file + ".hkl")).exists():
        X = hkl.load(os.path.join(dir, file + ".hkl"))
    elif (dir / (file + ".npz")).exists():
        X = np.load(dir / (file + ".npz"))["arr_0"]
    else:
        raise ValueError(f"No hkl or npz file={dir / file}")
    return X


def trim(X, max_n: int, max_t: int):
    N = min(X.shape[0], max_n)
    T = min(X.shape[1], max_t)
    return X[:N, :T]


def load_bball_data(data_dir, max_n: int, max_t: int, dt=0.1, plot=True):
    data_dir = Path(data_dir)

    Xtr = trim(load(data_dir, "training"), max_n, max_t)

    Ytr = dt * np.arange(0, Xtr.shape[1], dtype=np.float32)
    Ytr = np.tile(Ytr, [Xtr.shape[0], 1])

    Xval = trim(load(data_dir, "val"), max_n, max_t)
    Yval = dt * np.arange(0, Xval.shape[1], dtype=np.float32)
    Yval = np.tile(Yval, [Xval.shape[0], 1])

    Xtest = trim(load(data_dir, "test"), max_n, max_t)
    Ytest = dt * np.arange(0, Xtest.shape[1], dtype=np.float32)
    Ytest = np.tile(Ytest, [Xtest.shape[0], 1])

    dataset = MyDataset(Xtr, Ytr, Xval, Yval, Xtest, Ytest)

    if plot:
        X, y = dataset.train.next_batch(5)
        tt = min(20, X.shape[1])
        plt.figure(2, (tt, 5))
        for j in range(5):
            for i in range(tt):
                plt.subplot(5, tt, j * tt + i + 1)
                plt.imshow(np.reshape(X[j, i, :], [32, 32]), cmap="gray")
                plt.xticks([])
                plt.yticks([])
        plt.savefig("plots/bballs/data.png")
        plt.close()
    return dataset


def plot_bball_recs(
    X, Xrec, idxs=[0, 1, 2, 3, 4], show=False, fname="reconstructions.png"
):
    if X.shape[0] < np.max(idxs):
        idxs = np.arange(0, X.shape[0])
    tt = min(20, X.shape[1])
    plt.figure(2, (tt, 3 * len(idxs)))
    for j, idx in enumerate(idxs):
        for i in range(tt):
            plt.subplot(2 * len(idxs), tt, j * tt * 2 + i + 1)
            plt.imshow(np.reshape(X[idx, i, :], [32, 32]), cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(2 * len(idxs), tt, j * tt * 2 + i + tt + 1)
            plt.imshow(np.reshape(Xrec[idx, i, :], [32, 32]), cmap="gray")
            plt.xticks([])
            plt.yticks([])
    plt.savefig(fname)
    if show is False:
        plt.close()
