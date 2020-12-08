from .bballs import *
from .mnist import *
from .mnist_nonuniform import *
from .mocap_many import *
from .mocap_single import *


def load_data(
    data_dir,
    task: str,
    max_n: int,
    max_t: int,
    dt: float = 0.1,
    subject_id: int = 0,
    plot: bool = False,
):
    if task == "mnist":
        dataset = load_mnist_data(data_dir, dt=dt, plot=plot)
    if task == "mnist_nonuniform":
        dataset = load_mnist_nonuniform_data(data_dir, dt=dt, plot=plot)
    elif task == "mocap_many":
        dataset = load_mocap_data_many_walks(data_dir, dt=dt, plot=plot)
    elif task == "mocap_single":
        dataset = load_mocap_data_single_walk(
            data_dir, subject_id=subject_id, dt=dt, plot=plot
        )
    elif task == "bballs":
        dataset = load_bball_data(
            data_dir=data_dir,
            max_n=max_n,
            max_t=max_t,
            dt=dt,
            plot=plot,
		)
	else:
		raise ValueError(f"Task {task} not defined.")
    [N, T, D] = dataset.train.x.shape
    return dataset, N, T, D


def plot_reconstructions(task, X, Xrec, tobs, show=False, fname="reconstruction.png"):
    if task == "mnist":
        dataset = plot_mnist_recs(X, Xrec, show=show, fname=fname)
    if task == "mnist_nonuniform":
        dataset = plot_mnist_nonuniform_recs(X, Xrec, tobs, show=show, fname=fname)
    elif "mocap" in task:
        dataset = plot_cmu_mocap_recs(X, Xrec, show=show, fname=fname)
    elif task == "bballs":
        dataset = plot_bball_recs(X, Xrec, show=show, fname=fname)
