import pickle
from pathlib import Path
from typing import List

import fire
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main(datadir: str, names: str, outdir: str):
    datapath = Path(datadir)
    names = names.split(" ")
    training_metrics: pd.DataFrame = pickle.load(open(datapath / names[0], "rb"))
    outpath = Path(outdir)
    for value in training_metrics.columns:
        for name in names:
            if np.isreal(training_metrics[value]).any():
                training_metrics = pickle.load(open(datapath / name, "rb"))
                noise = float(".".join(name.split("_")[-1].split(".")[0:2]))
                human_name = " ".join(".".join(name.split(".")[:-1]).split("_"))
                if "pegged" in human_name:
                    human_name = f"distributional {noise}"
                training_metrics[value].plot(label=human_name)

        if value == "cost":
            value = "loss"
        if value == "p(z)":
            value = "log p(z)"
        if value == "q(z)":
            value = "log q(z)"
        if value == "p(x|z)":
            value = "log p(x|z)"
        plt.title(f"Training {value}")
        plt.xlabel("Epoch")
        plt.ylabel(value)
        plt.legend()
        plt.savefig(outpath / f"training_{value}.png")
        plt.close()


if __name__ == "__main__":
    fire.Fire(main)
