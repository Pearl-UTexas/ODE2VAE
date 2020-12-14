from pathlib import Path

import fire
import pandas as pd


def main(log: str, out: str):
    log_path = Path(log)
    training_metrics = pd.DataFrame(
        columns=[
            "Loss",
            "p(x|z)",
            "p(z)",
            "q(z)",
            "KL[ode||enc]",
            "Valid cost",
            "Valid error",
        ]
    )
    i = 0
    for line in log_path.open("r").readlines():
        if ":train.py:282:INFO" in line or ":train.py:292:INFO" in line:
            tokens = [
                token
                for token in line.split(
                    " ",
                )
                if token != ""
            ]
            assert int(tokens[2]) == i, f"Index {i} doesn't match epoch {tokens[2]}"
            cost, condition, p, q, kl, valid_cost = [float(s) for s in tokens[3:9]]
            valid_error = tokens[9] if len(tokens) >= 10 else float("nan")
            training_metrics.loc[i] = (
                cost,
                condition,
                p,
                q,
                kl,
                valid_cost,
                valid_error,
            )
            i += 1
    training_metrics.to_pickle(out)


if __name__ == "__main__":
    fire.Fire(main)
