import glob
import os
from typing import Dict
import torch
import logging


def save_best_model(stats: Dict[str, float], model: torch.nn.Module, destination: str):
    epoch = stats["epoch"]

    test_f2 = stats["test_f2"]
    test_acc = stats["test_acc"]
    test_conf_mat = stats["test_conf_mat"]

    train_f2 = stats["train_f2"]
    train_acc = stats["train_acc"]
    train_conf_mat = stats["train_conf_mat"]

    logging.info(f"F2 {test_f2:.5f}")
    out_f = destination.replace(".pkl", "_train_results.txt")

    out_txts = [
        f"-----------------------------------\n",
        f"Epoch: {epoch}\n",
        f"-----------------------------------\n",
        f"Test F2: {test_f2}\n",
        f"Test Accuracy: {test_acc}\n",
        f"Test Confusion Matrix:\n{test_conf_mat}\n",
        f"-----------------------------------\n",
        f"Train F2: {train_f2}\n",
        f"Train Accuracy: {train_acc}\n",
        f"Train Confusion Matrix:\n{train_conf_mat}\n",
        f"-----------------------------------\n",
    ]

    with open(out_f, "w") as fp:
        [fp.write(s) for s in out_txts]
        fp.write(str(model))
    [logging.info(s) for s in out_txts]

    torch.save(model, destination)
