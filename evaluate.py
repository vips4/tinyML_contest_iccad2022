import logging
import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.iegm import IEGM_DataSET
from models.tinyml_univr import TinyModel
from utils.logger import setup_logging  # required import to let the network load
from utils.metrics import compute_metrics, stats_report

torch.manual_seed(42)


def parse_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu_id",
        help="ID of the GPU. If CPU is the target, set -1",
        type=int,
        default=0,
    )

    parser.add_argument("--data_dir", type=str, default="./data/data_training/")
    parser.add_argument("--indexes_dir", type=str, default="./data/")
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument("--test_batch_size", type=int, default=400)
    parser.add_argument("--signal_length", type=int, default=1250)

    parser.add_argument(
        "--pretrained_model", type=str, default="./checkpoints/tinymodel.pkl"
    )

    return parser.parse_args()


def evaluate(args):
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    pretrained_model = args.pretrained_model
    assert os.path.isfile(pretrained_model), (
        "No checkpoint found in " + pretrained_model
    )
    net: TinyModel = torch.load(pretrained_model)

    # Start dataset loading
    path_data = args.data_dir
    path_indices = args.indexes_dir
    workers = args.workers
    SIZE = args.signal_length
    BATCH_SIZE_TEST = args.test_batch_size

    testset = IEGM_DataSET(
        root_dir=path_data,
        indice_dir=path_indices,
        mode="test",
        size=SIZE,
    )

    testloader = DataLoader(
        testset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=workers
    )

    logging.info("Start evaluation")
    net.eval()
    with torch.no_grad():
        preds = []
        gts = []
        for data_test in tqdm(testloader, desc="Test set evaluation"):
            IEGM_test: torch.Tensor = data_test["IEGM_seg"]
            labels: torch.Tensor = data_test["label"]

            IEGM_test = IEGM_test.float().to(device)

            labels = labels.to(device)

            prediction = net(IEGM_test)

            preds.append(prediction)
            gts.append(labels)

    gts_np, pred_np = (
        torch.cat(gts).detach().cpu().numpy(),
        torch.cat(preds).detach().cpu().numpy(),
    )
    test_acc, test_f2, test_conf_mat = compute_metrics(gts_np, pred_np)

    logging.info("-" * 50)
    logging.info("Testing model: " + pretrained_model)
    logging.info("Test F2: " + str(test_f2))
    logging.info("Test Accuracy: " + str(test_acc))
    logging.info("\nConfusion Matrix:\n" + str(test_conf_mat))
    logging.info("-" * 50)

    out_str = stats_report(test_conf_mat)
    logging.info("\n" + out_str)
    logging.info("-" * 50)

    with open(pretrained_model.replace(".pkl", "_eval_results.txt"), "w") as fp:
        fp.write("Confusion Matrix:\n" + str(test_conf_mat) + "\n")
        fp.write(out_str)


if __name__ == "__main__":
    setup_logging("tinyml-challenge-eval")
    args = parse_argv()
    evaluate(args)
