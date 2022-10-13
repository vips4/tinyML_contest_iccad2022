import os
import logging
import torch
from argparse import Namespace, ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
from fairseq.optim.adafactor import Adafactor

from convert import convert
from datasets.iegm import IEGM_DataSET
from models.tinyml_univr import TinyModel
from utils.logger import setup_logging
from utils.metrics import compute_metrics
from utils.train_test_utils import save_best_model
from utils.onnx_convert import pytorch2onnx

torch.manual_seed(42)


def parse_argv() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--gpu_id",
        help="ID of the GPU. If CPU is the target, set -1",
        type=int,
        default=0,
    )

    parser.add_argument("--data_dir", type=str, default="./data/data_training/")
    parser.add_argument("--indexes_dir", type=str, default="./data/")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=400)
    parser.add_argument("--signal_length", type=int, default=1250)

    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="tinymodel")
    parser.add_argument(
        "--autoconvert",
        help="Convert to ONNX after training",
        action="store_true",
    )
    parser.add_argument(
        "--not_concatenate_ts",
        help="Avoid concatenation of TrainSet and TestSet as final TrainSet.",
        action="store_true",
    )

    return parser.parse_args()


def train_run(args: Namespace):
    best_f2: float = 0

    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    net: TinyModel = TinyModel()
    net = net.to(device)

    if not os.path.isdir(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    checkpoint_path: str = os.path.join(
        args.checkpoints_dir, args.experiment_name + ".pkl"
    )

    if args.pretrained_model is not None:
        if os.path.isfile(checkpoint_path):
            pretrained = torch.load(checkpoint_path)
            net.load_state_dict(pretrained.state_dict())

    optimizer = Adafactor(
        net.parameters(), scale_parameter=True, relative_step=True, warmup_init=True
    )

    # Start dataset loading
    epoch_num = args.epochs
    path_data = args.data_dir
    path_indices = args.indexes_dir
    workers = args.workers
    SIZE = args.signal_length
    BATCH_SIZE = args.train_batch_size
    BATCH_SIZE_TEST = args.test_batch_size

    trainset = IEGM_DataSET(
        root_dir=path_data,
        indice_dir=path_indices,
        mode="train",
        size=SIZE,
    )

    testset = IEGM_DataSET(
        root_dir=path_data,
        indice_dir=path_indices,
        mode="test",
        size=SIZE,
    )

    if not args.not_concatenate_ts:
        logging.info("Concatenating Training and Testing set")
        trainset = torch.utils.data.ConcatDataset([trainset, testset])

    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers
    )

    testloader = DataLoader(
        testset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=workers
    )

    criterion = torch.nn.CrossEntropyLoss()

    logging.info("Start training")
    for epoch in tqdm(range(epoch_num), desc="Epochs"):
        train_preds = []
        train_gts = []

        net.train()
        for _, data in enumerate(tqdm(trainloader, desc="Train")):
            inputs: torch.Tensor = data["IEGM_seg"]
            labels: torch.Tensor = data["label"]

            inputs = inputs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            prediction = net(inputs)

            loss: torch.Tensor = criterion(prediction, labels)

            loss.backward()
            optimizer.step()

            train_preds.append(prediction)
            train_gts.append(labels)

        train_pred_np = torch.cat(train_preds).detach().cpu().numpy()
        train_gts_np = torch.cat(train_gts).detach().cpu().numpy()
        train_acc, train_f2, train_conf_mat = compute_metrics(
            train_gts_np, train_pred_np
        )

        net.eval()
        with torch.no_grad():
            preds = []
            gts = []
            for data_test in tqdm(testloader, desc="Test"):
                IEGM_test = data_test["IEGM_seg"]
                labels = data_test["label"]

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

        if test_f2 > best_f2:
            best_f2 = test_f2
            stats = {
                "epoch": epoch,
                "test_f2": test_f2,
                "test_acc": test_acc,
                "test_conf_mat": test_conf_mat,
                "train_f2": train_f2,
                "train_acc": train_acc,
                "train_conf_mat": train_conf_mat,
            }
            save_best_model(stats, net, checkpoint_path)

    if args.autoconvert:
        assert os.path.isfile(
            checkpoint_path
        ), "No checkpoint was saved, is your model learning?"
        # onnx_name = checkpoint_path.replace(".pkl", ".onnx")
        # pytorch2onnx(checkpoint_path, onnx_name, SIZE, verbose=True)
        convert(checkpoint_path, SIZE)


def main():
    setup_logging("tinyml-challenge-train")
    args = parse_argv()
    train_run(args)


if __name__ == "__main__":
    main()
