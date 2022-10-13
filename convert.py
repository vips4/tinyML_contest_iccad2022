from argparse import ArgumentParser, Namespace
import logging
import os
import time
from utils.logger import setup_logging
from utils.onnx_convert import pytorch2onnx
from models.tinyml_univr import TinyModel  # required import to let the network load


def parse_argv() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained_model", type=str, default="./checkpoints/tinymodel.pkl"
    )
    parser.add_argument("--signal_length", type=int, default=1250)
    return parser.parse_args()


def convert(checkpoint_path: str, size: int = 1250):
    assert os.path.isfile(checkpoint_path), "No checkpoint found in " + checkpoint_path
    # onnx_name = checkpoint_path.replace(".pkl", ".onnx")

    head, _ = os.path.split(checkpoint_path)
    onnx_name = os.path.join(head, "network.onnx")

    logging.info(f"Starting conversion of file {checkpoint_path} to file {onnx_name}")
    start = time.time()
    pytorch2onnx(checkpoint_path, onnx_name, size, verbose=True)
    logging.info(f"Done conversion in {time.time() - start}")


def main():
    setup_logging("tinyml-challenge-onnx")
    args = parse_argv()
    convert(args.pretrained_model, args.signal_length)


if __name__ == "__main__":
    main()
