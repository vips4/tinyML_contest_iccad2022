import os
import torch


def pytorch2onnx(
    net_path: str, out_name: str, size: int = 1250, verbose: bool = False
) -> str:
    assert os.path.isfile(net_path), "Checkpoint path not found!"
    net = torch.load(net_path, map_location=torch.device("cpu"))

    dummy_input = torch.randn(1, 1, size, 1)

    head, _ = os.path.split(out_name)
    if not os.path.exists(head):
        os.makedirs(head)
    torch.onnx.export(net, dummy_input, out_name, verbose=verbose)
    return out_name
