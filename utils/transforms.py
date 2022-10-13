from __future__ import annotations
from typing import Dict
import torch


class ToTensor(object):
    def __call__(self, sample) -> Dict[str, torch.Tensor | int]:
        text = sample["IEGM_seg"]
        return {"IEGM_seg": torch.from_numpy(text), "label": sample["label"]}
