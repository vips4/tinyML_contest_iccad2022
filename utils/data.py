import csv
from typing import Dict, List
import numpy as np


def loadCSV(csvf: str) -> Dict[str, List[int]]:
    dictLabels: Dict[str, List[int]] = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]

            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels


def txt_to_numpy(filename: str, row: int) -> np.ndarray:
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float)
    row_count = 0
    for line in lines:
        line = line.strip().split(" ")
        datamat[row_count] = line[0]
        row_count += 1

    return datamat
