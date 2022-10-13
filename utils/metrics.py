from __future__ import annotations
from typing import Tuple
import torch
import numpy as np
from sklearn.metrics import accuracy_score, fbeta_score, confusion_matrix


def compute_metrics(
    gts: np.ndarray | torch.Tensor, preds: np.ndarray | torch.Tensor
) -> Tuple[float, float, np.ndarray]:
    accuracy_binary = accuracy_score(gts, preds.argmax(1))
    f2_binary = fbeta_score(gts, preds.argmax(1), beta=2, average="binary")
    conf_mat = confusion_matrix(gts, preds.argmax(1))

    return accuracy_binary, f2_binary, conf_mat


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return (TP, FP, TN, FN)


def ACC(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    total = sum(mylist)
    acc = (tp + tn) / total
    return acc


def PPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then ppv should be 1
    if tp + fn == 0:
        ppv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tp + fp == 0 and tp + fn != 0:
        ppv = 0
    else:
        ppv = tp / (tp + fp)
    return ppv


def NPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then npv should be 1
    if tn + fp == 0:
        npv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tn + fn == 0 and tn + fp != 0:
        npv = 0
    else:
        npv = tn / (tn + fn)
    return npv


def Sensitivity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then sen should be 1
    if tp + fn == 0:
        sensitivity = 1
    else:
        sensitivity = tp / (tp + fn)
    return sensitivity


def Specificity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then spe should be 1
    if tn + fp == 0:
        specificity = 1
    else:
        specificity = tn / (tn + fp)
    return specificity


def BAC(mylist):
    sensitivity = Sensitivity(mylist)
    specificity = Specificity(mylist)
    b_acc = (sensitivity + specificity) / 2
    return b_acc


def F1(mylist):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def FB(mylist, beta=2):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (1 + beta**2) * (precision * recall) / ((beta**2) * precision + recall)
    return f1


def stats_report(conf_mat: np.ndarray) -> str:
    tn, fp, fn, tp = conf_mat.ravel()

    mylist = [tp, fn, fp, tn]
    f1 = round(F1(mylist), 5)
    fb = round(FB(mylist), 5)
    se = round(Sensitivity(mylist), 5)
    sp = round(Specificity(mylist), 5)
    bac = round(BAC(mylist), 5)
    acc = round(ACC(mylist), 5)
    ppv = round(PPV(mylist), 5)
    npv = round(NPV(mylist), 5)

    output = (
        "tp, fn, fp, tn : "
        + str(mylist)
        + "\n"
        + "F-1 = "
        + str(f1)
        + "\n"
        + "F-B = "
        + str(fb)
        + "\n"
        + "SEN = "
        + str(se)
        + "\n"
        + "SPE = "
        + str(sp)
        + "\n"
        + "BAC = "
        + str(bac)
        + "\n"
        + "ACC = "
        + str(acc)
        + "\n"
        + "PPV = "
        + str(ppv)
        + "\n"
        + "NPV = "
        + str(npv)
        + "\n"
    )

    return output
