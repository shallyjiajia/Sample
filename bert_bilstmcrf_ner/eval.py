#! -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from metrics import *
from train import *
from data_utils import load_data


def predict_label(data):
    y_pred = []
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        pred = NER.recognize(text)

        label = ['O' for _ in range(len(text))]
        b = 0
        for item in pred:
            word, typ = item[0], item[1]
            start = text.find(word, b)
            end = start + len(word)
            label[start] = 'B-' + typ
            for i in range(start + 1, end):
                label[i] = 'I-' + typ
            b += len(word)

        y_pred.append(label)

    return y_pred


def evaluate():
    data_path = 'data/dev.conll'
    test_data, y_true = load_data(data_path, 70)
    y_pred = predict_label(test_data)

    f1 = f1_score(y_true, y_pred, suffix=False)
    p = precision_score(y_true, y_pred, suffix=False)
    r = recall_score(y_true, y_pred, suffix=False)
    acc = accuracy_score(y_true, y_pred)

    print(
        "f1_score: {:.4f}, precision_score: {:.4f}, recall_score: {:.4f}, accuracy_score: {:.4f}".format(f1, p, r, acc))
    print(classification_report(y_true, y_pred, digits=4, suffix=False))


