import param
import numpy as np
from scipy import stats

doclen = param.doclen
embsize = param.embsize
max_sent = param.max_sent
NDclasses = param.NDclasses


def normalize(p, q):
    p, q = np.asarray(p), np.asarray(q)
    assert (p >= 0).all(), p
    assert (q >= 0).all()

    p, q = p / p.sum(), q / q.sum()
    return p, q


def RNSS(p, q):
    def SS(p, q):
        p, q = normalize(p, q)
        return ((p - q) ** 2).sum()

    return (SS(p, q) / 2) ** 0.5


def JSD(p, q, base=2):
    p, q = normalize(p, q)
    m = 1. / 2 * (p + q)
    return stats.entropy(p, m, base=base) / 2. + stats.entropy(q, m, base=base) / 2.


def NMD(pred, truth):
    """NMD: Normalized Match Distance"""
    pred, truth = normalize(pred, truth)
    cum_p, cum_q = np.cumsum(pred), np.cumsum(truth)
    return (np.abs(cum_p - cum_q)).sum() / (len(pred) - 1.)


def DW(pred, truth, i):
    return np.sum([np.abs(i - j) * ((pred[j] - truth[j]) ** 2) for j in range(len(pred))])


def OAD(pred, truth):
    return np.mean([DW(pred, truth, i) for i in range(len(pred)) if pred[i] > 0])


def RSNOD(pred, truth):
    """ RSNOD: Root Symmetric Normalised Order-Aware Divergence"""
    pred, truth = normalize(pred, truth)
    sod = (OAD(pred, truth) + OAD(truth, pred)) / 2.
    return np.sqrt((sod / (len(pred) - 1)))


def nugget_evaluation_CRF(pred, y):
    correct = 0
    total = 0
    confusion_matrix = [[0] * NDclasses for i in range(NDclasses)]
    for pred_sents, y_sents in zip(pred, y):  # (7, 7)
        for pred_sentidx, y_sentidxs in zip(pred_sents, y_sents):  # (7)
            if isinstance(y_sentidxs, np.ndarray):
                for y_sentidx in y_sentidxs:
                    confusion_matrix[pred_sentidx][y_sentidx] += 1
                correct = correct + 1 if pred_sentidx in y_sentidxs else correct
            else:
                if y_sentidxs == 7:
                    break
                else:
                    y_sentidx = y_sentidxs
                    confusion_matrix[pred_sentidx][y_sentidx] += 1
                    correct = correct + 1 if pred_sentidx == y_sentidx else correct
            total += 1
    acc = correct / total
    return confusion_matrix, acc


def nugget_evaluation(pred, y, turns, masks):
    # pred = y = (?, 7, 7)
    total_RNSS = 0
    total_JSD = 0
    total_sent = 0
    # pred = np.multiply(pred, masks)
    for pred_sents, y_sents in zip(pred, y):  # (7, 7)
        for pred_sent, y_sent in zip(pred_sents, y_sents):  # (7)
            if np.all(y_sent == 0):
                break
            else:
                total_sent += 1
                total_RNSS += RNSS(pred_sent, y_sent)
                total_JSD += JSD(pred_sent, y_sent)

    avg_RNSS = total_RNSS / total_sent
    avg_JSD = total_JSD / total_sent
    return avg_RNSS, avg_JSD


def quality_evaluation(pred, y):
    total_NMD = 0
    total_RSNOD = 0
    total_dialog = len(y)

    for p, t in zip(pred, y):
        total_NMD += NMD(p, t)
        total_RSNOD += RSNOD(p, t)

    avg_NMD = total_NMD / total_dialog
    avg_RSNOD = total_RSNOD / total_dialog
    return avg_NMD, avg_RSNOD
