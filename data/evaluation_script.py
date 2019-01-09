"""
For mathematic details, please check
https://waseda.app.box.com/v/SIGIR2018preprint
"""

from __future__ import print_function
from __future__ import division

import sys
import json
import numpy as np
from scipy import stats


C_NUGGET_L = 4  # CNUG0, CNUG, CNUG*, CNaN
H_NUGGET_L = 3  # HNUG, HNUG*, HNaN
QUALITY_L = 5   # 2, 1, 0, -1, -2


def normalize(p, q):
    p, q = np.asarray(p), np.asarray(q)
    assert (p >= 0).all()
    assert (q >= 0).all()

    p, q = p / p.sum(), q / q.sum()
    return p, q


def normalized_match_dist(p, q, L):
    """NMD"""
    p, q = normalize(p, q)
    cum_p, cum_q = np.cumsum(p), np.cumsum(q)
    return (np.abs(cum_p - cum_q)).sum() / (L - 1)


def rsnod(p, q):
    """ RSNOD: Root Symmetric Normalised Order-Aware Divergence
    """
    pass


def squared_error(p, q):
    """ SS
    """
    p, q = normalize(p, q)
    return ((p - q) ** 2).sum()


def root_normalized_sqaured_error(p, q):
    """ RNSS
    """
    return (squared_error(p, q) / 2) ** 0.5


def jensen_shannon_div(p, q, base=2):
    ''' JSD
    '''
    p, q = normalize(p, q)
    m = 1. / 2 * (p + q)
    return stats.entropy(p, m, base=base) / 2. + stats.entropy(q, m, base=base) / 2.


def main():
    if len(sys.argv) != 3:
        raise ValueError("Expect two arguments <ground_truth.json> <submission.json>")

    _, truth_path, submission_path = sys.args

    truth_data = json.load(open(truth_path, encoding="utf-8"))
    submission = json.load(open(submission_path, encoding="utf-8"))


if __name__ == "__main__":
    main()
