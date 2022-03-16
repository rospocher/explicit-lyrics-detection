import numpy as np
import researchpy as rp
import os
import argparse
import pandas as pd


def readEvalData(evaldata):

    data=np.loadtxt(evaldata)
    return pd.Series(data[0]), pd.Series(data[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute McNemar statistical test')
    parser.add_argument('eval1', help='eval file system 1')
    parser.add_argument('eval2', help='eval file system 2')


    args = parser.parse_args()

    eval1 = args.eval1
    print("Processing file {} containing eval data for system 1.".format(eval1))
    assert os.path.exists(eval1), "file containing eval data for system 1 NON FOUND!!!"

    eval2 = args.eval2
    print("Processing file {} containing eval data for system 1.".format(eval2))
    assert os.path.exists(eval2), "file containing eval data for system 1 NON FOUND!!!"

    gold1, system1 = readEvalData(eval1)
    gold2, system2 = readEvalData(eval2)

    table, res = rp.crosstab(system1, system2, test='mcnemar')
    print(table)
    print(res)