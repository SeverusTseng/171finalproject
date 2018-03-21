import numpy as np
#import pandas as pd
#import csv

# Edit Info
# Author: Bohan Zhou
# First Created: 2018/03/20
# Last Edited: 2018/03/21

"""
functions:
data_in_from_npy: read data from npy
data_out_as_csv: save data as csv

"""


def data_in_from_npy(path):
    """
    Read data.npy from path, Save as (#id)*(id+#feature+loss) array without header
    Remain NAN feature

    :param
    path: path of input data
    :return:
    (#id)*(id+#feature+loss) nparray

    (TESTED)
    """

    datain = np.load(path)
    row = len(datain)
    tem = datain[0].decode('UTF-8').split(',')
    column = len(tem)

    dataout = np.empty([row - 1, column])

    for i in range(row - 1):
        tem = datain[i + 1].decode('UTF-8').split(',')
        for j in range(column):
            if tem[j] == 'NA':
                tem[j] = np.nan
            # dataout[i][j] = tem[j]
        dataout[i, :] = tem

    return dataout


# Testing Part
#
# INPATH = "/Users/bhzhouzhou/PycharmProjects/ECS171/finals/ecs171train.npy"
# train = data_in_from_npy(INPATH)

def data_out_as_csv(data, path, myheader):
    """

    :param
    data: data to write into csv
    path: path to save
    myheader: add a header above data in csv

    :return:
    a csv file with myheader

    (TESTED)
    """

    np.savetxt(path, data, delimiter=',', header=myheader, comments="")

# Testing Part
#
# OUTPATH = "/Users/bhzhouzhou/PycharmProjects/ECS171/finals/submission.csv"
# myheader = "id,loss"
#
# row = train.shape[0]
#
# outdata = np.empty((row, 2))
# outdata[:, 0] = train[:, 0]
# outdata[:, 1] = train[:, -1]
# data_out_as_csv(outdata, OUTPATH, myheader)
