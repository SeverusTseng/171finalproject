import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Edit Info
# Author: Bohan Zhou
# First Created: 2018/03/20
# Last Edited: 2018/03/21

def nanDelete(data1, data2, threshold):
    """
    delete nan data for both data1 and data2 based on data1 over threshold, replace nan by 0 for the others.
    :param
    data1: each column is a feature; each row is a user;
    data2:
    threshold: threshold (rate) of nan in a feature
               currently choose number rather than rate
    :return:
    data: delete features with bigger nan rate
    """
    row = data1.shape[0]
    column = data1.shape[1]
    i = column - 1
    rate = np.zeros(column)
    index = []
    while i >= 0:
        rate[i] = np.count_nonzero(np.isnan(data1[:, i]))
        if rate[i] >= threshold:
            data1 = np.delete(data1, i, 1)
            data2 = np.delete(data2, i, 1)
            index.extend([i])

        i -= 1

    """
    Ploting Part
    """
    # fig, ax = plt.subplots()
    # bins_value = [0, 2500, 50000]
    # hist, bin_edges = np.histogram(rate, bins_value)
    #
    # ax.bar(range(len(hist)), hist, width=1, align='center', tick_label=
    # ['{} - {}'.format(bins_value[i], bins_value[i + 1]) for i, j in enumerate(hist)], color=['yellow', 'blue'])
    # ax.set_ylabel('Number of Features')
    # ax.set_xlabel('Number of Missing Data')
    # for a, b in zip(range(len(hist)), hist):
    #     plt.text(a, b+1, str(b-2))
    # plt.show()
    data1 = np.nan_to_num(data1)
    data2 = np.nan_to_num(data2)

    return data1, data2, index


