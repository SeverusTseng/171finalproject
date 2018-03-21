import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Edit Info
# Author: Bohan Zhou
# First Created: 2018/03/20
# Last Edited: 2018/03/21

def nanDelete(data, threshold):
    """

    :param
    data: each column is a feature; each row is a user
    threshold: threshold rate of nan in a feature
    :return:
    data: delete features with bigger nan rate
    """
    row = data.shape[0]
    column = data.shape[1]
    i = column - 1
    rate = np.zeros(column)
    while i >= 0:
        rate[i] = np.count_nonzero(np.isnan(data[:, i]))
        if rate[i] >= threshold:
            data = np.delete(data, i, 1)
        i -= 1
        print(i)
    # plt.plot(rate, 'o')
    # plt.show()
    pd.Series(rate).plot(kind='hist', bins=np.arange(0, 2000, 100))
    plt.show()
    return data
