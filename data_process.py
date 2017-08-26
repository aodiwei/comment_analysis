#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/7/19'
# 
"""
import numpy as np
import pandas as pd


def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        hot = dummies.values.tolist()
        df['label_o'] = hot
        # df = pd.concat([df, dummies], axis=1)
    return df


def load_data(filename='photo_data_clean.csv'):
    """
    load data
    :param filename:
    :return:
    """
    df = pd.read_csv(filename)

    df = one_hot(df, ['label'])

    df.to_csv(filename.replace('.csv', '_label_o.csv'), encoding='utf-8', index=False)

    return df['comment'].values, df['label_o'].values


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    load_data()
