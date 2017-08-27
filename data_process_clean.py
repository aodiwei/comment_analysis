#!/usr/bin/env python
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/8/21'
# 
"""
import re

import pandas as pd
import jieba
import jieba.analyse

p1 = re.compile(r'【.*】')
# p2 = re.compile(r'\w*', re.L)
p3 = re.compile(r'[ -\[\]【】#@&\*%\.]')

with open('stopword.txt', 'rt', encoding='utf-8') as f:
    stopwords = set([w.strip() for w in f])

with open('filter_words.csv', 'rt', encoding='utf-8') as f:
    filter_words = set([w.strip() for w in f])

jieba.analyse.set_stop_words(r'stopword.txt')

filter_stopword = True


def process(row):
    comment = row['comment'].lower().strip().replace(' ', '').replace('　　', '').replace('.', '').replace('\n', '').replace('wi-fi', 'wifi')
    comment = p1.sub('', comment)
    # comment = p2.sub('', comment)
    # comment = p3.sub('', comment)
    words = jieba.cut(comment)
    if filter_stopword:
        words = set(words) - stopwords
    else:
        words = list(words)
    words = set(words) - filter_words
    row['comment'] = ' '.join(words)
    print(row['comment'])

    return row


def clean_data(filename):
    """
    清洗
    :param filename:
    :return:
    """
    df = pd.read_csv(filename, names=['comment', 'label'], encoding='utf-8')
    df.dropna(inplace=True)
    df = df.apply(process, axis=1)
    df.dropna(inplace=True)

    df.to_csv(filename.replace('.csv', '_clean.csv'), encoding='utf-8', index=False)




if __name__ == '__main__':
    clean_data(r'mulclass.csv')
