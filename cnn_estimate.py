#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/7/24'
# 
"""
import os
import csv

import pandas as pd
import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib import learn

import define
from tools.const import Const
from tools.utility import Utility
from works.data_manage.save_data_manage import SaveDataManage
from works.work_botnet_supervise.domain_cnn import data_process


# log = Utility.get_logger('ml_cnn_domain')


class CnnEstimate:
    def __init__(self, model_name):
        Utility.init_alphabet_dict()
        self.model_name = model_name
        self.load_config()
        self.reload_model()
        self.save_data_manage = SaveDataManage()

    def load_config(self):
        """
        加载配置
        :return: 
        """
        config = Utility.conf_get('botnet')
        self.model_folder = os.path.join(config.get('root_path'), config.get('stable_model'), self.model_name)
        self.checkpoint_dir = os.path.join(self.model_folder, 'checkpoints')
        self.batch_size = config.get('es_batch_size')
        self.data_folder = os.path.join(config.get('root_path'), config.get('data_folder'))

        vocab_file = config.get('vocab')
        vocab_path = os.path.join(define.root, 'src', 'static_data', vocab_file)
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    def reload_model(self):
        """
        加载训练好的模型
        :return: 
        """
        checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_dir)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file), clear_devices=True)
                saver.restore(self.sess, checkpoint_file)

                # Get the placeholders from the graph by name
                self.input_x = graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]

    def estimate_domain(self, *domains):
        """
        判断域名
        :param domains: 
        :return: 
        """
        # x_raw = [' '.join(list(x)) for x in domains]
        # x_test = np.array(list(self.vocab_processor.transform(x_raw)))
        x_test = []
        domains_ = []
        for dom in domains:
            _, x, _ = Utility.pre_encode_domain(dom)
            if isinstance(x, list):
                x_test.append(x)
                domains_.append(dom)
        # x_test = [Utility.pre_encode_domain(x)[1] for x in domains]
        # x_test = [x for x in x_test if isinstance(x, list)]
        batches = data_process.batch_iter(x_test, self.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = self.sess.run(self.predictions, {self.input_x: x_test_batch, self.dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        predictions_human_readable = np.column_stack((np.array(domains_), all_predictions))
        df = pd.DataFrame(predictions_human_readable, columns=['domain', 'label'])
        try:
            self.save_data_manage.save_domain(df)
        except Exception as e:
            print(e)
            self.save_to_csv(df)

        return predictions_human_readable

    def save_to_csv(self, df):
        """
        csv
        :param df: 
        :return: 
        """
        out_path = os.path.join(self.data_folder, 'es_output')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        timestamp = time.strftime(Const.LOCAL_FORMAT_DATE, time.localtime(time.time()))
        out_path = os.path.join(out_path, '{}_prediction_{}.csv'.format(self.model_name, timestamp))
        if os.path.exists(out_path):
            df_pre = pd.read_csv(out_path)
            df = pd.concat([df, df_pre], ignore_index=True)
            df.drop_duplicates(inplace=True)
        df.to_csv(out_path, index=False, encoding='utf-8')

    def run(self):
        """
        
        :return: 
        """
        x_raw = ["nasscomminc.tk", "servmill.com", 'sina.com.cn', 'ifeng.com', 'baidu.com', 'avsxrcoq2q5fgrw2.dgjpgy.top']
        self.estimate_domain(*x_raw)


if __name__ == '__main__':
    es = CnnEstimate(model_name='cnn')
    df = pd.read_csv(r'E:\data\dataset\online_domain.csv')
    domains = df['domain'].values.tolist()
    es.estimate_domain(*domains)
    # es = CnnEstimate(model_name='cnn_small_data')
    # df = pd.read_csv(r'E:\data\dataset\online_domain.csv')
    # domains = df['domain'].values.tolist()
    # es.estimate_domain(*domains)
    # es.run()
