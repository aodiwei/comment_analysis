#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/7/19'
# 
"""
import os
import time

import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

from cnn_model import TextCNN
import data_process


# tf.logging.set_verbosity(tf.logging.INFO)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CnnTrainer:
    """
    self.cnn 训练器
    """

    def __init__(self, model_code=0):
        self.load_config()
        self.model_code = model_code


    def load_config(self):
        """
        配置
        :return: 
        """
        self.train_args = {'lr_decay_steps': 1000,
                           'filter_sizes': [2, 3, 4, 5],
                           'ps_hosts': ['192.168.199.244:2222'],
                           'model_dir': 'model',
                           'is_sync': False,
                           'evaluate_every': 10,
                           'sequence_length': 60,
                           'log_device_placement': False,
                           'num_filters': 128, 'dropout_keep_prob': 0.5,
                           'l2_reg_lambda': 0.5,
                           'embedding_dim': 256,
                           'num_classes': 2,
                           'batch_size': 64,
                           'dev_batch_size': 1000,
                           'checkpoint_every': 100,
                           'num_epochs': 200,
                           'worker_hosts': ['192.168.199.244:2223', '192.168.199.193:2222', '192.168.199.193:2223'],
                           'allow_soft_placement': True,
                           'dev_sample_percentage': 0.1
                           }
        timestamp = time.strftime("%m%d_%H_%M_%S", time.localtime(time.time()))
        self.out_dir = os.path.join('', timestamp)
        print("Writing to {}\n".format(self.out_dir))
        self.checkpoint_dir = os.path.join(self.out_dir, "checkpoints")

    def init_model_class(self, model_code=0):
        """
        根据model code 确定使用的模型类
        :param model_code: 
        :return: 
        """

        self.cnn = TextCNN(
            sequence_length=self.vocab_processor.max_document_length,
            num_classes=self.train_args['num_classes'],
            vocab_size=len(self.vocab_processor.vocabulary_),
            embedding_size=self.train_args['embedding_dim'],
            filter_sizes=self.train_args['filter_sizes'],
            num_filters=self.train_args['num_filters'],
            l2_reg_lambda=self.train_args['l2_reg_lambda'])

    def load_data(self):
        """
        data
        :return:
        """
        x, y = data_process.load_data()
        x = self.train_comm_vocab(x)

        data_train, data_valid = self.split_data(x, y)
        self.train_args['num_classes'] = len(y[0])

        return data_train, data_valid

    def train_comm_vocab(self, x_text):
        """
        训练通用词典，忽略词频
        :return: 
        """
        max_document_length = max([len(x.split(" ")) for x in x_text])
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(self.vocab_processor.fit_transform(x_text)))
        self.vocab_processor.save('vocab')

        return x

    def split_data(self, x, y):
        """
        分割数据
        :param data:
        :return:
        """
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        # Split train/test set
        dev_sample_index = -1 * int(self.train_args['dev_sample_percentage'] * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

        return list(zip(x_train, y_train)), list(zip(x_dev, y_dev))

    def train_step(self, x_batch, y_batch, summary_writer, summary_op):
        """
        A single training step
        """
        feed_dict = {
            self.cnn.input_x: x_batch,
            self.cnn.input_y: y_batch,
            self.cnn.dropout_keep_prob: self.train_args['dropout_keep_prob']
        }
        _, step = self.sess.run([self.train_op, self.global_step], feed_dict)
        if step % self.train_args['evaluate_every'] == 0 and step != 0:
            summaries, loss, accuracy = self.sess.run([summary_op, self.cnn.loss, self.cnn.accuracy],
                                                      feed_dict=feed_dict, )
            print("eval: step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
            summary_writer.add_summary(summaries, step)
        elif step % 10 == 0:
            loss, accuracy = self.sess.run([self.cnn.loss, self.cnn.accuracy], feed_dict)
            print("train: step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
        else:
            pass

        return step

    def dev_step(self, x_batch, y_batch, step, summary_writer, summary_op, op_type):
        """
        Evaluates model on a dev set, test or valid
        """
        feed_dict = {
            self.cnn.input_x: x_batch,
            self.cnn.input_y: y_batch,
            self.cnn.dropout_keep_prob: 1.0
        }
        summaries, loss, accuracy = self.sess.run([summary_op, self.cnn.loss, self.cnn.accuracy], feed_dict)
        print("{}: step {}, loss {:g}, acc {:g}".format(op_type, step, loss, accuracy))

        summary_writer.add_summary(summaries, step)

    def save_summary(self, grads_and_vars):
        """
        建summary
        :param grads_and_vars: 
        :return: 
        """
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", self.cnn.accuracy)

        train_summary_op, train_summary_writer = self.create_summary_op(loss_summary, acc_summary, 'train', grad_summaries_merged)
        test_summary_op, test_summary_writer = self.create_summary_op(loss_summary, acc_summary, 'test')
        valid_summary_op, valid_summary_writer = self.create_summary_op(loss_summary, acc_summary, 'valid')

        return train_summary_op, train_summary_writer, test_summary_op, test_summary_writer, valid_summary_op, valid_summary_writer

    def create_summary_op(self, loss_summary, acc_summary, op_type, grad_summaries_merged=None):
        """
        summary op
        :param grad_summaries_merged: 
        :param loss_summary: 
        :param acc_summary: 
        :param op_type: 
        :return: 
        """
        assert op_type in ['train', 'test', 'valid'], 'op_type must be one of train, test, valid'
        summary_dir = os.path.join(self.out_dir, 'summaries', op_type)
        if grad_summaries_merged is not None:
            summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        else:
            summary_op = tf.summary.merge([loss_summary, acc_summary])
            summary_writer = tf.summary.FileWriter(summary_dir)

        return summary_op, summary_writer

    def train(self):
        """
        training
        :return: 
        """
        print('****************************starting training***********************************')
        data_train, data_valid = self.load_data()
        test_x_batch, test_y_batch = zip(*data_valid)

        # Training
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=self.train_args['allow_soft_placement'],
                                          log_device_placement=self.train_args['log_device_placement'], )
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # 模型
                self.init_model_class()

                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                lr = tf.train.exponential_decay(learning_rate=0.005, global_step=self.global_step,
                                                decay_steps=self.train_args['lr_decay_steps'],
                                                decay_rate=0.96, staircase=True, name='learn_rate')
                optimizer = tf.train.AdamOptimizer(lr)
                grads_and_vars = optimizer.compute_gradients(self.cnn.loss)
                self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

                # Output directory for models and summaries
                train_summary_op, train_summary_writer, test_summary_op, test_summary_writer, valid_summary_op, \
                valid_summary_writer = self.save_summary(grads_and_vars=grads_and_vars)

                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)
                checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
                saver = tf.train.Saver(tf.global_variables(), save_relative_paths=True, max_to_keep=10)
                self.sess.run(tf.global_variables_initializer())

                # Generate batches
                batches = data_process.batch_iter(data_train, self.train_args['batch_size'], self.train_args['num_epochs'])
                # Training loop. For each batch...
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    current_step = self.train_step(x_batch, y_batch, summary_op=train_summary_op, summary_writer=train_summary_writer)
                    if current_step % self.train_args['evaluate_every'] == 0 and current_step != 0:
                        self.dev_step(test_x_batch, test_y_batch, current_step, test_summary_writer, test_summary_op, op_type='test')

                        path = saver.save(self.sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

                train_summary_writer.close()
                test_summary_writer.close()
                valid_summary_writer.close()
                print('****************************finish training***********************************')


if __name__ == '__main__':
    trainer = CnnTrainer()
    trainer.train()
