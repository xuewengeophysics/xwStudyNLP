# -*- coding: utf-8 -*-
"""
HierarchicalAttention:
1.Word Encoder
2.Word Attention
3.Sentence Encoder
4.Sentence Attention
5.linear classifier
"""

import tensorflow as tf
import numpy as np
import math

class HierarchicalAttention(object):
    def __init__(self, config):
        self.config = config

        # placeholders for input output and dropout
        # input_x的维度是[batch_size, sequence_length]
        # num_sentences在原文中用 L 表示，num_words在原文中用 T 表示，num_words * num_sentences = sequence_length
        # input_x的维度是[batch_size, num_classes]
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.config.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, self.config.num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # word embedding
        with tf.name_scope("embedding"):
            # 将输入的序列进行拆分，拆分成num_sentences个句子
            # 得到了num_sentences个维度为[None, num_words]的张量
            input_x = tf.split(self.input_x, self.config.num_sentences, axis=1)  # 在某个维度上split
            # 矩阵拼接，得到的张量维度为[None, self.num_sentences, num_words]
            input_x = tf.stack(input_x, axis=1)
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size],
                                             initializer=tf.constant_initializer(self.config.pre_training),
                                             trainable=True) # 是否使用预训练词向量，静态(False)或动态(默认True)
            # 张量维度为[None, num_sentences, num_words, embedding_size]
            embedding_vectors = tf.nn.embedding_lookup(self.embedding, input_x)
            # 输入到word encoder层的张量的维度为[batch_size*num_sentences, num_words, embedding_size]
            self.embedding_inputs = tf.reshape(embedding_vectors, shape=[-1, self.config.num_words, self.config.embedding_size])

        # word encoder
        with tf.name_scope("word_encoder"):
            # 给定文本词向量x(it)，得到正向隐藏状态h(it)、反向隐藏状态h(it)
            # 输出的隐藏状态张量的维度为[batch_size*num_sentences, num_words, hidden_size]
            hidden_state_fw, hidden_state_bw = self.build_bidirectional_rnn(self.embedding_inputs, "word_encoder")
            # 拼接得到h(it) = [正向h(it), 反向h(it)]，维度为[batch_size*num_sentences, num_words, hidden_size*2]
            word_hidden_state = tf.concat((hidden_state_fw, hidden_state_bw), 2)

        # word attention
        with tf.name_scope("word_attention"):
            # 得到sentence_vector s(i)=sum(alpha(it)*h(it))
            # 张量维度为[batch_size*num_sentences, hidden_size*2]
            sentence_vector = self.build_attention(word_hidden_state, "word_attention")

        # sentence encoder
        with tf.name_scope("sentence_encoder"):
            # 句子级输入的是句子，reshape得到维度为[batch_size, num_sentences, hidden_size*2]的张量
            sentence_vector = tf.reshape(sentence_vector, shape=[-1, self.config.num_sentences, self.config.hidden_size * 2])
            # 给定句子级向量s(i)，得到正向隐藏状态h(i)、反向隐藏状态h(i)
            # 输出的隐藏状态张量的维度为[batch_size, num_sentences, hidden_size]
            hidden_state_fw, hidden_state_bw = self.build_bidirectional_rnn(sentence_vector, "sentence_encoder")
            # 拼接得到h(i) = [正向h(i), 反向h(i)]，维度为[batch_size, num_sentences, hidden_size*2]
            sentence_hidden_state = tf.concat((hidden_state_fw, hidden_state_bw), 2)

        # sentence attention
        with tf.name_scope("sentence_attention"):
            # 得到document_vector v=sum(alpha(i)*h(i))
            # 张量维度为[batch_size, hidden_size * 2]
            document_vector = self.build_attention(sentence_hidden_state, "sentence_attention")

        # dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(document_vector, self.dropout_keep_prob)

        # classifier
        with tf.name_scope("output"):
            # 添加一个全连接层
            self.logits = tf.layers.dense(h_drop, self.config.num_classes, name='fc2')
            # 预测类别
            self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name="prediction")

        # loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # optimizer
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度值、变量
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.gradient_clip)
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            # global_step 自动+1

        # accuracy
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.predict, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')


    def rnn_cell(self):
        """获取rnn的cell，可选RNN、LSTM、GRU"""
        if self.config.rnn_type == "Vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(self.config.hidden_size)
        elif self.config.rnn_type == "LSTM":
            return tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)
        elif self.config.rnn_type == "GRU":
            return tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
        else:
            raise Exception("rnn_type must be Vanilla、LSTM or GRU!")

    def build_bidirectional_rnn(self, inputs, name):
        with tf.variable_scope(name):
            fw_cell = self.rnn_cell()
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.config.dropout_keep_prob)
            bw_cell = self.rnn_cell()
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.config.dropout_keep_prob)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                             cell_bw=bw_cell,
                                                                             inputs=inputs,
                                                                             dtype=tf.float32)
        return output_fw, output_bw

    def build_attention(self, inputs, name):
        with tf.variable_scope(name):
            # inputs词级h(it)的维度为[batch_size*num_sentences, num_words, hidden_size*2]
            # inputs句子级h(i)的维度为[batch_size, num_sentences, hidden_size*2]
            # 采用general形式计算权重，采用单层神经网络来给出attention中的score得分
            hidden_vec = tf.layers.dense(inputs, self.config.hidden_size * 2, activation=tf.nn.tanh, name='u_hidden')
            u_key = tf.Variable(tf.truncated_normal([self.config.hidden_size * 2]), name='u_key')
            # alpha就是attention中的score得分
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(hidden_vec, u_key), axis=2, keep_dims=True), dim=1)
            # 对隐藏状态进行加权
            attention_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
        return attention_output
