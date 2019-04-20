"""
参考资料：
[1]  https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction
"""

import tensorflow as tf

class BiLstmAttention(object):
    """
    Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
    1. Embeddding layer,
    2. Bi-LSTM layer,
    3. Attention layer,
    4. FC layer
    5. softmax
    """
    def __init__(self, config):
        self.config = config

        # placeholders for input output and dropout
        # input_x的维度是[batch_size, sequence_length]
        # input_y的维度是[batch_size, num_classes]
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.config.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, self.config.num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # 定义l2损失
        l2_loss = tf.constant(0.0)
        text_length = self._length(self.input_x)

        # 1.get embedding of words in the sentence
        with tf.name_scope("embedding"):
            # self.embedding = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.word_embedding_size], -1.0, 1.0),
            #                              name="embedding", trainable=True)
            # self.embedding = tf.Variable(tf.truncated_normal([self.config.vocab_size, self.config.word_embedding_size],
            #                              stddev=1.0 / math.sqrt(self.config.word_embedding_size)),
            #                              name="embedding", trainable=True)
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.word_embedding_size],
                                             initializer=tf.constant_initializer(self.config.pre_training),
                                             trainable=True) # 是否使用预训练词向量，静态(False)或动态(默认True)
            # 张量维度为[None, sequence_length, word_embedding_size]
            self.embedding_words = tf.nn.embedding_lookup(self.embedding, self.input_x)

        # 2. BiLSTM layer to get high level features
        # 输出张量 x_i = [c_l(w_i); e(w_i); c_r(w_i)], shape:[None, sequence_length, embedding_size*3]
        with tf.name_scope("bi-rnn"):
            fw_cell = self._get_cell(self.config.context_embedding_size, self.config.rnn_type)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            bw_cell = self._get_cell(self.config.context_embedding_size, self.config.rnn_type)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
            # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
            # outputs是一个元组(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size]
            # fw和bw的hidden_size一样
            # states是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元组(h, c)
            self.outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                   inputs=self.embedding_words,
                                                                   sequence_length=text_length,
                                                                   dtype=tf.float32)

            # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
            self.rnn_outputs = tf.add(self.outputs[0], self.outputs[1])

        # 3. Attention layer
        # produce a weight vector,
        # and merge word-level features from each time step into a sentence-level feature vector,
        # by multiplying the weight vector
        with tf.variable_scope('attention'):
            self.attention, self.alphas = self._get_attention(self.rnn_outputs)

        # Dropout
        with tf.variable_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.attention, self.dropout_keep_prob)

        # 4. classifier
        with tf.variable_scope('output'):
            # Fully connected layer
            self.logits = tf.layers.dense(self.h_drop, self.config.num_classes,
                                          kernel_initializer=tf.keras.initializers.glorot_normal())
            self.prediction = tf.argmax(self.logits, 1, name="prediction")

        # loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.config.l2_reg_lambda * l2_loss

        # optimizer
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度值、变量
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.gradient_clip)
            # 对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            # global_step 自动+1

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    def _get_attention(self, inputs):
        # Trainable parameters
        hidden_size = inputs.shape[2].value  # inputs的维度为[batch_size, time_step, hidden_size * 2]
        # omega_T的维度为[hidden_size * 2]
        omega_T = tf.get_variable("omega_T", [hidden_size], initializer=tf.keras.initializers.glorot_normal())

        with tf.name_scope('M'):
            # M的维度为[batch_size, time_step, hidden_size * 2]
            M = tf.tanh(inputs)

        # For each of the timestamps its vector of size A from `M` is reduced with `omega` vector
        vu = tf.tensordot(M, omega_T, axes=1, name='vu')  # [batch_size, time_step] shape
        alphas = tf.nn.softmax(vu, name='alphas')  # [batch_size, time_step] shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has [batch_size, hidden_size * 2] shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        # Final output with tanh
        output = tf.tanh(output)

        return output, alphas

    @staticmethod
    def _get_cell(hidden_size, rnn_type):
        if rnn_type == "Vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        elif rnn_type == "LSTM":
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        elif rnn_type == "GRU":
            return tf.nn.rnn_cell.GRUCell(hidden_size)
        else:
            print("ERROR: '" + rnn_type + "' is a wrong cell type !!!")
            return None

    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
