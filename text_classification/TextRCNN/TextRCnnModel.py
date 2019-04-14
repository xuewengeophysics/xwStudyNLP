"""
参考资料：
[1]  https://github.com/roomylee/rcnn-text-classification
"""
import tensorflow as tf

class TextRCnn(object):
    """
    Recurrent Convolutional neural network for text classification
    1. embeddding layer,
    2. Bi-LSTM layer,
    3. max pooling,
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

        l2_loss = tf.constant(0.0)
        text_length = self._length(self.input_x)

        # 1.get emebedding of words in the sentence
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

        # 2. Bidirectional(Left&Right) Recurrent Structure
        # 输出张量 x_i = [c_l(w_i); e(w_i); c_r(w_i)], shape:[None, sequence_length, embedding_size*3]
        with tf.name_scope("bi-rnn"):
            fw_cell = self._get_cell(self.config.context_embedding_size, self.config.rnn_type)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            bw_cell = self._get_cell(self.config.context_embedding_size, self.config.rnn_type)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
            (self.output_fw, self.output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                       cell_bw=bw_cell,
                                                                                       inputs=self.embedding_words,
                                                                                       sequence_length=text_length,
                                                                                       dtype=tf.float32)

        with tf.name_scope("context"):
            shape = [tf.shape(self.output_fw)[0], 1, tf.shape(self.output_fw)[2]]
            self.context_left = tf.concat([tf.zeros(shape), self.output_fw[:, :-1]], axis=1, name="context_left")
            self.context_right = tf.concat([self.output_bw[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

        with tf.name_scope("word-representation"):
            self.x = tf.concat([self.context_left, self.embedding_words, self.context_right], axis=2, name="x")
            embedding_size = 2*self.config.context_embedding_size + self.config.word_embedding_size

        # 2.1 apply nonlinearity
        with tf.name_scope("text-representation"):
            W2 = tf.Variable(tf.random_uniform([embedding_size, self.config.hidden_size], -1.0, 1.0), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[self.config.hidden_size]), name="b2")
            self.y2 = tf.tanh(tf.einsum('aij,jk->aik', self.x, W2) + b2)

        with tf.name_scope("text-representation"):
            W2 = tf.Variable(tf.random_uniform([embedding_size, self.config.hidden_size], -1.0, 1.0), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[self.config.hidden_size]), name="b2")
            self.y2 = tf.tanh(tf.einsum('aij,jk->aik', self.x, W2) + b2)

        # 3. max-pooling
        with tf.name_scope("max-pooling"):
            self.y3 = tf.reduce_max(self.y2, axis=1)  # shape:[None, hidden_size]

        # 4. classifier
        with tf.name_scope("output"):
            # inputs: A `Tensor` of shape `[batch_size, dim]`. The forward activations of the input network.
            W4 = tf.get_variable("W4", shape=[self.config.hidden_size, self.config.num_classes],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name="b4")
            l2_loss += tf.nn.l2_loss(W4)
            l2_loss += tf.nn.l2_loss(b4)
            self.logits = tf.nn.xw_plus_b(self.y3, W4, b4, name="logits")
            self.prediction = tf.argmax(tf.nn.softmax(self.logits), 1, name='prediction')

        # loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.config.l2_reg_lambda * l2_loss

        # optimizer
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))#计算变量梯度，得到梯度值、变量
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.gradient_clip)
            #对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            #global_step 自动+1

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

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

    # Extract the output of last cell of each sequence
    # Ex) The movie is good -> length = 4
    #     output = [ [1.314, -3.32, ..., 0.98]
    #                [0.287, -0.50, ..., 1.55]
    #                [2.194, -2.12, ..., 0.63]
    #                [1.938, -1.88, ..., 1.31]
    #                [  0.0,   0.0, ...,  0.0]
    #                ...
    #                [  0.0,   0.0, ...,  0.0] ]
    #     The output we need is 4th output of cell, so extract it.
    @staticmethod
    def last_relevant(seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)
