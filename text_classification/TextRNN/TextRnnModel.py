import tensorflow as tf

class TextRnn(object):
    """
    RNN for text classification
    """
    def __init__(self, config):
        self.config = config

        # placeholders for input output and dropout
        # input_x的维度是[batch_size, sequence_length]
        # input_y的维度是[batch_size, num_classes]
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.config.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, self.config.num_classes], name='input_y')
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name='seq_length')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # word embedding
        with tf.name_scope("embedding"):
            # self.embedding = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embedding_size], -1.0, 1.0),
            #                              name="embedding", trainable=True)
            # self.embedding = tf.Variable(tf.truncated_normal([self.config.vocab_size, self.config.embedding_size],
            #                              stddev=1.0 / math.sqrt(self.config.embedding_size)),
            #                              name="embedding", trainable=True)
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size],
                                             initializer=tf.constant_initializer(self.config.pre_training),
                                             trainable=True) # 是否使用预训练词向量，静态(False)或动态(默认True)
            # 张量维度为[None, sequence_length, embedding_size]
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        # RNN layer
        with tf.name_scope("rnn_layer"):
            cells = []
            for _ in range(self.config.hidden_layer_num):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, forget_bias=0.0, state_is_tuple=True)
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
                cells.append(lstm_cell)
            cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            self.embedding_inputs = tf.nn.dropout(self.embedding_inputs, self.dropout_keep_prob)
            # sequence_length: （可选）大小为[batch_size],数据的类型是int32/int64向量
            outputs, states = tf.nn.dynamic_rnn(cell, self.embedding_inputs, dtype=tf.float32,
                                                sequence_length=self.seq_length)
            # outputs:[batch_size, sequence_length, hidden_size]
            self.outputs = tf.reduce_sum(outputs, axis=1)

        # dropout
        with tf.name_scope('dropout'):
            self.outputs_dropout = tf.nn.dropout(self.outputs, keep_prob=self.dropout_keep_prob)

        # classifier
        with tf.name_scope('output'):
            W = tf.Variable(tf.truncated_normal([self.config.hidden_size, self.config.num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='b')
            self.logits = tf.matmul(self.outputs_dropout, W) + b
            self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict')

        # loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # optimizer
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))#计算变量梯度，得到梯度值、变量
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.gradient_clip)
            #对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            #global_step 自动+1

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.predict, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
